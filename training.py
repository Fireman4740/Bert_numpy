import os
import time
import math
import json
import logging
import argparse
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

# Supposez que ces imports proviennent de vos modules précédents
from model_architecture import NeoBERTMoBA, create_neobert_moba_model, create_neobert_moba_t4_model
from data_prep_french import FrenchDataPreparation

# Configuration logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingArguments:
    """Arguments d'entraînement pour NeoBERT-MOBA"""

    # Chemins et organisation
    output_dir: str = field(default="./outputs")
    logging_dir: str = field(default=None)
    checkpoint_dir: str = field(default=None)

    # Hyperparamètres généraux
    learning_rate: float = field(default=6e-4)
    weight_decay: float = field(default=0.1)
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.95)
    adam_epsilon: float = field(default=1e-8)
    max_grad_norm: float = field(default=1.0)

    # Paramètres d'entraînement
    num_train_epochs: int = field(default=1)
    max_steps: int = field(default=-1)
    warmup_steps: int = field(default=2000)
    warmup_ratio: float = field(default=0.0)

    # Taille de lot et accumulation de gradient
    per_device_train_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=1)

    # Efficacité
    fp16: bool = field(default=True)
    fp16_opt_level: str = field(default="O1")

    # Logging et évaluation
    logging_steps: int = field(default=100)
    eval_steps: int = field(default=1000)
    save_steps: int = field(default=5000)
    save_total_limit: Optional[int] = field(default=3)

    # Distributed training
    local_rank: int = field(default=-1)
    world_size: int = field(default=1)

    # MOBA spécifique
    moba_activation_step: Optional[int] = field(default=None)
    hybrid_layer_transition_steps: List[int] = field(default_factory=list)

    # Phase spécifique
    training_phase: int = field(default=1)

    def __post_init__(self):
        if self.logging_dir is None:
            self.logging_dir = os.path.join(self.output_dir, "logs")
        if self.checkpoint_dir is None:
            self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.logging_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)


class NeoBERTMoBATrainer:
    """Trainer pour l'entraînement du modèle NeoBERT-MOBA"""

    def __init__(
        self,
        model: NeoBERTMoBA,
        args: TrainingArguments,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        lr_scheduler=None,
    ):
        self.model = model
        self.args = args
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        # Distributed training setup
        self.is_distributed = args.local_rank != -1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.is_distributed:
            if args.local_rank == 0:
                logger.info(f"Distributed training enabled, world size: {args.world_size}")
            self.device = torch.device(f"cuda:{args.local_rank}")
            torch.cuda.set_device(args.local_rank)
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")

        self.model = self.model.to(self.device)

        # Activer le gradient checkpointing pour économiser de la mémoire
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        else:
            logger.info("Le gradient checkpointing n'est pas disponible sur ce modèle")

        # Wrap model in DDP
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True
            )

        # Prepare optimizer
        self.optimizer = self._create_optimizer()

        # LR Scheduler
        self.lr_scheduler = lr_scheduler
        if self.lr_scheduler is None:
            self.lr_scheduler = self._create_scheduler()

        # Mixed precision training
        self.scaler = GradScaler() if args.fp16 else None

        # Tensorboard logging
        self.tb_writer = None
        if args.local_rank in [-1, 0]:
            self.tb_writer = SummaryWriter(log_dir=args.logging_dir)

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')

        # Loss tracking
        self.tr_loss = 0.0
        self.logging_loss = 0.0

    def _create_optimizer(self):
        """Crée l'optimiseur pour l'entraînement"""
        no_decay = ["bias", "norm", "LayerNorm.weight", "RMSNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        return torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon
        )

    def _create_scheduler(self):
        """Crée le scheduler de learning rate avec warmup"""
        num_training_steps = self._get_total_training_steps()

        # Calculer les steps de warmup
        if self.args.warmup_ratio > 0:
            warmup_steps = int(num_training_steps * self.args.warmup_ratio)
        else:
            warmup_steps = self.args.warmup_steps

        logger.info(f"Warmup steps: {warmup_steps}")
        logger.info(f"Total training steps: {num_training_steps}")

        def lr_lambda(current_step: int):
            # Warmup phase
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            # Decay phase (cosine decay)
            progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _get_total_training_steps(self):
        """Calcule le nombre total de steps d'entraînement"""
        if self.args.max_steps > 0:
            return self.args.max_steps

        # Calcul basé sur le nombre d'époques
        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.args.gradient_accumulation_steps
        )

        return num_update_steps_per_epoch * self.args.num_train_epochs

    def train(self):
        """Entraîne le modèle selon les paramètres spécifiés"""
        num_training_steps = self._get_total_training_steps()

        logger.info("***** Démarrage de l'entraînement *****")
        logger.info(f"  Phase d'entraînement = {self.args.training_phase}")
        logger.info(f"  Nombre d'époques = {self.args.num_train_epochs}")
        logger.info(f"  Steps d'entraînement = {num_training_steps}")
        logger.info(f"  Gradient accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Batch size effectif = {self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps * (self.args.world_size if self.is_distributed else 1)}")

        self.model.train()
        self.model.zero_grad()

        train_iterator = range(self.args.num_train_epochs)

        for epoch in train_iterator:
            self.epoch = epoch
            epoch_iterator = self.train_dataloader

            for step, batch in enumerate(epoch_iterator):
                loss = self._training_step(batch)

                # Accumulate gradients if needed
                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    step + 1 == len(epoch_iterator)
                ):
                    # Clip gradients
                    if self.args.fp16:
                        self.scaler.unscale_(self.optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.max_grad_norm
                    )

                    # Update parameters
                    if self.args.fp16:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.lr_scheduler.step()
                    self.model.zero_grad()
                    self.global_step += 1

                    # Logging
                    if self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0:
                        self._log_metrics(loss)

                    # Evaluation
                    if self.eval_dataloader is not None and self.args.eval_steps > 0 and self.global_step % self.args.eval_steps == 0:
                        self._evaluate()

                    # Checkpointing
                    if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
                        self._save_checkpoint()

                # Apply MOBA activation at specific step if configured
                if self.args.moba_activation_step is not None and self.global_step == self.args.moba_activation_step:
                    self._activate_moba()

                # Apply hybrid layer transitions at specific steps if configured
                if self.args.hybrid_layer_transition_steps:
                    for step_idx, hybrid_count in enumerate(self.args.hybrid_layer_transition_steps):
                        if self.global_step == step_idx:
                            self._set_hybrid_layers(hybrid_count)

                # Check if we've reached max steps
                if 0 < self.args.max_steps <= self.global_step:
                    epoch_iterator.close()
                    break

            # Save checkpoint at the end of each epoch
            self._save_checkpoint(epoch=epoch)

            if 0 < self.args.max_steps <= self.global_step:
                train_iterator.close()
                break

        # Final evaluation
        if self.eval_dataloader is not None:
            self._evaluate()

        # Final checkpoint
        self._save_checkpoint(final=True)

        # Close tensorboard writer
        if self.tb_writer:
            self.tb_writer.close()

        logger.info("***** Entraînement terminé *****")

        return self.global_step, self.tr_loss / self.global_step

    def _training_step(self, batch):
        """Exécute un pas d'entraînement"""
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Mixed precision forward pass
        if self.args.fp16:
            with autocast():
                outputs = self.model(**batch)
                loss = outputs['loss'] if 'loss' in outputs else outputs['prediction_logits'].view(-1, outputs['prediction_logits'].size(-1)).gather(1, batch['labels'].view(-1, 1)).mean()
        else:
            outputs = self.model(**batch)
            loss = outputs['loss'] if 'loss' in outputs else outputs['prediction_logits'].view(-1, outputs['prediction_logits'].size(-1)).gather(1, batch['labels'].view(-1, 1)).mean()

        # Scale loss for gradient accumulation
        loss = loss / self.args.gradient_accumulation_steps

        # Backward pass
        if self.args.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update tr_loss for logging
        self.tr_loss += loss.item() * self.args.gradient_accumulation_steps

        return loss.item()

    def _log_metrics(self, loss):
        """Log les métriques sur Tensorboard"""
        if self.args.local_rank in [-1, 0]:
            # Calculer la perte moyennée depuis le dernier log
            avg_loss = (self.tr_loss - self.logging_loss) / self.args.logging_steps
            self.logging_loss = self.tr_loss

            # Log sur Tensorboard
            if self.tb_writer:
                self.tb_writer.add_scalar('lr', self.lr_scheduler.get_last_lr()[0], self.global_step)
                self.tb_writer.add_scalar('loss', avg_loss, self.global_step)

            # Log dans la console
            logger.info(f"Step {self.global_step}: loss = {avg_loss:.4f}, lr = {self.lr_scheduler.get_last_lr()[0]:.8f}")

    def _evaluate(self):
        """Évalue le modèle sur l'ensemble d'évaluation"""
        if self.eval_dataloader is None:
            return

        logger.info("***** Évaluation *****")
        self.model.eval()
        eval_loss = 0.0
        nb_eval_steps = 0

        for batch in self.eval_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = self.model(**batch)
                loss = outputs['loss'] if 'loss' in outputs else outputs['prediction_logits'].view(-1, outputs['prediction_logits'].size(-1)).gather(1, batch['labels'].view(-1, 1)).mean()

            eval_loss += loss.item()
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps

        # Log les résultats
        if self.args.local_rank in [-1, 0]:
            logger.info(f"Eval loss: {eval_loss:.4f}")

            if self.tb_writer:
                self.tb_writer.add_scalar('eval_loss', eval_loss, self.global_step)

            # Sauvegarder le meilleur modèle
            if eval_loss < self.best_eval_loss:
                self.best_eval_loss = eval_loss
                self._save_checkpoint(best=True)

        self.model.train()

    def _save_checkpoint(self, epoch=None, best=False, final=False):
        """Sauvegarde un checkpoint du modèle"""
        if self.args.local_rank in [-1, 0]:
            # Créer le nom du checkpoint
            prefix = "best" if best else "final" if final else f"checkpoint-{self.global_step}"
            if epoch is not None:
                prefix = f"{prefix}-epoch-{epoch}"

            # Créer le répertoire de sauvegarde
            checkpoint_dir = os.path.join(self.args.checkpoint_dir, prefix)
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Sauvegarder le modèle
            logger.info(f"Sauvegarde du modèle dans {checkpoint_dir}")

            # Sauvegarde du modèle (gestion pour DDP)
            model_to_save = self.model.module if hasattr(self.model, "module") else self.model
            torch.save(model_to_save.state_dict(), os.path.join(checkpoint_dir, "pytorch_model.bin"))

            # Sauvegarder la configuration
            with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
                json.dump(model_to_save.config, f)

            # Sauvegarder l'état d'entraînement
            torch.save(
                {
                    "optimizer": self.optimizer.state_dict(),
                    "lr_scheduler": self.lr_scheduler.state_dict(),
                    "scaler": self.scaler.state_dict() if self.scaler else None,
                    "epoch": self.epoch,
                    "global_step": self.global_step,
                    "best_eval_loss": self.best_eval_loss,
                },
                os.path.join(checkpoint_dir, "optimizer.pt"),
            )

            # Nettoyer les checkpoints anciens
            if self.args.save_total_limit > 0:
                self._rotate_checkpoints()

    def _rotate_checkpoints(self):
        """Supprime les checkpoints les plus anciens si la limite est atteinte"""
        # Chercher tous les checkpoints
        glob_checkpoints = [
            str(x) for x in os.listdir(self.args.checkpoint_dir)
            if x.startswith("checkpoint-") and os.path.isdir(os.path.join(self.args.checkpoint_dir, x))
        ]

        if len(glob_checkpoints) <= self.args.save_total_limit:
            return

        # Trier par étape (plus récent en dernier)
        ordering_and_checkpoint = []
        for checkpoint in glob_checkpoints:
            try:
                step = int(checkpoint.split("-")[1])
                ordering_and_checkpoint.append((step, checkpoint))
            except:
                continue

        # Garder uniquement les N plus récents
        checkpoints_sorted = sorted(ordering_and_checkpoint)
        checkpoints_sorted = checkpoints_sorted[:(len(checkpoints_sorted) - self.args.save_total_limit)]

        # Supprimer les plus anciens
        for _, checkpoint in checkpoints_sorted:
            checkpoint_dir = os.path.join(self.args.checkpoint_dir, checkpoint)
            logger.info(f"Suppression du checkpoint obsolète {checkpoint_dir}")
            for filename in os.listdir(checkpoint_dir):
                os.remove(os.path.join(checkpoint_dir, filename))
            os.rmdir(checkpoint_dir)

    def _activate_moba(self):
        """Active le mode MoBA pour les couches configurées"""
        logger.info("Activation du mécanisme MOBA")

        # Accéder au modèle (gérer DDP)
        model = self.model.module if hasattr(self.model, "module") else self.model

        # Configurer les couches selon le nombre de couches hybrides défini
        hybrid_count = model.config.get("hybrid_layer_count", 3)
        model.switch_to_hybrid_mode(hybrid_count)

    def _set_hybrid_layers(self, hybrid_count):
        """Définit le nombre de couches hybrides (utilisant l'attention complète)"""
        logger.info(f"Configuration des couches hybrides: {hybrid_count} couches en attention complète")

        # Accéder au modèle (gérer DDP)
        model = self.model.module if hasattr(self.model, "module") else self.model

        # Configurer les couches
        model.switch_to_hybrid_mode(hybrid_count)


def train_phase1(data_prep, model_config, training_args=None):
    """Phase 1: Pré-entraînement initial (1024 tokens)"""

    # Configuration pour Phase 1
    if training_args is None:
        training_args = TrainingArguments(
            output_dir="./outputs/phase1",
            logging_dir="./logs/phase1",
            training_phase=1,

            # Hyperparamètres spécifiques à Phase 1
            learning_rate=6e-4,
            warmup_steps=2000,
            weight_decay=0.1,
            max_grad_norm=1.0,

            # Taille de lot et accumulation
            per_device_train_batch_size=32,
            gradient_accumulation_steps=8,

            # Durée d'entraînement
            max_steps=1000000,  # 1M steps comme indiqué dans le protocole

            # Logging et checkpointing
            logging_steps=100,
            save_steps=5000,
            eval_steps=5000,

            # Pas de MOBA dans la Phase 1
            moba_activation_step=None,

            # Efficacité
            fp16=True,
        )

    # Créer/charger le modèle
    model = create_neobert_moba_model(model_config)

    # Préparation des données
    train_dataloader = data_prep.get_dataloaders_for_phase(1, ddp=training_args.local_rank != -1)["main_len1024_bs32"]

    # Créer un dataloader d'évaluation si nécessaire
    eval_dataloader = None

    # Créer le trainer
    trainer = NeoBERTMoBATrainer(
        model=model,
        args=training_args,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader
    )

    # Entraîner le modèle
    global_step, avg_loss = trainer.train()

    logger.info(f"Phase 1 terminée. Steps: {global_step}, Perte moyenne: {avg_loss}")

    return model


def train_phase2(data_prep, model_config, model=None, training_args=None):
    """Phase 2: Extension du contexte (4096 tokens)"""

    # Configuration pour Phase 2
    if training_args is None:
        training_args = TrainingArguments(
            output_dir="./outputs/phase2",
            logging_dir="./logs/phase2",
            training_phase=2,

            # Hyperparamètres spécifiques à Phase 2
            learning_rate=3e-4,  # Learning rate plus faible pour fine-tuning
            warmup_steps=1000,
            weight_decay=0.1,
            max_grad_norm=1.0,

            # Taille de lot et accumulation
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,

            # Durée d'entraînement
            max_steps=50000,  # 50K steps comme indiqué dans le protocole

            # Logging et checkpointing
            logging_steps=100,
            save_steps=1000,
            eval_steps=1000,

            # Pas de MOBA dans la Phase 2
            moba_activation_step=None,

            # Efficacité
            fp16=True,
        )

    # Charger le modèle de la Phase 1 si non fourni
    if model is None:
        # Charger le dernier checkpoint de la Phase 1
        checkpoint_path = "./outputs/phase1/checkpoints/final"
        model = create_neobert_moba_model(model_config)
        model.load_state_dict(torch.load(os.path.join(checkpoint_path, "pytorch_model.bin")))
        logger.info(f"Modèle chargé depuis le checkpoint: {checkpoint_path}")

    # Préparation des données pour Phase 2
    train_dataloader = data_prep.get_dataloaders_for_phase(2, ddp=training_args.local_rank != -1)

    # Créer un dataloader d'évaluation si nécessaire
    eval_dataloader = None

    # Créer le trainer
    trainer = NeoBERTMoBATrainer(
        model=model,
        args=training_args,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader
    )

    # Entraîner le modèle
    global_step, avg_loss = trainer.train()

    logger.info(f"Phase 2 terminée. Steps: {global_step}, Perte moyenne: {avg_loss}")

    return model


def train_phase3(data_prep, model_config, model=None, training_args=None):
    """Phase 3: Activation MOBA (jusqu'à 32K tokens)"""

    # Configuration pour Phase 3
    if training_args is None:
        training_args = TrainingArguments(
            output_dir="./outputs/phase3",
            logging_dir="./logs/phase3",
            training_phase=3,

            # Hyperparamètres spécifiques à Phase 3
            learning_rate=2e-4,  # Learning rate encore plus faible
            warmup_steps=500,
            weight_decay=0.1,
            max_grad_norm=1.0,

            # Taille de lot et accumulation
            per_device_train_batch_size=1,  # Séquences très longues
            gradient_accumulation_steps=8,

            # Durée d'entraînement
            max_steps=50000,  # 50K steps comme indiqué dans le protocole

            # Activer MOBA après le premier step
            moba_activation_step=1,

            # Transitions hybrides
            # À 90% des steps (45000), passer à l'attention complète sur toutes les couches
            hybrid_layer_transition_steps=[model_config["num_hidden_layers"]] * 45000 + [0] * 5000,

            # Logging et checkpointing
            logging_steps=50,
            save_steps=1000,
            eval_steps=1000,

            # Efficacité
            fp16=True,
        )

    # Charger le modèle de la Phase 2 si non fourni
    if model is None:
        # Charger le dernier checkpoint de la Phase 2
        checkpoint_path = "./outputs/phase2/checkpoints/final"
        model = create_neobert_moba_model(model_config)
        model.load_state_dict(torch.load(os.path.join(checkpoint_path, "pytorch_model.bin")))
        logger.info(f"Modèle chargé depuis le checkpoint: {checkpoint_path}")

    # Préparation des données pour Phase 3
    train_dataloader = data_prep.get_dataloaders_for_phase(3, ddp=training_args.local_rank != -1)

    # Créer un dataloader d'évaluation si nécessaire
    eval_dataloader = None

    # Créer le trainer
    trainer = NeoBERTMoBATrainer(
        model=model,
        args=training_args,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader
    )

    # Entraîner le modèle
    global_step, avg_loss = trainer.train()

    logger.info(f"Phase 3 terminée. Steps: {global_step}, Perte moyenne: {avg_loss}")

    return model


def train_phase1_t4(data_prep, model_config, training_args=None):
    """Phase 1: Pré-entraînement initial adapté pour T4"""

    # Configuration pour Phase 1
    if training_args is None:
        training_args = TrainingArguments(
            output_dir="./outputs/phase1",
            logging_dir="./logs/phase1",
            training_phase=1,

            # Hyperparamètres spécifiques à Phase 1
            learning_rate=6e-4,
            warmup_steps=1000,  # Réduit de 2000 à 1000
            weight_decay=0.1,
            max_grad_norm=1.0,

            # Taille de lot et accumulation adaptées aux T4
            per_device_train_batch_size=8,  # Réduit de 32 à 8
            gradient_accumulation_steps=4,  # Réduit de 8 à 4

            # Durée d'entraînement
            max_steps=5000,  # Réduit pour les tests

            # Logging et checkpointing
            logging_steps=50,  # Plus fréquent pour les tests
            save_steps=1000,  # Plus fréquent pour les tests
            eval_steps=1000,

            # Efficacité
            fp16=True,
        )

    # Créer la version T4 du modèle
    model = create_neobert_moba_t4_model() if model_config is None else create_neobert_moba_model(model_config)

    # Préparation des données
    train_dataloader = data_prep.get_dataloaders_for_phase(1, ddp=training_args.local_rank != -1)["main_len1024_bs8"]  # Ajuster la taille du batch

    # Créer un dataloader d'évaluation si nécessaire
    eval_dataloader = None

    # Créer le trainer
    trainer = NeoBERTMoBATrainer(
        model=model,
        args=training_args,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader
    )

    # Entraîner le modèle
    global_step, avg_loss = trainer.train()

    logger.info(f"Phase 1 terminée. Steps: {global_step}, Perte moyenne: {avg_loss}")

    return model


def train_phase2_t4(data_prep, model_config, model=None, training_args=None):
    """Phase 2: Extension du contexte adaptée pour T4"""

    # Configuration pour Phase 2
    if training_args is None:
        training_args = TrainingArguments(
            output_dir="./outputs/phase2",
            logging_dir="./logs/phase2",
            training_phase=2,

            # Hyperparamètres spécifiques à Phase 2
            learning_rate=3e-4,
            warmup_steps=500,  # Réduit de 1000 à 500
            weight_decay=0.1,
            max_grad_norm=1.0,

            # Taille de lot et accumulation adaptées aux T4
            per_device_train_batch_size=2,  # Réduit de 4 à 2
            gradient_accumulation_steps=4,  # Maintenu à 4

            # Durée d'entraînement
            max_steps=2000,  # Réduit pour les tests

            # Logging et checkpointing
            logging_steps=50,
            save_steps=500,
            eval_steps=500,

            # Pas de MOBA dans la Phase 2
            moba_activation_step=None,

            # Efficacité
            fp16=True,
        )

    # Charger le modèle de la Phase 1 si non fourni
    if model is None:
        # Charger le dernier checkpoint de la Phase 1
        checkpoint_path = "./outputs/phase1/checkpoints/final"
        model = create_neobert_moba_t4_model() if model_config is None else create_neobert_moba_model(model_config)
        model.load_state_dict(torch.load(os.path.join(checkpoint_path, "pytorch_model.bin")))
        logger.info(f"Modèle chargé depuis le checkpoint: {checkpoint_path}")

    # Préparation des données pour Phase 2 avec T4
    train_dataloader = data_prep.get_dataloaders_for_phase(2, ddp=training_args.local_rank != -1)

    # Créer un dataloader d'évaluation si nécessaire
    eval_dataloader = None

    # Créer le trainer
    trainer = NeoBERTMoBATrainer(
        model=model,
        args=training_args,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader
    )

    # Entraîner le modèle
    global_step, avg_loss = trainer.train()

    logger.info(f"Phase 2 terminée. Steps: {global_step}, Perte moyenne: {avg_loss}")

    return model


def train_phase3_t4(data_prep, model_config, model=None, training_args=None):
    """Phase 3: Activation MOBA adaptée pour T4"""

    # Configuration pour Phase 3
    if training_args is None:
        training_args = TrainingArguments(
            output_dir="./outputs/phase3",
            logging_dir="./logs/phase3",
            training_phase=3,

            # Hyperparamètres spécifiques à Phase 3
            learning_rate=2e-4,
            warmup_steps=200,  # Réduit de 500 à 200
            weight_decay=0.1,
            max_grad_norm=1.0,

            # Taille de lot et accumulation adaptées aux T4
            per_device_train_batch_size=1,  # Maintenu à 1 (séquences très longues)
            gradient_accumulation_steps=8,  # Maintenu à 8

            # Durée d'entraînement
            max_steps=1000,  # Réduit pour les tests

            # Activer MOBA après le premier step
            moba_activation_step=1,

            # Transitions hybrides
            hybrid_layer_transition_steps=[model_config["num_hidden_layers"]] * 900 + [0] * 100,

            # Logging et checkpointing
            logging_steps=25,
            save_steps=250,
            eval_steps=500,

            # Efficacité
            fp16=True,
        )

    # Charger le modèle de la Phase 2 si non fourni
    if model is None:
        # Charger le dernier checkpoint de la Phase 2
        checkpoint_path = "./outputs/phase2/checkpoints/final"
        model = create_neobert_moba_t4_model() if model_config is None else create_neobert_moba_model(model_config)
        model.load_state_dict(torch.load(os.path.join(checkpoint_path, "pytorch_model.bin")))
        logger.info(f"Modèle chargé depuis le checkpoint: {checkpoint_path}")

    # Préparation des données pour Phase 3 avec T4
    train_dataloader = data_prep.get_dataloaders_for_phase(3, ddp=training_args.local_rank != -1)

    # Créer un dataloader d'évaluation si nécessaire
    eval_dataloader = None

    # Créer le trainer
    trainer = NeoBERTMoBATrainer(
        model=model,
        args=training_args,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader
    )

    # Entraîner le modèle
    global_step, avg_loss = trainer.train()

    logger.info(f"Phase 3 terminée. Steps: {global_step}, Perte moyenne: {avg_loss}")

    return model


def configure_for_kaggle():
    """Configure l'entraînement pour l'environnement Kaggle avec 2 T4"""
    import os

    # Détecter si nous sommes dans Kaggle
    is_kaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')

    if is_kaggle:
        # Configuration pour 2 T4
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

        # Vérifier les GPUs disponibles
        import torch
        num_gpus = torch.cuda.device_count()
        print(f"Nombre de GPUs disponibles dans Kaggle: {num_gpus}")

        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"Mémoire totale: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

        # Retourner une configuration adaptée
        return {
            "max_seq_length": 16384,  # Longueur maximale pour T4
            "per_gpu_batch_size": {
                "phase1": 8,
                "phase2": 2,
                "phase3": 1
            },
            "gradient_accumulation": {
                "phase1": 4,
                "phase2": 4,
                "phase3": 8
            },
            "model_config": {
                "hidden_size": 512,
                "num_hidden_layers": 16,
                "num_attention_heads": 8,
                "max_position_embeddings": 16384
            }
        }
    else:
        print("Non exécuté dans un environnement Kaggle")
        return None


def main():
    """Fonction principale qui exécute l'ensemble du protocole d'entraînement"""

    parser = argparse.ArgumentParser(description="Entraînement de NeoBERT-MOBA")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3], help="Phase d'entraînement à exécuter")
    parser.add_argument("--continue_from_checkpoint", type=str, default=None, help="Chemin vers un checkpoint pour continuer l'entraînement")
    parser.add_argument("--local_rank", type=int, default=-1, help="Rang local pour l'entraînement distribué")
    parser.add_argument("--data_dir", type=str, default="./data", help="Répertoire des données")
    parser.add_argument("--t4_mode", action="store_true", help="Activer le mode optimisé pour GPUs T4")
    parser.add_argument("--kaggle", action="store_true", help="Optimiser pour l'environnement Kaggle")

    args = parser.parse_args()

    # Configuration pour Kaggle si demandé
    kaggle_config = None
    if args.kaggle:
        kaggle_config = configure_for_kaggle()
        if kaggle_config:
            args.t4_mode = True  # Activer automatiquement le mode T4 dans Kaggle

    # Initialisation pour distributed training
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")

    # Configuration du modèle
    model_config = None
    if args.t4_mode:
        model_config = {
            # Configuration NeoBERT réduite pour T4
            "vocab_size": 30000,
            "hidden_size": 512,
            "num_hidden_layers": 16,
            "num_attention_heads": 8,
            "max_position_embeddings": 16384,

            # Configuration MOBA
            "moba_block_size": 512,
            "moba_top_k": 3,
            "hybrid_layer_count": 2,
        }
        # Si nous avons une configuration Kaggle, l'utiliser
        if kaggle_config:
            model_config.update(kaggle_config["model_config"])
    else:
        model_config = {
            # Configuration NeoBERT
            "vocab_size": 30000,
            "hidden_size": 768,
            "num_hidden_layers": 28,
            "num_attention_heads": 12,
            "max_position_embeddings": 32768,

            # Configuration MOBA
            "moba_block_size": 512,
            "moba_top_k": 3,
            "hybrid_layer_count": 3,
        }

    # Préparation des données
    data_prep = FrenchDataPreparation(
        base_dir=args.data_dir,
        vocab_size=model_config["vocab_size"],
        max_length=model_config["max_position_embeddings"],
        mlm_probability=0.15
    )

    # Initialiser le modèle si on ne continue pas depuis un checkpoint
    model = None
    if args.continue_from_checkpoint:
        if args.t4_mode:
            model = create_neobert_moba_t4_model()
        else:
            model = create_neobert_moba_model(model_config)
        model.load_state_dict(torch.load(os.path.join(args.continue_from_checkpoint, "pytorch_model.bin")))
        logger.info(f"Modèle chargé depuis le checkpoint: {args.continue_from_checkpoint}")

    # Exécuter la phase d'entraînement spécifiée avec la version appropriée
    if args.t4_mode:
        # Utiliser les versions T4 des fonctions d'entraînement
        if args.phase == 1:
            model = train_phase1_t4(data_prep, model_config, model)
        elif args.phase == 2:
            model = train_phase2_t4(data_prep, model_config, model)
        elif args.phase == 3:
            model = train_phase3_t4(data_prep, model_config, model)
    else:
        # Version standard
        if args.phase == 1:
            model = train_phase1(data_prep, model_config, model)
        elif args.phase == 2:
            model = train_phase2(data_prep, model_config, model)
        elif args.phase == 3:
            model = train_phase3(data_prep, model_config, model)

    logger.info("Entraînement terminé avec succès!")


if __name__ == "__main__":
    main()
    
    
# Le script peut être exécuté avec différentes options:

# --phase: Sélectionne la phase d'entraînement (1, 2 ou 3)
# --continue_from_checkpoint: Reprend l'entraînement depuis un checkpoint existant
# --local_rank: Rang local pour l'entraînement distribué
# --data_dir: Répertoire des données prétraitées