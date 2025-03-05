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
from torch.utils.tensorboard import SummaryWriter

# Imports des modules personnalisés
from model_architecture import NeoBERTMoBA, create_neobert_moba_model, createneobert_moba_t4_model
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

        # Mixed precision training - MISE À JOUR pour utiliser la nouvelle API AMP
        self.scaler = torch.amp.GradScaler('cuda') if args.fp16 else None

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

        # Mixed precision forward pass - MISE À JOUR pour utiliser la nouvelle API AMP
        if self.args.fp16:
            with torch.amp.autocast('cuda'):
                outputs = self.model(**batch)
                loss = outputs['loss'] if 'loss' in outputs else None

                # Si le modèle ne calcule pas directement la perte, la calculer ici
                if loss is None:
                    prediction_logits = outputs['prediction_logits']
                    labels = batch['labels']
                    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                    active_loss = labels.view(-1) != -100
                    active_logits = prediction_logits.view(-1, prediction_logits.size(-1))[active_loss]
                    active_labels = labels.view(-1)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
        else:
            outputs = self.model(**batch)
            loss = outputs['loss'] if 'loss' in outputs else None

            # Si le modèle ne calcule pas directement la perte, la calculer ici
            if loss is None:
                prediction_logits = outputs['prediction_logits']
                labels = batch['labels']
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                active_loss = labels.view(-1) != -100
                active_logits = prediction_logits.view(-1, prediction_logits.size(-1))[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)

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

                # Ajouter des métriques supplémentaires pour le monitoring MOBA
                model = self.model.module if hasattr(self.model, "module") else self.model
                if hasattr(model, "config") and "hybrid_layer_count" in model.config:
                    self.tb_writer.add_scalar('moba/hybrid_layer_count',
                                             model.config["hybrid_layer_count"],
                                             self.global_step)

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
                loss = outputs['loss'] if 'loss' in outputs else None

                # Si le modèle ne calcule pas directement la perte, la calculer ici
                if loss is None:
                    prediction_logits = outputs['prediction_logits']
                    labels = batch['labels']
                    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                    active_loss = labels.view(-1) != -100
                    active_logits = prediction_logits.view(-1, prediction_logits.size(-1))[active_loss]
                    active_labels = labels.view(-1)[active_loss]
                    loss = loss_fct(active_logits, active_labels)

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


def train_t4_test(data_prep, model_config=None):
    """Version simplifiée pour tester l'architecture sur T4 avec un petit dataset"""

    # Configuration pour le test
    training_args = TrainingArguments(
        output_dir="./outputs/t4_test",
        logging_dir="./logs/t4_test",
        training_phase=1,

        # Hyperparamètres adaptés pour tests
        learning_rate=3e-4,
        warmup_steps=100,
        weight_decay=0.01,
        max_grad_norm=1.0,

        # Taille de lot et accumulation adaptées aux T4
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,

        # Durée d'entraînement très courte pour les tests
        max_steps=500,

        # Logging et checkpointing plus fréquents
        logging_steps=10,
        save_steps=100,
        eval_steps=100,

        # Activer MOBA après 200 steps
        moba_activation_step=200,

        # Efficacité
        fp16=True,
    )

    # Créer la version T4 du modèle
    if model_config is None:
        model_config = {
            "vocab_size": 30000,
            "hidden_size": 384,  # Réduit encore plus pour les tests
            "num_hidden_layers": 8,  # Réduit encore plus pour les tests
            "num_attention_heads": 8,
            "max_position_embeddings": 4096,  # Séquences plus courtes pour les tests
            "moba_block_size": 512,
            "moba_top_k": 3,
            "hybrid_layer_count": 2,
        }

    model = create_neobert_moba_model(model_config)

    # Utiliser un sous-ensemble des données pour les tests
    train_dataloader = data_prep.get_dataloaders_for_phase(1, ddp=training_args.local_rank != -1)["main_len1024_bs8"]

    # Créer un petit dataloader d'évaluation pour les tests
    eval_dataloader = None  # Optionnel: ajouter un petit jeu d'évaluation

    # Créer le trainer
    trainer = NeoBERTMoBATrainer(
        model=model,
        args=training_args,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader
    )

    # Entraîner le modèle
    global_step, avg_loss = trainer.train()

    logger.info(f"Test terminé. Steps: {global_step}, Perte moyenne: {avg_loss}")

    return model


def main():
    """Fonction principale qui exécute l'ensemble du protocole d'entraînement"""

    parser = argparse.ArgumentParser(description="Test d'entraînement NeoBERT-MOBA sur T4")
    parser.add_argument("--test", action="store_true", help="Lancer un entraînement de test rapide")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3], help="Phase d'entraînement")
    parser.add_argument("--continue_from_checkpoint", type=str, default=None, help="Chemin vers un checkpoint")
    parser.add_argument("--local_rank", type=int, default=-1, help="Rang local pour DDP")
    parser.add_argument("--data_dir", type=str, default="./data", help="Répertoire des données")

    args = parser.parse_args()

    # Priorité à la variable d'environnement LOCAL_RANK (nécessaire pour torchrun)
    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])

    # Configuration pour 2 GPUs T4
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    # Initialisation pour entraînement distribué
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")

    # Afficher les informations sur les GPUs
    if args.local_rank in [-1, 0]:
        num_gpus = torch.cuda.device_count()
        print(f"Nombre de GPUs disponibles: {num_gpus}")

        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"Mémoire totale: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

    # Configuration du modèle réduit pour T4
    model_config = {
        "vocab_size": 30000,
        "hidden_size": 384,  # Réduit pour tests sur T4
        "num_hidden_layers": 8,  # Réduit pour tests
        "num_attention_heads": 8,
        "max_position_embeddings": 4096,  # Plus petit pour tests
        "moba_block_size": 512,
        "moba_top_k": 3,
        "hybrid_layer_count": 2,
    }

    # Préparation des données - taille réduite pour tests
    data_prep = FrenchDataPreparation(
        base_dir=args.data_dir,
        vocab_size=model_config["vocab_size"],
        max_length=model_config["max_position_embeddings"],
        mlm_probability=0.15
    )

    # Exécuter un entraînement de test rapide
    if args.test:
        model = train_t4_test(data_prep, model_config)
        logger.info("Test d'entraînement terminé avec succès!")
    else:
        # Exécuter selon la phase demandée
        if args.phase == 1:
            # Configuration pour Phase 1 - version T4
            training_args = TrainingArguments(
                output_dir="./outputs/phase1",
                logging_dir="./logs/phase1",
                training_phase=1,

                # Hyperparamètres spécifiques
                learning_rate=5e-4,
                warmup_steps=500,
                weight_decay=0.1,
                max_grad_norm=1.0,

                # Taille de lot adaptée aux T4
                per_device_train_batch_size=6,
                gradient_accumulation_steps=4,

                # Durée d'entraînement
                max_steps=5000,

                # Logging et checkpointing
                logging_steps=50,
                save_steps=500,
                eval_steps=500,

                # Pas de MOBA dans la Phase 1
                moba_activation_step=None,

                # Efficacité
                fp16=True,

                # DDP
                local_rank=args.local_rank,
            )

            model = create_neobert_moba_model(model_config)

            # Préparation des données
            train_dataloader = data_prep.get_dataloaders_for_phase(1, ddp=training_args.local_rank != -1)["main_len1024_bs8"]

            # Créer le trainer
            trainer = NeoBERTMoBATrainer(
                model=model,
                args=training_args,
                train_dataloader=train_dataloader,
                eval_dataloader=None
            )

            # Entraîner le modèle
            global_step, avg_loss = trainer.train()

            logger.info(f"Phase 1 terminée. Steps: {global_step}, Perte moyenne: {avg_loss}")

    logger.info("Entraînement terminé avec succès!")


if __name__ == "__main__":
    main()