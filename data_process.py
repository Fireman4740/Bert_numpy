import os
import json
import random
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from datasets import load_dataset, concatenate_datasets
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers, processors
from transformers import PreTrainedTokenizerFast
from typing import Dict, List, Optional, Tuple, Union
import multiprocessing
from tqdm import tqdm

# Configuration du logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class DataPreparation:
    def __init__(
        self,
        base_dir: str = "./data",
        vocab_size: int = 30000,
        max_length: int = 32768,
        mlm_probability: float = 0.15,
        seed: int = 42
    ):
        self.base_dir = base_dir
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        self.seed = seed
        self.tokenizer = None

        # Créer le répertoire de base s'il n'existe pas
        os.makedirs(base_dir, exist_ok=True)

        # Fixer la seed pour la reproductibilité
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def download_data(self):
        """
        Télécharge les données de RefinedWeb pour l'entraînement.
        """
        logger.info("Téléchargement des données...")

        # Utiliser RefinedWeb comme source principale
        from datasets import load_dataset
        rw = load_dataset("tiiuae/falcon-refinedweb")
        
        # Stocker dans notre dictionnaire de datasets
        datasets = {}
        datasets["main"] = rw
        
        # Pour les contextes plus longs, nous pouvons utiliser PG19
        pg19 = load_dataset("pg19", split="train")

        # Sauvegarder les données
        os.makedirs(os.path.join(self.base_dir, "raw"), exist_ok=True)

        logger.info(f"Sauvegarde du corpus RefinedWeb... ({len(datasets['main'])} exemples)")
        datasets["main"].save_to_disk(os.path.join(self.base_dir, "raw", "main"))

        logger.info(f"Sauvegarde du corpus PG19... ({len(pg19)} exemples)")
        pg19.save_to_disk(os.path.join(self.base_dir, "raw", "pg19"))

        logger.info("Téléchargement des données terminé!")

        return datasets

    def train_tokenizer(self, dataset, vocab_size: int = None):
        """
        Entraîne un tokenizer WordPiece sur le dataset fourni
        """
        vocab_size = vocab_size or self.vocab_size
        logger.info(f"Entraînement du tokenizer avec une taille de vocabulaire de {vocab_size}...")

        # Créer le dossier pour le tokenizer
        tokenizer_dir = os.path.join(self.base_dir, "tokenizer")
        os.makedirs(tokenizer_dir, exist_ok=True)

        # Initialiser le tokenizer (WordPiece comme BERT original)
        tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

        # Normaliser le texte
        tokenizer.normalizer = normalizers.Sequence([
            normalizers.NFD(),
            normalizers.Lowercase(),
            normalizers.StripAccents(),
        ])

        # Pre-tokenisation
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        # Configurer l'entraîneur
        trainer = trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        )

        # Fonction pour générer les textes pour l'entraînement du tokenizer
        def batch_iterator(batch_size=1000):
            for i in range(0, len(dataset), batch_size):
                yield dataset[i:i+batch_size]["text"]

        # Entraîner le tokenizer
        tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=len(dataset))

        # Ajouter le post-processing pour gérer les paires de séquences (comme BERT)
        tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B [SEP]",
            special_tokens=[
                ("[CLS]", tokenizer.token_to_id("[CLS]")),
                ("[SEP]", tokenizer.token_to_id("[SEP]")),
            ],
        )

        # Ajouter le décodeur
        tokenizer.decoder = decoders.WordPiece()

        # Sauvegarder le tokenizer
        tokenizer_path = os.path.join(tokenizer_dir, "wordpiece_tokenizer.json")
        tokenizer.save(tokenizer_path)
        logger.info(f"Tokenizer sauvegardé: {tokenizer_path}")

        # Créer un tokenizer compatible avec transformers
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="[UNK]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            mask_token="[MASK]",
        )

        # Sauvegarder le tokenizer sous format transformers
        self.tokenizer.save_pretrained(tokenizer_dir)
        logger.info(f"Tokenizer HF sauvegardé: {tokenizer_dir}")

        return self.tokenizer

    def tokenize_dataset(self, dataset, max_length=None, tokenizer=None):
        """
        Tokenise le dataset pour l'entraînement
        """
        max_length = max_length or self.max_length
        tokenizer = tokenizer or self.tokenizer

        if tokenizer is None:
            raise ValueError("Le tokenizer n'a pas été initialisé")

        logger.info(f"Tokenisation du dataset ({len(dataset)} exemples)...")

        # Fonction de tokenisation pour map
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                return_special_tokens_mask=True,
            )

        # Tokeniser le dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=multiprocessing.cpu_count(),
            remove_columns=["text"],
            desc="Tokenisation du dataset",
        )

        return tokenized_dataset

    def create_sequence_datasets(self, tokenized_dataset, lengths=[1024, 2048, 4096, 8192]):
        """
        Crée des datasets filtrés par longueur de séquence
        """
        logger.info("Création des datasets filtrés par longueur...")

        sequence_datasets = {}

        for min_length in lengths:
            # Filtrer par longueur
            filtered = tokenized_dataset.filter(
                lambda x: len(x["input_ids"]) >= min_length,
                num_proc=multiprocessing.cpu_count(),
                desc=f"Filtrage {min_length}+"
            )

            # Sauvegarder le dataset filtré
            output_dir = os.path.join(self.base_dir, "processed", f"seq_{min_length}")
            os.makedirs(output_dir, exist_ok=True)
            filtered.save_to_disk(output_dir)

            sequence_datasets[f"seq_{min_length}"] = filtered

            logger.info(f"Dataset seq_{min_length}: {len(filtered)} exemples")

        return sequence_datasets

    def create_mlm_dataset(self, tokenized_dataset):
        """
        Préparation finale des données pour l'entraînement MLM
        """
        class MLMDataset(Dataset):
            def __init__(self, tokenized_dataset, tokenizer, mlm_probability=0.15, max_length=512):
                self.tokenized_dataset = tokenized_dataset
                self.tokenizer = tokenizer
                self.mlm_probability = mlm_probability
                self.max_length = max_length

            def __len__(self):
                return len(self.tokenized_dataset)

            def __getitem__(self, idx):
                # Récupérer un exemple
                item = self.tokenized_dataset[idx]

                # Tronquer si nécessaire
                input_ids = item["input_ids"]
                if len(input_ids) > self.max_length:
                    start = random.randint(0, len(input_ids) - self.max_length)
                    input_ids = input_ids[start:start + self.max_length]

                # Créer le tenseur d'attention
                attention_mask = [1] * len(input_ids)

                # Gérer le padding
                padding_length = self.max_length - len(input_ids)
                if padding_length > 0:
                    input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
                    attention_mask = attention_mask + [0] * padding_length

                # Créer des labels pour MLM
                labels = input_ids.copy()

                # Appliquer le masquage MLM
                # On ne masque pas [CLS], [SEP], [PAD]
                special_tokens_mask = item.get("special_tokens_mask", [0] * len(input_ids))
                probability_matrix = torch.full((len(input_ids),), self.mlm_probability)
                probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

                masked_indices = torch.bernoulli(probability_matrix).bool()
                labels = torch.tensor(labels)
                labels[~masked_indices] = -100  # We only compute loss on masked tokens

                # 80% des tokens masqués sont remplacés par [MASK]
                indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
                input_ids = torch.tensor(input_ids)
                input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

                # 10% des tokens masqués sont remplacés par un token aléatoire
                indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
                random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
                input_ids[indices_random] = random_words[indices_random]

                return {
                    "input_ids": input_ids.long(),
                    "attention_mask": torch.tensor(attention_mask).long(),
                    "labels": labels.long(),
                }

        return MLMDataset(tokenized_dataset, self.tokenizer, self.mlm_probability, self.max_length)

    def create_dataloaders(self, datasets, batch_sizes, max_lengths, ddp=False):
        """
        Crée les dataloaders pour l'entraînement avec différentes configurations
        """
        dataloaders = {}

        for name, dataset in datasets.items():
            for max_length in max_lengths:
                for batch_size in batch_sizes:
                    # Créer le dataset MLM
                    self.max_length = max_length
                    mlm_dataset = self.create_mlm_dataset(dataset)

                    # Créer le sampler pour DDP si nécessaire
                    sampler = None
                    if ddp:
                        sampler = DistributedSampler(mlm_dataset)

                    # Créer le dataloader
                    dataloader = DataLoader(
                        mlm_dataset,
                        batch_size=batch_size,
                        shuffle=(sampler is None),
                        sampler=sampler,
                        num_workers=4,
                        pin_memory=True,
                    )

                    key = f"{name}_len{max_length}_bs{batch_size}"
                    dataloaders[key] = dataloader

                    logger.info(f"Dataloader créé: {key} ({len(dataloader)} batches)")

        return dataloaders

    def mixing_strategy_dataloader(self, datasets, max_length, batch_size,
                                   mixing_weights=None, ddp=False):
        """
        Crée un dataloader qui mélange plusieurs datasets selon une stratégie spécifique
        """
        if mixing_weights is None:
            mixing_weights = {name: 1.0 / len(datasets) for name in datasets.keys()}

        # Normaliser les poids
        total = sum(mixing_weights.values())
        mixing_weights = {k: v / total for k, v in mixing_weights.items()}

        # Créer les datasets MLM
        mlm_datasets = {}
        for name, dataset in datasets.items():
            self.max_length = max_length
            mlm_datasets[name] = self.create_mlm_dataset(dataset)

        # Créer un dataloader mixte
        class MixedDataLoader:
            def __init__(self, dataloaders, mixing_weights):
                self.dataloaders = dataloaders
                self.mixing_weights = mixing_weights
                self.iterators = {name: iter(dl) for name, dl in self.dataloaders.items()}
                self.lengths = {name: len(dl) for name, dl in self.dataloaders.items()}
                self.total_steps = sum(
                    int(length * weight) for name, (length, weight) in
                    zip(self.dataloaders.keys(), zip(self.lengths.values(), self.mixing_weights.values()))
                )

            def __len__(self):
                return self.total_steps

            def __iter__(self):
                self.iterators = {name: iter(dl) for name, dl in self.dataloaders.items()}
                for _ in range(self.total_steps):
                    # Choisir un dataset selon les poids
                    dataset_name = random.choices(
                        list(self.mixing_weights.keys()),
                        weights=list(self.mixing_weights.values()),
                        k=1
                    )[0]

                    # Récupérer un batch
                    try:
                        batch = next(self.iterators[dataset_name])
                    except StopIteration:
                        # Réinitialiser l'iterator si nécessaire
                        self.iterators[dataset_name] = iter(self.dataloaders[dataset_name])
                        batch = next(self.iterators[dataset_name])

                    yield batch

        # Créer les dataloaders individuels
        dataloaders = {}
        for name, mlm_dataset in mlm_datasets.items():
            # Créer le sampler pour DDP si nécessaire
            sampler = None
            if ddp:
                sampler = DistributedSampler(mlm_dataset)

            # Créer le dataloader
            dataloaders[name] = DataLoader(
                mlm_dataset,
                batch_size=batch_size,
                shuffle=(sampler is None),
                sampler=sampler,
                num_workers=4,
                pin_memory=True,
            )

        # Créer et retourner le dataloader mixte
        mixed_dataloader = MixedDataLoader(dataloaders, mixing_weights)

        return mixed_dataloader

    def get_dataloaders_for_phase(self, phase, ddp=False):
        """
        Retourne les dataloaders spécifiques pour chaque phase d'entraînement
        """
        if phase == 1:
            # Phase 1: Pré-entraînement initial (1024 tokens)
            main_dataset = load_dataset("json", data_files=os.path.join(self.base_dir, "processed", "main"))
            return self.create_dataloaders(
                {"main": main_dataset},
                batch_sizes=[32],
                max_lengths=[1024],
                ddp=ddp
            )

        elif phase == 2:
            # Phase 2: Extension du contexte (4096 tokens)
            datasets = {
                "main": load_dataset("json", data_files=os.path.join(self.base_dir, "processed", "main")),
                "seq_1024": load_dataset("json", data_files=os.path.join(self.base_dir, "processed", "seq_1024")),
                "seq_2048": load_dataset("json", data_files=os.path.join(self.base_dir, "processed", "seq_2048")),
            }

            return self.mixing_strategy_dataloader(
                datasets,
                max_length=4096,
                batch_size=4,
                mixing_weights={"main": 0.2, "seq_1024": 0.4, "seq_2048": 0.4},
                ddp=ddp
            )

        elif phase == 3:
            # Phase 3: Activation MOBA (jusqu'à 32K tokens)
            datasets = {
                "main": load_dataset("json", data_files=os.path.join(self.base_dir, "processed", "main")),
                "seq_1024": load_dataset("json", data_files=os.path.join(self.base_dir, "processed", "seq_1024")),
                "seq_2048": load_dataset("json", data_files=os.path.join(self.base_dir, "processed", "seq_2048")),
                "seq_4096": load_dataset("json", data_files=os.path.join(self.base_dir, "processed", "seq_4096")),
                "seq_8192": load_dataset("json", data_files=os.path.join(self.base_dir, "processed", "seq_8192")),
            }

            return self.mixing_strategy_dataloader(
                datasets,
                max_length=32768,
                batch_size=1,
                mixing_weights={"main": 0.1, "seq_1024": 0.2, "seq_2048": 0.2, "seq_4096": 0.25, "seq_8192": 0.25},
                ddp=ddp
            )

        else:
            raise ValueError(f"Phase d'entraînement non reconnue: {phase}")


def process_complete_pipeline():
    """
    Exécute le pipeline complet de préparation des données
    """
    # Initialisation
    data_prep = DataPreparation(
        base_dir="./data",
        vocab_size=30000,
        max_length=32768,
        mlm_probability=0.15
    )

    # Téléchargement des données
    datasets = data_prep.download_data()

    # Entraînement du tokenizer
    tokenizer = data_prep.train_tokenizer(datasets["main"])

    # Tokenisation du dataset principal
    tokenized_main = data_prep.tokenize_dataset(datasets["main"])

    # Création des datasets filtrés par longueur
    sequence_datasets = data_prep.create_sequence_datasets(
        tokenized_main,
        lengths=[1024, 2048, 4096, 8192]
    )

    # Sauvegarde
    os.makedirs(os.path.join(data_prep.base_dir, "processed"), exist_ok=True)
    tokenized_main.save_to_disk(os.path.join(data_prep.base_dir, "processed", "main"))

    # Création des dataloaders pour chaque phase
    phase1_loaders = data_prep.get_dataloaders_for_phase(1)

    logger.info("Pipeline de préparation des données terminé!")
    logger.info(f"Datasets disponibles: {list(sequence_datasets.keys())}")
    logger.info(f"Dataloaders pour la phase 1: {list(phase1_loaders.keys())}")

    return {
        "tokenizer": tokenizer,
        "datasets": sequence_datasets,
        "dataloaders": {
            "phase1": phase1_loaders,
            "phase2": data_prep.get_dataloaders_for_phase(2),
            "phase3": data_prep.get_dataloaders_for_phase(3)
        }
    }


if __name__ == "__main__":
    results = process_complete_pipeline()
    print("Préparation des données terminée!")