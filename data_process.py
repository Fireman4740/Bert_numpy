# dataprep_kaggle.py
# Installation des packages nécessaires
# !pip install -q datasets tokenizers transformers torch psutil tqdm aiohttp aiofiles

import os
import json
import random
import logging
import numpy as np
import torch
import time
import glob
import asyncio
import aiohttp
import aiofiles
import itertools
import concurrent.futures
import multiprocessing
import psutil
import warnings
from pathlib import Path
from threading import Thread, Lock
from typing import Dict, List, Optional, Tuple, Union, Set
from tqdm.auto import tqdm
from datasets import load_dataset, concatenate_datasets, config

# Correction de l'importation du DownloadManager
try:
    # Pour les versions plus récentes de datasets
    from datasets.download import DownloadManager
except ImportError:
    try:
        # Pour les versions intermédiaires
        from datasets.data_files import DownloadManager
    except ImportError:
        # Fallback: nous n'utiliserons pas le patching du DownloadManager
        DownloadManager = None
        logger = logging.getLogger(__name__)
        logger.warning("DownloadManager non trouvé, le suivi des téléchargements sera désactivé.")

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers, processors
from transformers import PreTrainedTokenizerFast
from torch.utils.data import Dataset, DataLoader, DistributedSampler

# Supprimer les avertissements pour un affichage propre
warnings.filterwarnings("ignore")

# Configuration du logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Classe pour suivre les téléchargements
class DownloadTracker:
    def __init__(self):
        self.files_to_download = set()
        self.files_downloaded = set()
        self.lock = Lock()
        self.total_size = 0
        self.downloaded_size = 0
        self.download_start_time = time.time()
        self.master_pbar = None
        self.file_pbars = {}

    def register_file(self, file_url):
        with self.lock:
            file_id = file_url.split("/")[-1]
            self.files_to_download.add(file_id)
            return file_id

    def mark_complete(self, file_id, size):
        with self.lock:
            if file_id in self.files_to_download:
                self.files_to_download.remove(file_id)
                self.files_downloaded.add(file_id)
                self.downloaded_size += size
                if file_id in self.file_pbars:
                    self.file_pbars[file_id].close()
                    del self.file_pbars[file_id]

    def get_progress(self):
        with self.lock:
            total = len(self.files_downloaded) + len(self.files_to_download)
            complete = len(self.files_downloaded)
            return complete, total, self.downloaded_size

    def create_pbar(self, file_id, total_size):
        with self.lock:
            pbar = tqdm(
                total=total_size,
                desc=f"Downloading {file_id[:10]}...",
                unit="B",
                unit_scale=True,
                position=len(self.file_pbars) + 1,
                leave=False
            )
            self.file_pbars[file_id] = pbar
            return pbar

# Créer l'instance globale pour le suivi
download_tracker = DownloadTracker()

# Patch de la méthode de téléchargement de Hugging Face seulement si DownloadManager est disponible
if DownloadManager is not None:
    try:
        original_download_method = DownloadManager._download

        def patched_download(self, url_or_urls):
            """Version patchée de la méthode _download pour traquer les téléchargements"""
            if isinstance(url_or_urls, str):
                file_id = download_tracker.register_file(url_or_urls)
            else:
                for url in url_or_urls:
                    if isinstance(url, str):
                        download_tracker.register_file(url)
            return original_download_method(self, url_or_urls)

        # Appliquer le patch
        DownloadManager._download = patched_download
    except (AttributeError, TypeError) as e:
        logger.warning(f"Impossible de patcher DownloadManager: {e}")

# Moniteur de téléchargements
class DownloadMonitor(Thread):
    def __init__(self, tracker):
        super().__init__(daemon=True)
        self.tracker = tracker
        self.running = True

    def run(self):
        self.tracker.master_pbar = tqdm(
            desc="Téléchargements globaux",
            unit="fichiers",
            position=0,
            leave=True
        )

        # Ajouter une barre pour la surveillance de la vitesse réseau
        network_pbar = tqdm(
            desc="Vitesse du réseau",
            bar_format="{desc}: {postfix}",
            position=len(self.tracker.file_pbars) + 2,
            leave=True
        )

        # Initialiser les compteurs réseau
        last_bytes_recv = psutil.net_io_counters().bytes_recv
        last_time = time.time()

        while self.running:
            try:
                # Mise à jour du suivi réseau
                current_bytes_recv = psutil.net_io_counters().bytes_recv
                current_time = time.time()

                # Calculer la vitesse
                bytes_per_sec = (current_bytes_recv - last_bytes_recv) / (current_time - last_time)
                mbps = bytes_per_sec / (1024 * 1024) * 8  # Convertir en Mbps

                # Mise à jour des barres de progression
                complete, total, downloaded_size = self.tracker.get_progress()

                # Mise à jour de la barre principale
                self.tracker.master_pbar.total = total
                self.tracker.master_pbar.n = complete

                # Calculer ETA
                elapsed = current_time - self.tracker.download_start_time
                if complete > 0 and total > complete:
                    eta = elapsed * (total - complete) / complete
                    eta_str = f"{eta/60:.1f} min"
                else:
                    eta_str = "?"

                # Mise à jour des informations affichées
                self.tracker.master_pbar.set_postfix({
                    "ETA": eta_str,
                    "Size": f"{downloaded_size / (1024*1024):.1f} MB"
                })

                # Mise à jour de la barre réseau
                network_pbar.set_postfix({
                    "Vitesse": f"{mbps:.2f} Mbps",
                    "Reçu": f"{current_bytes_recv / (1024*1024*1024):.2f} GB"
                })

                # Mettre à jour pour la prochaine itération
                last_bytes_recv = current_bytes_recv
                last_time = current_time

                # Rafraîchir les barres
                self.tracker.master_pbar.refresh()
                network_pbar.refresh()

                time.sleep(0.5)
            except Exception as e:
                logger.warning(f"Erreur dans le moniteur: {e}")
                time.sleep(1)

        # Fermer les barres à la fin
        if self.tracker.master_pbar:
            self.tracker.master_pbar.close()
        network_pbar.close()

    def stop(self):
        self.running = False

# Classe principale de préparation des données
class DataPreparation:
    def __init__(
        self,
        base_dir: str = "./data",
        vocab_size: int = 30000,
        max_length: int = 16384,  # Réduit pour T4
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

        # Configuration pour Kaggle
        self.configure_for_kaggle()

    def configure_for_kaggle(self):
        """Configure l'environnement pour Kaggle"""
        # Détecter Kaggle
        is_kaggle = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

        if is_kaggle:
            logger.info("Environnement Kaggle détecté, optimisation des paramètres...")
            # Configuration du cache HF dans un dossier persistant sur Kaggle
            cache_dir = "/kaggle/working/.cache/huggingface"
            os.makedirs(cache_dir, exist_ok=True)
            os.environ["HF_HOME"] = cache_dir

            # Activer le protocole de transfert HF pour téléchargements plus rapides
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

            # Automatically accept T&C for datasets
            os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"

            # Configuration pour limiter l'utilisation de RAM
            config.IN_MEMORY_MAX_SIZE = 1 * 1024 * 1024 * 1024  # 1GB max in memory

            # Détecter les GPUs disponibles
            if torch.cuda.is_available():
                logger.info(f"GPUs disponibles: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    logger.info(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
                    mem_info = torch.cuda.get_device_properties(i).total_memory
                    logger.info(f"    Mémoire totale: {mem_info / (1024**3):.2f} GB")
            else:
                logger.warning("Aucun GPU détecté!")

    def download_data_parallel(self, use_sample=True, sample_size="10BT", max_examples=5000):
        """
        Télécharge les données avec une approche parallèle optimisée pour Kaggle
        """
        # Si le patching du download manager n'a pas fonctionné, désactiver le moniteur
        use_monitor = DownloadManager is not None

        # Préparation du moniteur de téléchargement
        monitor = None
        if use_monitor:
            monitor = DownloadMonitor(download_tracker)
            monitor.start()

        try:
            # Téléchargement en parallèle
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                # Téléchargement du dataset principal
                main_future = executor.submit(
                    self._download_fineweb_edu,
                    use_sample=use_sample,
                    sample_size=sample_size,
                    max_examples=max_examples
                )

                # Téléchargement de PG19 en parallèle
                pg19_future = executor.submit(
                    self._download_pg19,
                    max_examples=500
                )

                # Attendre et récupérer les résultats
                main_dataset = main_future.result()
                pg19_dataset = pg19_future.result()

            # Création du dictionnaire de datasets
            datasets = {
                "main": main_dataset,
                "pg19": pg19_dataset
            }

            return datasets

        finally:
            # Arrêter le moniteur s'il existe
            if monitor:
                monitor.stop()
                monitor.join(timeout=2)

    def _download_fineweb_edu(self, use_sample=True, sample_size="10BT", max_examples=5000):
        """Télécharge FineWeb-Edu de manière optimisée"""
        logger.info(f"Téléchargement de FineWeb-Edu ({sample_size if use_sample else 'complet'})...")

        try:
            # Configuration
            load_kwargs = {
                "split": "train",
                "trust_remote_code": True
            }

            if use_sample:
                load_kwargs["name"] = f"sample-{sample_size}"
            else:
                load_kwargs["name"] = "CC-MAIN-2024-10"  # Un dump spécifique plus léger

            # Télécharger en streaming pour économiser de la mémoire
            fw_edu_stream = load_dataset("HuggingFaceFW/fineweb-edu", streaming=True, **load_kwargs)

            # Filtrer et collecter les exemples avec une barre de progression
            examples = []
            with tqdm(desc="Collecte FineWeb-Edu", total=max_examples, position=2) as pbar:
                for i, example in enumerate(fw_edu_stream):
                    # Filtrer les documents trop courts ou trop longs
                    doc_len = len(example["text"])
                    if 200 <= doc_len <= 100000:
                        examples.append(example)
                        pbar.update(1)

                        # Mettre à jour les statistiques
                        pbar.set_postfix({"len": doc_len, "count": len(examples)})

                        if len(examples) >= max_examples:
                            break

            # Convertir en Dataset
            from datasets import Dataset
            dataset = Dataset.from_list(examples)

            # Sauvegarder dans un fichier JSONL
            output_path = os.path.join(self.base_dir, "raw")
            os.makedirs(output_path, exist_ok=True)

            with tqdm(desc="Sauvegarde FineWeb-Edu", total=len(dataset), position=3) as save_pbar:
                self._save_dataset_in_batches(
                    dataset,
                    os.path.join(output_path, "main"),
                    batch_size=500,
                    progress_bar=save_pbar
                )

            logger.info(f"FineWeb-Edu: {len(dataset)} exemples téléchargés et sauvegardés")
            return dataset

        except Exception as e:
            logger.error(f"Erreur lors du téléchargement de FineWeb-Edu: {e}")
            # Solution de repli: créer un petit dataset synthétique pour les tests
            logger.info("Création d'un dataset de test synthétique...")

            from datasets import Dataset
            synthetic_examples = []

            for i in range(max_examples // 10):  # Réduire encore plus la taille
                # Générer des textes de différentes longueurs
                length = random.choice([300, 1000, 3000])
                text = ' '.join([f'word{random.randint(0, 1000)}' for _ in range(length)])
                synthetic_examples.append({"text": text})

            return Dataset.from_list(synthetic_examples)

    def _download_pg19(self, max_examples=500):
        """Télécharge PG19 de manière optimisée"""
        logger.info(f"Téléchargement d'un échantillon de PG19 ({max_examples} exemples)...")

        try:
            # Télécharger un échantillon limité
            pg19 = load_dataset("pg19", split=f"train[:{max_examples}]", trust_remote_code=True)

            # Sauvegarder le dataset
            output_path = os.path.join(self.base_dir, "raw", "pg19")
            os.makedirs(output_path, exist_ok=True)
            pg19.save_to_disk(output_path)

            logger.info(f"PG19: {len(pg19)} exemples téléchargés et sauvegardés")
            return pg19

        except Exception as e:
            logger.error(f"Erreur lors du téléchargement de PG19: {e}")
            # Créer un dataset synthétique comme solution de repli
            from datasets import Dataset
            synthetic_examples = []

            for i in range(100):  # Très petit dataset
                length = random.randint(5000, 10000)
                text = ' '.join([f'word{random.randint(0, 1000)}' for _ in range(length)])
                synthetic_examples.append({"text": text})

            return Dataset.from_list(synthetic_examples)

    def _save_dataset_in_batches(self, dataset, output_path, batch_size=500, progress_bar=None):
        """Sauvegarde un dataset en lots pour optimiser la mémoire avec barre de progression"""
        import json
        import os

        # Créer le répertoire de sortie
        os.makedirs(output_path, exist_ok=True)
        jsonl_path = f"{output_path}.jsonl"

        # Sauvegarder en lots
        with open(jsonl_path, 'w') as f:
            batch = []
            for i, item in enumerate(dataset):
                batch.append(item)

                if len(batch) >= batch_size or i == len(dataset) - 1:
                    # Écrire le lot actuel
                    for example in batch:
                        f.write(json.dumps(example) + '\n')

                    # Mettre à jour la barre de progression
                    if progress_bar is not None:
                        progress_bar.update(len(batch))

                    # Réinitialiser le lot
                    batch = []

                    # Synchronisation sur disque périodique
                    f.flush()
                    os.fsync(f.fileno())

        logger.info(f"Dataset sauvegardé par lots dans {jsonl_path}")
        return jsonl_path

    def train_tokenizer(self, dataset, vocab_size: int = None, max_examples=10000):
        """
        Entraîne un tokenizer WordPiece sur le dataset fourni
        """
        vocab_size = vocab_size or self.vocab_size
        logger.info(f"Entraînement du tokenizer avec une taille de vocabulaire de {vocab_size}...")

        # Créer le dossier pour le tokenizer
        tokenizer_dir = os.path.join(self.base_dir, "tokenizer")
        os.makedirs(tokenizer_dir, exist_ok=True)

        # Limiter les exemples pour l'entraînement du tokenizer
        if len(dataset) > max_examples:
            logger.info(f"Utilisation d'un sous-ensemble de {max_examples} exemples pour entraîner le tokenizer")
            dataset = dataset.select(range(min(max_examples, len(dataset))))

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

        # Entraîner le tokenizer avec barre de progression
        with tqdm(desc="Entraînement tokenizer", total=len(dataset), position=1) as pbar:
            # Fonction pour générer les textes avec mise à jour de la barre
            def batch_iterator(batch_size=1000):
                for i in range(0, len(dataset), batch_size):
                    batch = dataset[i:min(i+batch_size, len(dataset))]
                    pbar.update(len(batch))
                    yield batch["text"]

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

    def tokenize_dataset(self, dataset, max_length=None, tokenizer=None, num_proc=None):
        """
        Tokenise le dataset pour l'entraînement
        """
        max_length = max_length or self.max_length
        tokenizer = tokenizer or self.tokenizer

        if tokenizer is None:
            raise ValueError("Le tokenizer n'a pas été initialisé")

        logger.info(f"Tokenisation du dataset ({len(dataset)} exemples)...")

        # Déterminer le nombre de processus
        if num_proc is None:
            # Pour Kaggle, limiter pour économiser la RAM
            num_proc = min(4, multiprocessing.cpu_count())

        # Fonction de tokenisation pour map
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                return_special_tokens_mask=True,
            )

        # Tokeniser le dataset avec barre de progression
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=num_proc,
            remove_columns=["text"],
            desc="Tokenisation du dataset",
        )

        return tokenized_dataset

    def create_sequence_datasets(self, tokenized_dataset, lengths=[1024, 2048, 4096]):
        """
        Crée des datasets filtrés par longueur de séquence
        """
        logger.info("Création des datasets filtrés par longueur...")
        sequence_datasets = {}

        # Pour Kaggle T4, limiter les processus
        num_proc = min(4, multiprocessing.cpu_count())

        # Filtrer en parallèle avec des barres de progression
        for min_length in lengths:
            logger.info(f"Filtrage des séquences >= {min_length} tokens...")

            # Filtrer par longueur
            with tqdm(desc=f"Filtrage {min_length}+", position=1) as filter_pbar:
                # Fonction pour mettre à jour la barre
                def filter_with_progress(example, idx):
                    is_valid = len(example["input_ids"]) >= min_length
                    if idx % 100 == 0:
                        filter_pbar.update(100)
                    return is_valid

                filtered = tokenized_dataset.filter(
                    filter_with_progress,
                    with_indices=True,
                    num_proc=num_proc
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
                        num_workers=2,  # Réduit pour Kaggle
                        pin_memory=True,
                    )

                    key = f"{name}_len{max_length}_bs{batch_size}"
                    dataloaders[key] = dataloader

                    logger.info(f"Dataloader créé: {key} ({len(dataloader)} batches)")

        return dataloaders

    def get_dataloaders_for_phase(self, phase, ddp=False):
        """
        Retourne les dataloaders spécifiques pour chaque phase d'entraînement
        Adapté pour T4 GPUs
        """
        if phase == 1:
            # Phase 1: Pré-entraînement initial (1024 tokens)
            main_dataset = load_dataset("json", data_files=os.path.join(self.base_dir, "processed", "main"))
            return self.create_dataloaders(
                {"main": main_dataset},
                batch_sizes=[8],  # Réduit pour T4
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
                batch_size=2,  # Réduit pour T4
                mixing_weights={"main": 0.2, "seq_1024": 0.4, "seq_2048": 0.4},
                ddp=ddp
            )

        elif phase == 3:
            # Phase 3: Activation MOBA (jusqu'à 16K tokens - réduit pour T4)
            datasets = {
                "main": load_dataset("json", data_files=os.path.join(self.base_dir, "processed", "main")),
                "seq_1024": load_dataset("json", data_files=os.path.join(self.base_dir, "processed", "seq_1024")),
                "seq_2048": load_dataset("json", data_files=os.path.join(self.base_dir, "processed", "seq_2048")),
                "seq_4096": load_dataset("json", data_files=os.path.join(self.base_dir, "processed", "seq_4096")),
            }

            return self.mixing_strategy_dataloader(
                datasets,
                max_length=16384,  # Réduit pour T4
                batch_size=1,
                mixing_weights={"main": 0.1, "seq_1024": 0.3, "seq_2048": 0.3, "seq_4096": 0.3},
                ddp=ddp
            )

        else:
            raise ValueError(f"Phase d'entraînement non reconnue: {phase}")

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
                num_workers=2,  # Réduit pour T4
                pin_memory=True,
            )

        # Créer et retourner le dataloader mixte
        mixed_dataloader = MixedDataLoader(dataloaders, mixing_weights)

        return mixed_dataloader


def process_complete_pipeline():
    """
    Exécute le pipeline complet de préparation des données
    """
    # Initialisation
    data_prep = DataPreparation(
        base_dir="./data",
        vocab_size=30000,
        max_length=16384,  # Réduit pour T4
        mlm_probability=0.15
    )

    # Téléchargement des données de manière parallèle
    start_time = time.time()
    datasets = data_prep.download_data_parallel(use_sample=True, sample_size="10BT", max_examples=2000)
    download_time = time.time() - start_time
    logger.info(f"Téléchargement terminé en {download_time:.2f} secondes")

    # Entraînement du tokenizer
    tokenizer = data_prep.train_tokenizer(datasets["main"], max_examples=2000)

    # Tokenisation du dataset principal
    tokenized_main = data_prep.tokenize_dataset(datasets["main"], num_proc=2)

    # Création des datasets filtrés par longueur
    sequence_datasets = data_prep.create_sequence_datasets(
        tokenized_main,
        lengths=[1024, 2048, 4096]  # Pas de 8192 pour les T4
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


def display_download_statistics(dataset, download_time):
    """Affiche des statistiques détaillées sur le téléchargement réalisé"""
    from prettytable import PrettyTable
    import numpy as np

    # Calculer les statistiques
    doc_lengths = [len(example["text"]) for example in dataset]
    total_chars = sum(doc_lengths)
    mean_length = np.mean(doc_lengths)
    median_length = np.median(doc_lengths)
    max_length = max(doc_lengths)
    min_length = min(doc_lengths)

    # Créer un tableau pour les statistiques
    table = PrettyTable()
    table.field_names = ["Métrique", "Valeur"]
    table.add_row(["Nombre d'exemples", len(dataset)])
    table.add_row(["Taille totale (caractères)", f"{total_chars:,}"])
    table.add_row(["Taille moyenne (caractères)", f"{mean_length:.2f}"])
    table.add_row(["Taille médiane (caractères)", f"{median_length:.2f}"])
    table.add_row(["Taille maximale (caractères)", f"{max_length:,}"])
    table.add_row(["Taille minimale (caractères)", f"{min_length:,}"])
    table.add_row(["Temps de téléchargement", f"{download_time:.2f} secondes"])
    table.add_row(["Vitesse moyenne", f"{total_chars/download_time:.2f} caractères/seconde"])

    print("\n=== Statistiques de téléchargement ===")
    print(table)
    print("=====================================\n")


if __name__ == "__main__":
    # Installation des dépendances si nécessaires
    try:
        import psutil
    except ImportError:
        import subprocess
        print("Installation des dépendances requises...")
        subprocess.check_call(["pip", "install", "-q", "psutil", "aiohttp", "aiofiles", "prettytable"])
        print("Dépendances installées avec succès!")
        import psutil

    # Exécuter le pipeline complet
    try:
        start_time = time.time()
        results = process_complete_pipeline()
        execution_time = time.time() - start_time

        # Afficher les statistiques finales
        print(f"\nPréparation des données terminée en {execution_time/60:.2f} minutes!")
        print(f"Mémoire utilisée: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")

        # Afficher les statistiques sur les données
        if "main" in results["datasets"]:
            display_download_statistics(results["datasets"]["main"], execution_time)
    except Exception as e:
        import traceback
        print(f"Erreur lors de l'exécution: {e}")
        traceback.print_exc()