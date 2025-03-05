# dataprep_french.py
# Un script simplifié pour préparer des données françaises pour l'entraînement d'un modèle BERT
import os
import json
import random
import logging
import numpy np
import torch
import time
from pathlib import Path
from typing import Dict, List, Optional
import tqdm.auto
from datasets import load_dataset, Dataset, concatenate_datasets

# Pour le traitement des tokens
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers, processors
from transformers import PreTrainedTokenizerFast
from torch.utils.data import Dataset as TorchDataset, DataLoader

# Configuration du logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class FrenchDataPreparation:
    def __init__(
        self,
        base_dir: str = "./data_french",
        vocab_size: int = 32000,
        max_length: int = 512,
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
        os.makedirs(os.path.join(base_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "processed"), exist_ok=True)

        # Fixer la seed pour la reproductibilité
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Configuration du cache HF
        cache_dir = os.path.join(base_dir, ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["HF_HOME"] = cache_dir
        os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"

    def download_french_datasets(self, max_examples: int = 5000):
        """
        Télécharge plusieurs datasets français et les combine
        """
        logger.info(f"Téléchargement des datasets français (max {max_examples} exemples)...")

        datasets_info = {
            "wikipedia": {"total": max_examples // 3, "downloaded": 0},
            "oscar": {"total": max_examples // 3, "downloaded": 0},
            "books": {"total": max_examples // 3, "downloaded": 0}
        }

        combined_examples = []
        errors = []

        # 1. Télécharger Wikipedia français
        try:
            logger.info("Téléchargement de Wikipedia français...")
            wiki_dataset = load_dataset(
                "wikipedia",
                "20220301.fr",
                split="train",
                streaming=True,
                trust_remote_code=True
            )

            # Limiter le nombre d'exemples et traiter le texte
            count = 0
            with tqdm.auto.tqdm(total=datasets_info["wikipedia"]["total"], desc="Wikipedia FR") as pbar:
                for example in wiki_dataset:
                    # Extraire le texte et nettoyer
                    if len(example["text"]) > 200:  # Ignorer les articles trop courts
                        combined_examples.append({"text": example["text"], "source": "wikipedia"})
                        count += 1
                        pbar.update(1)

                        if count >= datasets_info["wikipedia"]["total"]:
                            break

            datasets_info["wikipedia"]["downloaded"] = count
            logger.info(f"Wikipedia FR: {count} exemples téléchargés")

        except Exception as e:
            logger.error(f"Erreur lors du téléchargement de Wikipedia: {e}")
            errors.append(f"Wikipedia: {str(e)}")

        # 2. Télécharger OSCAR (textes web en français)
        try:
            logger.info("Téléchargement d'OSCAR (textes web français)...")
            oscar_dataset = load_dataset(
                "oscar-corpus/OSCAR-2301",
                "fr",
                split="train",
                streaming=True,
                trust_remote_code=True
            )

            # Limiter le nombre d'exemples
            count = 0
            with tqdm.auto.tqdm(total=datasets_info["oscar"]["total"], desc="OSCAR FR") as pbar:
                for example in oscar_dataset:
                    # Vérifier la qualité du texte (éviter les textes trop courts ou trop longs)
                    text_length = len(example["text"])
                    if 200 <= text_length <= 100000:
                        combined_examples.append({"text": example["text"], "source": "oscar"})
                        count += 1
                        pbar.update(1)

                        if count >= datasets_info["oscar"]["total"]:
                            break

            datasets_info["oscar"]["downloaded"] = count
            logger.info(f"OSCAR FR: {count} exemples téléchargés")

        except Exception as e:
            logger.error(f"Erreur lors du téléchargement d'OSCAR: {e}")
            errors.append(f"OSCAR: {str(e)}")

        # 3. Télécharger des livres en français (Gutenberg ou autre source)
        try:
            logger.info("Téléchargement de livres français...")
            books_dataset = load_dataset(
                "gutenberg_multilingual",
                "fr",
                split="train",
                trust_remote_code=True
            )

            # Limiter et traiter les exemples
            shuffled_indices = list(range(len(books_dataset)))
            random.shuffle(shuffled_indices)

            count = 0
            target = min(datasets_info["books"]["total"], len(books_dataset))

            with tqdm.auto.tqdm(total=target, desc="Livres FR") as pbar:
                for idx in shuffled_indices[:target]:
                    example = books_dataset[idx]
                    # Vérifier que le texte est suffisamment long
                    if len(example["text"]) > 500:
                        combined_examples.append({"text": example["text"], "source": "books"})
                        count += 1
                        pbar.update(1)

                        if count >= datasets_info["books"]["total"]:
                            break

            datasets_info["books"]["downloaded"] = count
            logger.info(f"Livres FR: {count} exemples téléchargés")

        except Exception as e:
            logger.error(f"Erreur lors du téléchargement des livres: {e}")
            errors.append(f"Livres: {str(e)}")

        # Si tous les téléchargements ont échoué, générer un dataset synthétique de secours
        if len(combined_examples) == 0:
            logger.warning("Tous les téléchargements ont échoué! Génération d'un dataset synthétique...")
            # Générer des exemples synthétiques en français
            for i in range(min(1000, max_examples)):
                length = random.randint(500, 3000)
                # Générer un texte aléatoire avec des mots français communs
                words = ["le", "la", "un", "une", "des", "et", "ou", "mais", "donc",
                         "car", "je", "tu", "il", "elle", "nous", "vous", "ils", "elles",
                         "bonjour", "maison", "chat", "chien", "arbre", "soleil", "lune",
                         "France", "Paris", "livre", "école", "travail", "amour", "famille"]
                text = " ".join(random.choices(words, k=length))
                combined_examples.append({"text": text, "source": "synthetic"})

            logger.info(f"Dataset synthétique: {len(combined_examples)} exemples générés")

        # Créer le dataset final
        final_dataset = Dataset.from_list(combined_examples)

        # Sauvegarder le dataset
        output_path = os.path.join(self.base_dir, "raw", "french_dataset.json")
        final_dataset.to_json(output_path)

        # Afficher les statistiques
        logger.info(f"Dataset français combiné: {len(final_dataset)} exemples au total")
        for source, info in datasets_info.items():
            logger.info(f"  - {source}: {info['downloaded']}/{info['total']} exemples")

        if errors:
            logger.warning("Des erreurs se sont produites pendant le téléchargement:")
            for error in errors:
                logger.warning(f"  - {error}")

        return final_dataset

    def get_dataloaders_for_phase(self, phase, ddp=False):
        """
        Retourne les dataloaders appropriés pour une phase spécifique d'entraînement.
        Compatible avec l'API du code d'entraînement NeoBERT-MOBA.

        Args:
            phase (int): La phase d'entraînement (1, 2 ou 3)
            ddp (bool): Si True, configure pour l'entraînement distribué

        Returns:
            dict: Dictionnaire de dataloaders pour la phase
        """
        # Vérifier si nous avons déjà des données préparées
        processed_dir = os.path.join(self.base_dir, "processed")
        tokenized_dir = os.path.join(processed_dir, "tokenized")

        # Ajuster les paramètres selon la phase
        if phase == 1:
            # Phase 1: Séquences courtes (1024 tokens)
            seq_len = min(1024, self.max_length)
            batch_sizes = {"main_len1024_bs32": 32, "main_len1024_bs8": 8}
        elif phase == 2:
            # Phase 2: Séquences moyennes (4096 tokens)
            seq_len = min(4096, self.max_length)
            batch_sizes = {"main": 2}
        elif phase == 3:
            # Phase 3: Séquences longues (16k+ tokens)
            seq_len = self.max_length
            batch_sizes = {"main": 1}
        else:
            raise ValueError(f"Phase inconnue: {phase}")

        # Si le dataset tokenisé existe, le charger
        if os.path.exists(tokenized_dir):
            try:
                from datasets import load_from_disk
                tokenized_dataset = load_from_disk(tokenized_dir)
                logger.info(f"Dataset tokenisé chargé: {len(tokenized_dataset)} exemples")
            except Exception as e:
                logger.error(f"Erreur lors du chargement du dataset tokenisé: {e}")
                # Charger ou créer le dataset de base
                dataset_path = os.path.join(self.base_dir, "raw", "french_dataset.json")
                if os.path.exists(dataset_path):
                    from datasets import load_dataset
                    dataset = load_dataset("json", data_files=dataset_path, split="train")
                else:
                    dataset = self.download_french_datasets()

                # Vérifier le tokenizer
                if self.tokenizer is None:
                    self.tokenizer = self.train_tokenizer(dataset)

                # Tokeniser le dataset
                tokenized_dataset = self.tokenize_dataset(dataset)
        else:
            # Créer le pipeline complet
            dataset = self.download_french_datasets()
            self.tokenizer = self.train_tokenizer(dataset)
            tokenized_dataset = self.tokenize_dataset(dataset)

        # Vérifier explicitement le tokenizer avant de créer le MLM dataset
        if self.tokenizer is None:
            # Si vous avez déjà un tokenizer sauvegardé, le charger
            tokenizer_dir = os.path.join(self.base_dir, "tokenizer")
            if os.path.exists(os.path.join(tokenizer_dir, "french_tokenizer.json")):
                from transformers import PreTrainedTokenizerFast
                self.tokenizer = PreTrainedTokenizerFast(
                    tokenizer_file=os.path.join(tokenizer_dir, "french_tokenizer.json"),
                    unk_token="[UNK]",
                    cls_token="[CLS]",
                    sep_token="[SEP]",
                    pad_token="[PAD]",
                    mask_token="[MASK]",
                    model_max_length=self.max_length,
                )
                logger.info(f"Tokenizer chargé depuis {tokenizer_dir}")
            else:
                # Si on n'a toujours pas de tokenizer, utiliser un tokenizer par défaut
                logger.warning("Impossible de charger ou créer un tokenizer! Utilisation d'un tokenizer par défaut.")
                from transformers import BertTokenizerFast
                self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")

        # Créer le dataset MLM avec le tokenizer vérifié
        mlm_dataset = self.create_mlm_dataset(tokenized_dataset)

        # Préparer les dataloaders avec les tailles de batch appropriées
        result = {}

        for name, batch_size in batch_sizes.items():
            # Ajuster le batch_size pour DDP
            effective_batch_size = batch_size
            if ddp:
                # En DDP, chaque processus reçoit une portion des données
                import torch.distributed as dist
                world_size = dist.get_world_size() if dist.is_initialized() else 1
                effective_batch_size = max(1, batch_size // world_size)

            # Créer le dataloader
            dataloader = DataLoader(
                mlm_dataset,
                batch_size=effective_batch_size,
                shuffle=True,
                num_workers=min(4, os.cpu_count() or 1),
                pin_memory=True if torch.cuda.is_available() else False,
            )

            result[name] = dataloader

        # Si c'est la phase 1, retourner un dictionnaire avec deux tailles de batch
        # Sinon, retourner directement le dataloader principal
        if phase == 1:
            return result
        else:
            return result["main"]

    def train_tokenizer(self, dataset, vocab_size: int = None, max_examples=10000):
        """
        Entraîne un tokenizer WordPiece adapté au français sur le dataset fourni
        """
        vocab_size = vocab_size or self.vocab_size
        logger.info(f"Entraînement du tokenizer français avec vocabulaire de {vocab_size} tokens...")

        # Créer le dossier pour le tokenizer
        tokenizer_dir = os.path.join(self.base_dir, "tokenizer")
        os.makedirs(tokenizer_dir, exist_ok=True)

        # Limiter les exemples pour l'entraînement du tokenizer
        if len(dataset) > max_examples:
            logger.info(f"Utilisation d'un sous-ensemble de {max_examples} exemples pour le tokenizer")
            indices = random.sample(range(len(dataset)), min(max_examples, len(dataset)))
            train_dataset = dataset.select(indices)
        else:
            train_dataset = dataset

        # Initialiser le tokenizer (WordPiece comme BERT)
        tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

        # Normaliser le texte (adapté pour le français)
        tokenizer.normalizer = normalizers.Sequence([
            normalizers.NFD(),
            normalizers.Lowercase(),
            # Ne pas supprimer les accents pour le français
        ])

        # Pre-tokenisation
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        # Configurer l'entraîneur avec tokens spéciaux
        trainer = trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
            min_frequency=2,  # Réduire le bruit pour les mots rares
        )

        # Entraîner le tokenizer avec barre de progression
        with tqdm.auto.tqdm(desc="Entraînement tokenizer", total=len(train_dataset)) as pbar:
            # Fonction pour générer les textes avec mise à jour de la barre
            def batch_iterator(batch_size=1000):
                for i in range(0, len(train_dataset), batch_size):
                    batch = train_dataset[i:min(i+batch_size, len(train_dataset))]
                    pbar.update(len(batch))
                    yield batch["text"]

            # Entraînement robuste avec gestion des erreurs
            try:
                tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=len(train_dataset))
            except Exception as e:
                logger.error(f"Erreur pendant l'entraînement du tokenizer: {e}")
                # Réduire la taille du vocabulaire et réessayer
                logger.info("Tentative avec un vocabulaire plus petit...")
                trainer = trainers.WordPieceTrainer(
                    vocab_size=min(vocab_size // 2, 10000),
                    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
                )
                pbar.reset()
                tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=len(train_dataset))

        # Ajouter le post-processing (comme BERT)
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
        tokenizer_path = os.path.join(tokenizer_dir, "french_tokenizer.json")
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
            model_max_length=self.max_length,
        )

        # Sauvegarder le tokenizer sous format transformers
        self.tokenizer.save_pretrained(tokenizer_dir)
        logger.info(f"Tokenizer HF sauvegardé: {tokenizer_dir}")

        return self.tokenizer

    def tokenize_dataset(self, dataset, num_proc=4):
        """
        Tokenise le dataset pour l'entraînement avec gestion des erreurs
        """
        if self.tokenizer is None:
            raise ValueError("Le tokenizer n'a pas été initialisé")

        logger.info(f"Tokenisation du dataset ({len(dataset)} exemples)...")

        # Fonction de tokenisation pour map avec troncation
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_special_tokens_mask=True,
            )

        # Tokeniser avec gestion d'erreurs
        try:
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                num_proc=num_proc,
                remove_columns=["text", "source"],
                desc="Tokenisation du dataset",
            )
        except Exception as e:
            logger.error(f"Erreur pendant la tokenisation: {e}")
            logger.info("Tentative avec des paramètres plus conservateurs...")

            # Réessayer avec moins de processus et sans padding
            def simple_tokenize_function(examples):
                return self.tokenizer(
                    examples["text"],
                    truncation=True,
                    max_length=self.max_length,
                    return_special_tokens_mask=True,
                )

            tokenized_dataset = dataset.map(
                simple_tokenize_function,
                batched=True,
                num_proc=1,  # Réduit à 1 processus
                remove_columns=["text", "source"],
                desc="Tokenisation simplifiée",
            )

        # Sauvegarder le dataset tokenisé
        output_dir = os.path.join(self.base_dir, "processed", "tokenized")
        os.makedirs(output_dir, exist_ok=True)
        tokenized_dataset.save_to_disk(output_dir)

        logger.info(f"Dataset tokenisé: {len(tokenized_dataset)} exemples")
        return tokenized_dataset

    def create_mlm_dataset(self, tokenized_dataset):
        """
        Crée un dataset PyTorch pour l'entraînement MLM
        """
        # Vérifier si le tokenizer est initialisé
        if self.tokenizer is None:
            logger.warning("Le tokenizer n'est pas initialisé. Tentative de chargement...")
            
            # Tenter de charger un tokenizer sauvegardé
            tokenizer_dir = os.path.join(self.base_dir, "tokenizer")
            if os.path.exists(os.path.join(tokenizer_dir, "french_tokenizer.json")):
                from transformers import PreTrainedTokenizerFast
                self.tokenizer = PreTrainedTokenizerFast(
                    tokenizer_file=os.path.join(tokenizer_dir, "french_tokenizer.json"),
                    unk_token="[UNK]",
                    cls_token="[CLS]",
                    sep_token="[SEP]",
                    pad_token="[PAD]",
                    mask_token="[MASK]",
                    model_max_length=self.max_length,
                )
                logger.info(f"Tokenizer chargé depuis {tokenizer_dir}")
            else:
                # Utiliser un tokenizer par défaut en dernier recours
                logger.warning("Aucun tokenizer trouvé. Utilisation d'un tokenizer par défaut.")
                from transformers import BertTokenizerFast
                self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")

        class MLMDataset(TorchDataset):
            def __init__(self, tokenized_dataset, tokenizer, mlm_probability=0.15):
                self.tokenized_dataset = tokenized_dataset
                self.tokenizer = tokenizer
                self.mlm_probability = mlm_probability
                
                # Définir des valeurs par défaut si le tokenizer est None
                if self.tokenizer is None:
                    logger.warning("Attention: Tokenizer non initialisé! Utilisation de valeurs par défaut.")
                    self.pad_token_id = 0
                    self.mask_token_id = 1
                    self.vocab_size = 30000
                else:
                    self.pad_token_id = self.tokenizer.pad_token_id
                    self.mask_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
                    self.vocab_size = len(self.tokenizer)

            def __len__(self):
                return len(self.tokenized_dataset)

            def __getitem__(self, idx):
                # Récupérer un exemple
                item = self.tokenized_dataset[idx]

                # Créer les tenseurs d'entrée
                input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
                attention_mask = torch.tensor(item["attention_mask"], dtype=torch.long)

                # Créer les labels pour MLM (copie des input_ids)
                labels = input_ids.clone()

                # Créer un masque pour le MLM
                probability_matrix = torch.full(labels.shape, self.mlm_probability)

                # Ne pas masquer les tokens spéciaux
                special_tokens_mask = torch.tensor(item["special_tokens_mask"], dtype=torch.bool)
                probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

                # Ne pas masquer les tokens de padding
                padding_mask = input_ids.eq(self.pad_token_id)
                probability_matrix.masked_fill_(padding_mask, value=0.0)

                # Appliquer le masque aléatoirement
                masked_indices = torch.bernoulli(probability_matrix).bool()

                # Remplacer les labels des tokens non masqués par -100 (ignorés dans la loss)
                labels[~masked_indices] = -100

                # 80% des tokens masqués deviennent [MASK]
                indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
                input_ids[indices_replaced] = self.mask_token_id

                # 10% des tokens masqués sont remplacés par un token aléatoire
                indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
                random_words = torch.randint(self.vocab_size, labels.shape, dtype=torch.long)
                input_ids[indices_random] = random_words[indices_random]

                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }

        return MLMDataset(tokenized_dataset, self.tokenizer, self.mlm_probability)

    def prepare_dataloaders(self, tokenized_dataset, batch_size=32, val_split=0.1):
        """
        Prépare les dataloaders pour l'entraînement et la validation
        """
        # Diviser en ensembles d'entraînement et de validation
        if val_split > 0:
            # Calculer les tailles des ensembles
            val_size = int(len(tokenized_dataset) * val_split)
            train_size = len(tokenized_dataset) - val_size

            # Diviser le dataset
            train_dataset, val_dataset = torch.utils.data.random_split(
                tokenized_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.seed)
            )

            logger.info(f"Dataset divisé en {train_size} exemples d'entraînement et {val_size} exemples de validation")
        else:
            train_dataset = tokenized_dataset
            val_dataset = None
            logger.info(f"Utilisation de tout le dataset ({len(tokenized_dataset)} exemples) pour l'entraînement")

        # Créer les dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=min(4, os.cpu_count() or 1),
            pin_memory=True if torch.cuda.is_available() else False,
        )

        val_dataloader = None
        if val_dataset:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=min(4, os.cpu_count() or 1),
                pin_memory=True if torch.cuda.is_available() else False,
            )

        return {"train": train_dataloader, "val": val_dataloader}
    import tqdm.auto

    print("\n============== STATISTIQUES ==============")
    print(f"Temps total d'exécution: {execution_time/60:.2f} minutes")

    # Statistiques sur le dataset
    print("\n----- Dataset -----")
    print(f"Nombre total d'exemples: {len(dataset)}")

    # Sources des données
    sources = {}
    for example in tqdm.auto.tqdm(dataset, desc="Analyse des sources"):
        source = example.get("source", "unknown")
        sources[source] = sources.get(source, 0) + 1

    for source, count in sources.items():
        print(f"  - {source}: {count} exemples ({count/len(dataset)*100:.1f}%)")

    # Statistiques sur le tokenizer
    print("\n----- Tokenizer -----")
    print(f"Taille du vocabulaire: {len(tokenizer)}")

    # Échantillon de tokens
    sample_tokens = list(tokenizer.get_vocab().keys())[:20]
    print(f"Exemples de tokens: {', '.join(sample_tokens)}")

    # Statistiques sur les dataloaders
    print("\n----- Dataloaders -----")
    train_dataloader = dataloaders.get("train")
    val_dataloader = dataloaders.get("val")

    if train_dataloader:
        print(f"Batches d'entraînement: {len(train_dataloader)}")
        print(f"Taille du batch: {train_dataloader.batch_size}")
        print(f"Exemples d'entraînement: ~{len(train_dataloader) * train_dataloader.batch_size}")

    if val_dataloader:
        print(f"Batches de validation: {len(val_dataloader)}")
        print(f"Exemples de validation: ~{len(val_dataloader) * val_dataloader.batch_size}")

    print("==========================================")

if __name__ == "__main__":
    # Installation des dépendances si nécessaires
    try:
        import tqdm
    except ImportError:
        import subprocess
        print("Installation des dépendances requises...")
        subprocess.check_call(["pip", "install", "-q", "tqdm", "datasets", "tokenizers", "transformers", "torch"])
        print("Dépendances installées avec succès!")

    # Lancer le pipeline
    print("Démarrage du pipeline de préparation des données françaises...")
    results = process_french_data_pipeline(max_examples=5000, batch_size=32)
    print("Pipeline terminé!")