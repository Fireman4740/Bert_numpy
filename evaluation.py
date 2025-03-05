import os
import json
import time
import torch
import random
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Tuple
from collections import defaultdict
import pandas as pd
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets

# Pour l'intégration aux benchmarks
from transformers import AutoTokenizer, AutoModelForMaskedLM
from evaluate import load
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Import de nos modules pour NeoBERT-MOBA
from model_architecture import NeoBERTMoBA, create_neobert_moba_model

# Configuration du logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationArguments:
    """Arguments pour l'évaluation du modèle NeoBERT-MOBA"""
    model_path: str = field(
        default=None,
        metadata={"help": "Chemin vers le checkpoint du modèle à évaluer"}
    )
    output_dir: str = field(
        default="./evaluation_results",
        metadata={"help": "Répertoire de sortie pour les résultats d'évaluation"}
    )
    config_path: str = field(
        default=None,
        metadata={"help": "Chemin vers le fichier de configuration du modèle"}
    )
    tokenizer_path: str = field(
        default=None,
        metadata={"help": "Chemin vers le tokenizer"}
    )

    # Évaluations à effectuer
    run_glue: bool = field(
        default=False,
        metadata={"help": "Exécuter l'évaluation GLUE"}
    )
    run_mteb: bool = field(
        default=False,
        metadata={"help": "Exécuter l'évaluation MTEB"}
    )
    run_long_context: bool = field(
        default=True,
        metadata={"help": "Exécuter les évaluations de contexte long"}
    )
    run_efficiency: bool = field(
        default=True,
        metadata={"help": "Exécuter les évaluations d'efficacité"}
    )
    run_ablation: bool = field(
        default=False,
        metadata={"help": "Exécuter les études d'ablation"}
    )

    # Configurations spécifiques
    context_lengths: List[int] = field(
        default_factory=lambda: [2048, 4096, 8192, 16384, 32768],
        metadata={"help": "Longueurs de contexte à évaluer"}
    )
    batch_size: int = field(
        default=1,
        metadata={"help": "Taille du batch pour l'évaluation"}
    )
    max_samples: int = field(
        default=100,
        metadata={"help": "Nombre maximum d'échantillons à évaluer"}
    )

    # Configurations MOBA pour les ablations
    moba_configs: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {"block_size": 512, "top_k": 2},
            {"block_size": 512, "top_k": 3},
            {"block_size": 512, "top_k": 4},
            {"block_size": 256, "top_k": 3},
            {"block_size": 1024, "top_k": 3}
        ],
        metadata={"help": "Configurations MOBA pour les études d'ablation"}
    )

    device: str = field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        metadata={"help": "Device à utiliser pour l'évaluation"}
    )


class LongContextDataset(Dataset):
    """Dataset pour l'évaluation des capacités de contexte long"""

    def __init__(self, tokenizer, max_length=32768, dataset_name="pg19", split="test", max_samples=100, seed=42):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_samples = max_samples

        np.random.seed(seed)
        random.seed(seed)

        logger.info(f"Chargement du dataset {dataset_name} pour l'évaluation de contexte long")
        self.dataset = load_dataset(dataset_name, split=split)

        # Limiter le nombre d'échantillons
        if max_samples and max_samples < len(self.dataset):
            indices = np.random.choice(len(self.dataset), max_samples, replace=False)
            self.dataset = self.dataset.select(indices)

        logger.info(f"Dataset chargé: {len(self.dataset)} échantillons")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]

        # Tokeniser le texte
        tokens = self.tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=self.max_length)

        return {
            "input_ids": tokens,
            "text": text
        }

    def collate_fn(self, batch):
        """Fonction de collation personnalisée pour le dataloader"""
        input_ids = [item["input_ids"] for item in batch]
        texts = [item["text"] for item in batch]

        # Padding
        max_len = max(len(ids) for ids in input_ids)
        padded_input_ids = []
        attention_masks = []

        for ids in input_ids:
            padding_length = max_len - len(ids)
            padded_ids = ids + [self.tokenizer.pad_token_id] * padding_length
            mask = [1] * len(ids) + [0] * padding_length

            padded_input_ids.append(padded_ids)
            attention_masks.append(mask)

        return {
            "input_ids": torch.tensor(padded_input_ids),
            "attention_mask": torch.tensor(attention_masks),
            "texts": texts
        }


class NeedleInHaystackDataset(Dataset):
    """Dataset pour l'évaluation 'Needle in a Haystack'"""

    def __init__(self, tokenizer, max_length=32768, num_samples=100, seed=42):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_samples = num_samples

        np.random.seed(seed)
        random.seed(seed)

        # Charger un dataset de phrases pour créer les contextes
        logger.info("Chargement des données pour Needle in a Haystack")
        self.wiki_dataset = load_dataset("wikipedia", "20220301.en", split="train")

        # Générer les exemples
        self.examples = self._generate_examples()

    def _generate_examples(self):
        examples = []

        logger.info(f"Génération de {self.num_samples} exemples Needle in a Haystack")
        for _ in tqdm(range(self.num_samples)):
            # Sélectionner une phrase "aiguille"
            needle_idx = random.randint(0, len(self.wiki_dataset) - 1)
            wiki_text = self.wiki_dataset[needle_idx]["text"]

            # Extraire une phrase de longueur moyenne
            sentences = wiki_text.split(". ")
            if len(sentences) < 2:
                continue

            needle_sentence_idx = random.randint(0, len(sentences) - 1)
            needle = sentences[needle_sentence_idx].strip() + "."

            # Vérifier que l'aiguille n'est pas trop courte ni trop longue
            if len(needle.split()) < 5 or len(needle.split()) > 25:
                continue

            # Créer un champ de foin (texte aléatoire)
            haystack_parts = []

            # Position où insérer l'aiguille (aléatoire)
            position_percentile = random.uniform(0.1, 0.9)  # Entre 10% et 90% de la longueur totale

            # Construire le champ de foin en tokenisant pour contrôler la longueur
            haystack_tokens = 0
            target_tokens = int(self.max_length * 0.95)  # Utiliser 95% de la longueur max

            # Déterminer où insérer l'aiguille en tokens
            needle_position = int(target_tokens * position_percentile)

            # Construire le début du champ de foin
            while haystack_tokens < needle_position:
                # Sélectionner un texte aléatoire différent de celui contenant l'aiguille
                filler_idx = random.randint(0, len(self.wiki_dataset) - 1)
                if filler_idx == needle_idx:
                    continue

                filler_text = self.wiki_dataset[filler_idx]["text"]

                # Prendre une portion du texte
                filler_tokens = self.tokenizer.encode(filler_text, add_special_tokens=False)
                tokens_to_take = min(needle_position - haystack_tokens, len(filler_tokens))

                if tokens_to_take <= 0:
                    continue

                text_to_add = self.tokenizer.decode(filler_tokens[:tokens_to_take])
                haystack_parts.append(text_to_add)
                haystack_tokens += tokens_to_take

            # Ajouter l'aiguille
            haystack_parts.append(needle)
            needle_tokens = self.tokenizer.encode(needle, add_special_tokens=False)
            haystack_tokens += len(needle_tokens)

            # Continuer à ajouter du texte jusqu'à atteindre la longueur cible
            while haystack_tokens < target_tokens:
                filler_idx = random.randint(0, len(self.wiki_dataset) - 1)
                if filler_idx == needle_idx:
                    continue

                filler_text = self.wiki_dataset[filler_idx]["text"]

                filler_tokens = self.tokenizer.encode(filler_text, add_special_tokens=False)
                tokens_to_take = min(target_tokens - haystack_tokens, len(filler_tokens))

                if tokens_to_take <= 0:
                    continue

                text_to_add = self.tokenizer.decode(filler_tokens[:tokens_to_take])
                haystack_parts.append(text_to_add)
                haystack_tokens += tokens_to_take

            # Construire le texte complet
            haystack = " ".join(haystack_parts)

            # Générer 5 questions sur l'aiguille
            question = f"Quelle information se trouve dans la phrase: '{needle}'?"

            examples.append({
                "haystack": haystack,
                "needle": needle,
                "question": question,
                "position_percentile": position_percentile
            })

        logger.info(f"Généré {len(examples)} exemples valides")
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Tokeniser le texte complet (haystack + question)
        full_text = f"{example['haystack']}\n\nQuestion: {example['question']}\nRéponse:"

        # Tokenisation
        tokens = self.tokenizer.encode(full_text, add_special_tokens=True, truncation=True, max_length=self.max_length)

        return {
            "input_ids": tokens,
            "needle": example["needle"],
            "position_percentile": example["position_percentile"],
            "text": full_text
        }

    def collate_fn(self, batch):
        """Fonction de collation personnalisée pour le dataloader"""
        input_ids = [item["input_ids"] for item in batch]
        needles = [item["needle"] for item in batch]
        positions = [item["position_percentile"] for item in batch]
        texts = [item["text"] for item in batch]

        # Padding
        max_len = max(len(ids) for ids in input_ids)
        padded_input_ids = []
        attention_masks = []

        for ids in input_ids:
            padding_length = max_len - len(ids)
            padded_ids = ids + [self.tokenizer.pad_token_id] * padding_length
            mask = [1] * len(ids) + [0] * padding_length

            padded_input_ids.append(padded_ids)
            attention_masks.append(mask)

        return {
            "input_ids": torch.tensor(padded_input_ids),
            "attention_mask": torch.tensor(attention_masks),
            "needles": needles,
            "positions": positions,
            "texts": texts
        }


class RULERDataset(Dataset):
    """Dataset pour l'évaluation RULER (Reasoning Understanding Long & Extended Reasoning)"""

    def __init__(self, tokenizer, max_length=32768, num_samples=50, sequence_length=128000, seed=42):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_samples = num_samples
        self.sequence_length = sequence_length

        np.random.seed(seed)
        random.seed(seed)

        logger.info("Création des exemples RULER")
        self.examples = self._generate_examples()

    def _generate_examples(self):
        """Génère des exemples de raisonnement à longue distance"""
        examples = []

        for i in range(self.num_samples):
            # Créer un problème de raisonnement à longue distance
            example = self._create_long_dependency_problem()
            examples.append(example)

        return examples

    def _create_long_dependency_problem(self):
        """Crée un problème avec des dépendances à longue distance"""
        # Générer un ensemble de variables avec des valeurs
        variables = {}
        for var in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            variables[var] = random.randint(1, 100)

        # Créer une série d'étapes de calcul
        steps = []
        dependencies = []
        final_variable = None

        # Choisir des variables à utiliser dans ce problème
        selected_vars = random.sample(list(variables.keys()), min(10, len(variables)))

        # Créer le problème avec des dépendances à longue distance
        operations = ["+", "-", "*", "//"]

        for i in range(len(selected_vars) - 1):
            # Choisir deux variables pour l'opération
            if i == 0:
                # Première étape: utiliser deux variables initiales
                var1, var2 = selected_vars[0], selected_vars[1]
                dependencies.extend([var1, var2])
            else:
                # Utiliser le résultat précédent et une nouvelle variable
                var1 = f"result_{i-1}"
                var2 = selected_vars[i+1]
                dependencies.append(var2)

            # Choisir une opération
            op = random.choice(operations)

            # Créer l'étape
            result_var = f"result_{i}"
            if op == "+":
                step = f"{result_var} = {var1} + {var2}"
            elif op == "-":
                step = f"{result_var} = {var1} - {var2}"
            elif op == "*":
                step = f"{result_var} = {var1} * {var2}"
            else:  # division entière
                # Éviter la division par zéro
                if var2 == "0" or (var2 in variables and variables[var2] == 0):
                    op = "+"
                    step = f"{result_var} = {var1} + {var2}"
                else:
                    step = f"{result_var} = {var1} // {var2}"

            steps.append(step)
            final_variable = result_var

        # Créer le texte du problème
        problem_text = "Vous devez suivre les étapes de calcul suivantes:\n\n"

        # Ajouter les définitions de variables
        problem_text += "Définitions des variables:\n"
        for var in selected_vars:
            problem_text += f"{var} = {variables[var]}\n"

        problem_text += "\nCalculs à effectuer:\n"

        # Ajouter beaucoup de texte entre les définitions et les étapes
        filler_text = "Ceci est un texte de remplissage pour créer une longue séquence. " * 5000
        problem_text += f"\n{filler_text}\n\n"

        # Ajouter les étapes
        for step in steps:
            problem_text += f"{step}\n"

        # Ajouter la question
        question = f"Quelle est la valeur finale de {final_variable}?"
        problem_text += f"\n{question}"

        # Calculer la réponse
        namespace = variables.copy()
        for step in steps:
            exec(step, namespace)

        answer = namespace[final_variable]

        return {
            "problem_text": problem_text,
            "question": question,
            "answer": answer,
            "dependencies": dependencies
        }

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Tokeniser le texte complet
        tokens = self.tokenizer.encode(example["problem_text"], add_special_tokens=True, truncation=True, max_length=self.max_length)

        return {
            "input_ids": tokens,
            "question": example["question"],
            "answer": example["answer"],
            "dependencies": example["dependencies"],
            "text": example["problem_text"]
        }

    def collate_fn(self, batch):
        """Fonction de collation personnalisée pour le dataloader"""
        input_ids = [item["input_ids"] for item in batch]
        questions = [item["question"] for item in batch]
        answers = [item["answer"] for item in batch]
        dependencies = [item["dependencies"] for item in batch]
        texts = [item["text"] for item in batch]

        # Padding
        max_len = max(len(ids) for ids in input_ids)
        padded_input_ids = []
        attention_masks = []

        for ids in input_ids:
            padding_length = max_len - len(ids)
            padded_ids = ids + [self.tokenizer.pad_token_id] * padding_length
            mask = [1] * len(ids) + [0] * padding_length

            padded_input_ids.append(padded_ids)
            attention_masks.append(mask)

        return {
            "input_ids": torch.tensor(padded_input_ids),
            "attention_mask": torch.tensor(attention_masks),
            "questions": questions,
            "answers": answers,
            "dependencies": dependencies,
            "texts": texts
        }


class NeoBERTMoBAEvaluator:
    """Classe principale pour l'évaluation de NeoBERT-MOBA"""

    def __init__(self, args):
        self.args = args

        # Créer le répertoire de sortie
        os.makedirs(args.output_dir, exist_ok=True)

        # Charger le modèle
        self.model = self._load_model()

        # Charger le tokenizer
        self.tokenizer = self._load_tokenizer()

        self.device = torch.device(args.device)
        self.model.to(self.device)

        logger.info(f"Modèle et tokenizer chargés. Dispositif: {self.device}")

    def _load_model(self):
        """Charge le modèle NeoBERT-MOBA à partir du checkpoint"""
        if self.args.model_path is None:
            raise ValueError("Le chemin du modèle doit être spécifié")

        logger.info(f"Chargement du modèle depuis {self.args.model_path}")

        # Charger la configuration
        config_path = self.args.config_path or os.path.join(self.args.model_path, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        # Créer le modèle
        model = create_neobert_moba_model(config)

        # Charger les poids
        weights_path = self.args.model_path
        if os.path.isdir(weights_path):
            weights_path = os.path.join(weights_path, "pytorch_model.bin")

        model.load_state_dict(torch.load(weights_path, map_location="cpu"))

        return model

    def _load_tokenizer(self):
        """Charge le tokenizer pour le modèle"""
        tokenizer_path = self.args.tokenizer_path or self.args.model_path

        logger.info(f"Chargement du tokenizer depuis {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # S'assurer que les tokens spéciaux sont définis
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or "[PAD]"

        return tokenizer

    def evaluate_all(self):
        """Exécute toutes les évaluations demandées"""
        results = {}

        # Évaluation GLUE
        if self.args.run_glue:
            logger.info("Démarrage de l'évaluation GLUE")
            glue_results = self.evaluate_glue()
            results["glue"] = glue_results

            # Sauvegarde des résultats intermédiaires
            with open(os.path.join(self.args.output_dir, "glue_results.json"), "w") as f:
                json.dump(glue_results, f, indent=2)

        # Évaluation MTEB
        if self.args.run_mteb:
            logger.info("Démarrage de l'évaluation MTEB")
            mteb_results = self.evaluate_mteb()
            results["mteb"] = mteb_results

            # Sauvegarde des résultats intermédiaires
            with open(os.path.join(self.args.output_dir, "mteb_results.json"), "w") as f:
                json.dump(mteb_results, f, indent=2)

        # Évaluation de contexte long
        if self.args.run_long_context:
            logger.info("Démarrage de l'évaluation de contexte long")
            long_context_results = self.evaluate_long_context()
            results["long_context"] = long_context_results

            # Sauvegarde des résultats intermédiaires
            with open(os.path.join(self.args.output_dir, "long_context_results.json"), "w") as f:
                json.dump(long_context_results, f, indent=2)

        # Évaluation d'efficacité
        if self.args.run_efficiency:
            logger.info("Démarrage de l'évaluation d'efficacité")
            efficiency_results = self.evaluate_efficiency()
            results["efficiency"] = efficiency_results

            # Sauvegarde des résultats intermédiaires
            with open(os.path.join(self.args.output_dir, "efficiency_results.json"), "w") as f:
                json.dump(efficiency_results, f, indent=2)

        # Études d'ablation
        if self.args.run_ablation:
            logger.info("Démarrage des études d'ablation")
            ablation_results = self.run_ablation_studies()
            results["ablation"] = ablation_results

            # Sauvegarde des résultats intermédiaires
            with open(os.path.join(self.args.output_dir, "ablation_results.json"), "w") as f:
                json.dump(ablation_results, f, indent=2)

        # Sauvegarde des résultats complets
        logger.info("Sauvegarde des résultats d'évaluation")
        with open(os.path.join(self.args.output_dir, "evaluation_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        # Générer des visualisations
        self.generate_visualizations(results)

        return results

    def evaluate_glue(self):
        """Évalue le modèle sur les tâches GLUE"""
        logger.info("Évaluation sur GLUE non implémentée dans cette version")
        return {"status": "not_implemented"}

    def evaluate_mteb(self):
        """Évalue le modèle sur les tâches MTEB"""
        logger.info("Évaluation sur MTEB non implémentée dans cette version")
        return {"status": "not_implemented"}

    def evaluate_long_context(self):
        """Évalue les capacités de contexte long du modèle"""
        results = {}

        # 1. Position-wise LM loss
        logger.info("Évaluation de la perte de langage par position")
        position_loss_results = self.evaluate_position_wise_loss()
        results["position_wise_loss"] = position_loss_results

        # 2. Trailing token loss
        logger.info("Évaluation de la perte sur les tokens finaux")
        trailing_loss_results = self.evaluate_trailing_token_loss()
        results["trailing_token_loss"] = trailing_loss_results

        # 3. Needle in a Haystack
        logger.info("Évaluation Needle in a Haystack")
        needle_results = self.evaluate_needle_in_haystack()
        results["needle_in_haystack"] = needle_results

        # 4. RULER (Raisonnement à longue distance)
        logger.info("Évaluation RULER")
        ruler_results = self.evaluate_ruler()
        results["ruler"] = ruler_results

        return results

    def evaluate_position_wise_loss(self):
        """Évalue la perte par position dans le contexte"""
        results = {}

        # Pour chaque longueur de contexte
        for context_length in self.args.context_lengths:
            if context_length > self.model.config["max_position_embeddings"]:
                logger.info(f"Skipping context length {context_length} (exceeds model's max position embeddings)")
                continue

            logger.info(f"Évaluation de la perte position par position pour longueur {context_length}")

            # Créer un dataset pour cette longueur
            dataset = LongContextDataset(
                self.tokenizer,
                max_length=context_length,
                max_samples=min(20, self.args.max_samples)  # Moins d'échantillons pour les contextes longs
            )

            dataloader = DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                collate_fn=dataset.collate_fn
            )

            # Mesurer la perte par position
            position_losses = defaultdict(list)

            self.model.eval()
            with torch.no_grad():
                for batch in tqdm(dataloader, desc=f"Évaluation contexte {context_length}"):
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)

                    # Créer des labels (décalés d'une position)
                    labels = input_ids.clone()
                    labels[:, :-1] = input_ids[:, 1:]
                    labels[:, -1] = -100  # Ignorer le dernier token

                    # Forward pass
                    outputs = self.model(input_ids, attention_mask)
                    logits = outputs["prediction_logits"]

                    # Calculer la perte par position
                    for pos in range(min(context_length, input_ids.size(1))):
                        # Extraire les logits et labels pour cette position
                        pos_logits = logits[:, pos, :]
                        pos_labels = labels[:, pos]

                        # Ignorer les positions masquées
                        if (pos_labels != -100).sum() == 0:
                            continue

                        # Calculer la perte
                        loss = F.cross_entropy(pos_logits, pos_labels, ignore_index=-100, reduction="mean")

                        # Déterminer le bin de position
                        bin_idx = pos // 2000  # Regrouper par blocs de 2K tokens
                        bin_name = f"{bin_idx*2000}-{(bin_idx+1)*2000}"

                        position_losses[bin_name].append(loss.item())

            # Calculer la moyenne des pertes par bin
            avg_losses = {bin_name: np.mean(losses) for bin_name, losses in position_losses.items()}
            perplexities = {bin_name: np.exp(avg_loss) for bin_name, avg_loss in avg_losses.items()}

            results[str(context_length)] = {
                "losses": avg_losses,
                "perplexities": perplexities
            }

        return results

    def evaluate_trailing_token_loss(self):
        """Évalue la perte sur les derniers tokens des longues séquences"""
        results = {}

        # Pour chaque longueur de contexte
        for context_length in self.args.context_lengths:
            if context_length > self.model.config["max_position_embeddings"]:
                continue

            logger.info(f"Évaluation de la perte sur les tokens finaux pour longueur {context_length}")

            # Créer un dataset pour cette longueur
            dataset = LongContextDataset(
                self.tokenizer,
                max_length=context_length,
                max_samples=min(20, self.args.max_samples)
            )

            dataloader = DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                collate_fn=dataset.collate_fn
            )

            # Mesurer la perte sur les derniers 2K tokens
            trailing_losses = []

            self.model.eval()
            with torch.no_grad():
                for batch in tqdm(dataloader, desc=f"Évaluation trailing {context_length}"):
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)

                    # S'assurer que nous avons des séquences assez longues
                    if input_ids.size(1) < 2048:
                        continue

                    # Créer des labels (décalés d'une position)
                    labels = input_ids.clone()
                    labels[:, :-1] = input_ids[:, 1:]
                    labels[:, -1] = -100  # Ignorer le dernier token

                    # Forward pass
                    outputs = self.model(input_ids, attention_mask)
                    logits = outputs["prediction_logits"]

                    # Isoler les 2K derniers tokens
                    trailing_start = max(0, input_ids.size(1) - 2048)
                    trailing_logits = logits[:, trailing_start:, :]
                    trailing_labels = labels[:, trailing_start:]

                    # Calculer la perte
                    loss = F.cross_entropy(
                        trailing_logits.reshape(-1, trailing_logits.size(-1)),
                        trailing_labels.reshape(-1),
                        ignore_index=-100,
                        reduction="mean"
                    )

                    trailing_losses.append(loss.item())

            # Calculer la moyenne des pertes
            if trailing_losses:
                avg_loss = np.mean(trailing_losses)
                perplexity = np.exp(avg_loss)

                results[str(context_length)] = {
                    "loss": avg_loss,
                    "perplexity": perplexity
                }
            else:
                results[str(context_length)] = {
                    "loss": None,
                    "perplexity": None,
                    "note": "Pas assez de données pour calculer la perte"
                }

        return results

    def evaluate_needle_in_haystack(self):
        """Évalue la capacité à trouver de l'information dans un long contexte"""
        results = {}

        # Définir les longueurs de contexte pour ce test
        test_lengths = [8192, 16384, 32768]
        test_lengths = [l for l in test_lengths if l <= self.model.config["max_position_embeddings"]]

        if not test_lengths:
            logger.warning("Aucune longueur de contexte valide pour le test Needle in a Haystack")
            return {"status": "skipped"}

        # Pour chaque longueur de contexte
        for context_length in test_lengths:
            logger.info(f"Évaluation Needle in a Haystack pour longueur {context_length}")

            # Créer le dataset
            dataset = NeedleInHaystackDataset(
                self.tokenizer,
                max_length=context_length,
                num_samples=min(30, self.args.max_samples)
            )

            dataloader = DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                collate_fn=dataset.collate_fn
            )

            # Collecter les résultats par quartile de position
            position_results = {
                "q1": [],  # 0-25%
                "q2": [],  # 25-50%
                "q3": [],  # 50-75%
                "q4": []   # 75-100%
            }

            self.model.eval()
            with torch.no_grad():
                for batch in tqdm(dataloader, desc=f"Needle in Haystack {context_length}"):
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    needles = batch["needles"]
                    positions = batch["positions"]

                    # Forward pass
                    outputs = self.model(input_ids, attention_mask)
                    logits = outputs["prediction_logits"]

                    # Évaluer les résultats
                    for i, (pos, needle) in enumerate(zip(positions, needles)):
                        # Déterminer le quartile
                        quartile = "q1" if pos < 0.25 else "q2" if pos < 0.5 else "q3" if pos < 0.75 else "q4"

                        # Trouver où la question commence
                        text = batch["texts"][i]
                        question_start = text.find("Question: ")

                        if question_start == -1:
                            continue

                        # Isoler la partie après "Réponse:"
                        response_start = text.find("Réponse:", question_start)
                        if response_start == -1:
                            response_start = len(text) - 1

                        # Convertir en tokens
                        response_token_idx = len(self.tokenizer.encode(text[:response_start], add_special_tokens=True)) - 1

                        # Obtenir les prédictions pour les prochains tokens
                        next_token_logits = logits[i, response_token_idx:response_token_idx+5]

                        # Vérifier si les prédictions évoquent le contenu de l'aiguille
                        keywords = needle.split()
                        keywords = [k.lower() for k in keywords if len(k) > 3]  # Mots significatifs

                        if not keywords:
                            continue

                        # Prendre les top-5 tokens prédits
                        top_tokens = []
                        for token_logits in next_token_logits:
                            top5_values, top5_indices = torch.topk(token_logits, 5)
                            top_tokens.extend(self.tokenizer.convert_ids_to_tokens(top5_indices.tolist()))

                        # Vérifier si un des mots clés de l'aiguille apparaît dans les prédictions
                        found = False
                        for keyword in keywords:
                            if any(keyword.lower() in token.lower() for token in top_tokens):
                                found = True
                                break

                        position_results[quartile].append(int(found))

            # Calculer les taux de succès par quartile
            success_rates = {}
            for quartile, results_list in position_results.items():
                if results_list:
                    success_rates[quartile] = {
                        "success_rate": np.mean(results_list),
                        "count": len(results_list)
                    }
                else:
                    success_rates[quartile] = {
                        "success_rate": None,
                        "count": 0,
                        "note": "Pas d'exemples dans ce quartile"
                    }

            results[str(context_length)] = success_rates

        return results

    def evaluate_ruler(self):
        """Évalue la compréhension du raisonnement à longue distance"""
        results = {}

        # Définir les longueurs de contexte pour ce test
        test_lengths = [32768]
        test_lengths = [l for l in test_lengths if l <= self.model.config["max_position_embeddings"]]

        if not test_lengths:
            logger.warning("Aucune longueur de contexte valide pour le test RULER")
            return {"status": "skipped"}

        # Pour chaque longueur de contexte
        for context_length in test_lengths:
            logger.info(f"Évaluation RULER pour longueur {context_length}")

            # Créer le dataset
            dataset = RULERDataset(
                self.tokenizer,
                max_length=context_length,
                num_samples=min(20, self.args.max_samples)
            )

            dataloader = DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                collate_fn=dataset.collate_fn
            )

            # Collecter les résultats
            correct_answers = 0
            total_examples = 0

            self.model.eval()
            with torch.no_grad():
                for batch in tqdm(dataloader, desc=f"RULER {context_length}"):
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    answers = batch["answers"]

                    # Forward pass
                    outputs = self.model(input_ids, attention_mask)
                    logits = outputs["prediction_logits"]

                    # Évaluer les résultats
                    for i, answer in enumerate(answers):
                        # Trouver où la question finit
                        text = batch["texts"][i]
                        question_end = text.find("?") + 1

                        if question_end <= 0:
                            continue

                        # Convertir en tokens
                        response_token_idx = len(self.tokenizer.encode(text[:question_end], add_special_tokens=True)) - 1

                        # Obtenir les prédictions pour les prochains tokens
                        next_token_logits = logits[i, response_token_idx:response_token_idx+10]

                        # Convertir la réponse attendue en texte
                        expected_answer_str = str(answer)

                        # Décoder les tokens prédits
                        predicted_tokens = []
                        for token_logits in next_token_logits:
                            top_idx = torch.argmax(token_logits).item()
                            predicted_tokens.append(top_idx)

                        predicted_text = self.tokenizer.decode(predicted_tokens)

                        # Vérifier si la réponse attendue apparaît dans la prédiction
                        if expected_answer_str in predicted_text:
                            correct_answers += 1

                        total_examples += 1

            # Calculer le taux de succès
            if total_examples > 0:
                success_rate = correct_answers / total_examples
            else:
                success_rate = None

            results[str(context_length)] = {
                "success_rate": success_rate,
                "correct": correct_answers,
                "total": total_examples
            }

        return results

    def evaluate_efficiency(self):
        """Évalue l'efficacité computationnelle du modèle"""
        results = {}

        # 1. Mesurer la consommation mémoire
        logger.info("Mesure de la consommation mémoire")
        memory_usage = self.measure_memory_usage()
        results["memory_usage"] = memory_usage

        # 2. Mesurer les temps de calcul
        logger.info("Mesure des temps de calcul")
        computation_times = self.measure_computation_times()
        results["computation_times"] = computation_times

        # 3. Calculer le ratio d'accélération
        logger.info("Calcul du ratio d'accélération MOBA vs attention complète")
        speedup_ratio = self.calculate_speedup_ratio()
        results["speedup_ratio"] = speedup_ratio

        return results

    def measure_memory_usage(self):
        """Mesure la consommation mémoire pour différentes longueurs"""
        results = {}

        # Récupérer l'accès à la mémoire CUDA si disponible
        torch.cuda.empty_cache()

        for context_length in self.args.context_lengths:
            if context_length > self.model.config["max_position_embeddings"]:
                continue

            logger.info(f"Mesure de la consommation mémoire pour longueur {context_length}")

            # Créer un input de la taille souhaitée
            dummy_input = torch.randint(
                0, self.model.config["vocab_size"],
                (1, context_length),
                device=self.device
            )

            # Mesurer la mémoire avant
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                start_mem = torch.cuda.memory_allocated()

            # Forward pass
            self.model.eval()
            with torch.no_grad():
                # Mode MOBA
                model_to_test = self.model
                if hasattr(model_to_test, "module"):
                    model_to_test = model_to_test.module

                # Activer le mode MOBA
                if hasattr(model_to_test, "switch_to_hybrid_mode"):
                    model_to_test.switch_to_hybrid_mode(model_to_test.config["hybrid_layer_count"])

                _ = self.model(dummy_input)

                # Mesurer la mémoire après
                if torch.cuda.is_available():
                    peak_mem_moba = torch.cuda.max_memory_allocated() - start_mem
                    torch.cuda.empty_cache()
                else:
                    peak_mem_moba = None

                # Réinitialiser les statistiques de mémoire
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.empty_cache()
                    start_mem = torch.cuda.memory_allocated()

                # Mode attention complète
                if hasattr(model_to_test, "switch_to_full_attention"):
                    model_to_test.switch_to_full_attention()

                _ = self.model(dummy_input)

                # Mesurer la mémoire après
                if torch.cuda.is_available():
                    peak_mem_full = torch.cuda.max_memory_allocated() - start_mem
                    torch.cuda.empty_cache()
                else:
                    peak_mem_full = None

                # Restaurer le mode MOBA
                if hasattr(model_to_test, "switch_to_hybrid_mode"):
                    model_to_test.switch_to_hybrid_mode(model_to_test.config["hybrid_layer_count"])

            # Enregistrer les résultats
            results[str(context_length)] = {
                "moba_memory": peak_mem_moba / (1024 * 1024) if peak_mem_moba is not None else None,  # MB
                "full_memory": peak_mem_full / (1024 * 1024) if peak_mem_full is not None else None,  # MB
                "memory_reduction": (
                    (1 - peak_mem_moba / peak_mem_full) * 100
                    if peak_mem_moba is not None and peak_mem_full is not None else None
                ),  # Pourcentage
            }

        return results

    def measure_computation_times(self):
        """Mesure les temps de calcul pour différentes longueurs"""
        results = {}

        for context_length in self.args.context_lengths:
            if context_length > self.model.config["max_position_embeddings"]:
                continue

            logger.info(f"Mesure des temps de calcul pour longueur {context_length}")

            # Créer un input de la taille souhaitée
            dummy_input = torch.randint(
                0, self.model.config["vocab_size"],
                (1, context_length),
                device=self.device
            )

            # Mesurer le temps pour MOBA
            self.model.eval()

            # Mode MOBA
            model_to_test = self.model
            if hasattr(model_to_test, "module"):
                model_to_test = model_to_test.module

            # Activer le mode MOBA
            if hasattr(model_to_test, "switch_to_hybrid_mode"):
                model_to_test.switch_to_hybrid_mode(model_to_test.config["hybrid_layer_count"])

            # Warm-up
            with torch.no_grad():
                for _ in range(3):
                    _ = self.model(dummy_input)

            # Mesurer le temps
            torch.cuda.synchronize()
            start_time = time.time()

            with torch.no_grad():
                for _ in range(5):
                    _ = self.model(dummy_input)

            torch.cuda.synchronize()
            moba_time = (time.time() - start_time) / 5

            # Mode attention complète
            if hasattr(model_to_test, "switch_to_full_attention"):
                model_to_test.switch_to_full_attention()

            # Warm-up
            with torch.no_grad():
                for _ in range(3):
                    _ = self.model(dummy_input)

            # Mesurer le temps
            torch.cuda.synchronize()
            start_time = time.time()

            with torch.no_grad():
                for _ in range(5):
                    _ = self.model(dummy_input)

            torch.cuda.synchronize()
            full_time = (time.time() - start_time) / 5

            # Restaurer le mode MOBA
            if hasattr(model_to_test, "switch_to_hybrid_mode"):
                model_to_test.switch_to_hybrid_mode(model_to_test.config["hybrid_layer_count"])

            # Enregistrer les résultats
            results[str(context_length)] = {
                "moba_time": moba_time,
                "full_time": full_time,
                "speedup": full_time / moba_time if moba_time > 0 else None
            }

        return results

    def calculate_speedup_ratio(self):
        """Calcule le ratio d'accélération entre MOBA et attention complète"""
        # Utiliser les résultats déjà calculés des temps de calcul
        computation_times = self.measure_computation_times()

        # Extraire les speedups
        speedups = {
            length: data["speedup"]
            for length, data in computation_times.items()
            if "speedup" in data and data["speedup"] is not None
        }

        # Calculer des statistiques
        if speedups:
            avg_speedup = np.mean(list(speedups.values()))
            max_speedup = max(speedups.values())
            min_speedup = min(speedups.values())
        else:
            avg_speedup = None
            max_speedup = None
            min_speedup = None

        return {
            "per_length": speedups,
            "average": avg_speedup,
            "maximum": max_speedup,
            "minimum": min_speedup
        }

    def run_ablation_studies(self):
        """Exécute les études d'ablation sur différentes configurations MOBA"""
        results = {}

        # Définir une longueur de contexte de test pour les ablations
        test_length = 8192
        if test_length > self.model.config["max_position_embeddings"]:
            test_length = self.model.config["max_position_embeddings"]

        logger.info(f"Exécution des études d'ablation avec contexte de {test_length} tokens")

        # 1. Impact de la granularité des blocs
        logger.info("Ablation: Impact de la granularité des blocs")
        block_granularity_results = self.ablation_block_granularity(test_length)
        results["block_granularity"] = block_granularity_results

        # 2. Impact de la stratégie hybride
        logger.info("Ablation: Impact de la stratégie hybride")
        hybrid_strategy_results = self.ablation_hybrid_strategy(test_length)
        results["hybrid_strategy"] = hybrid_strategy_results

        # 3. Impact du choix des métriques pour le routage
        # Note: Cette ablation nécessiterait de modifier le code du modèle
        # pour supporter différentes métriques de routage
        logger.info("Ablation: Impact des métriques de routage non implémentée")
        results["routing_metrics"] = {"status": "not_implemented"}

        return results

    def ablation_block_granularity(self, context_length):
        """Étudie l'impact de la granularité des blocs"""
        results = {}

        # Créer un input de la taille souhaitée
        dummy_input = torch.randint(
            0, self.model.config["vocab_size"],
            (1, context_length),
            device=self.device
        )

        model_to_test = self.model
        if hasattr(model_to_test, "module"):
            model_to_test = model_to_test.module

        # Tester différentes configurations
        for config in self.args.moba_configs:
            block_size = config["block_size"]
            top_k = config["top_k"]

            logger.info(f"Test de granularité: block_size={block_size}, top_k={top_k}")

            # Mettre à jour la configuration
            if not hasattr(model_to_test, "config"):
                logger.warning("Le modèle n'a pas d'attribut config, impossible de modifier la configuration MOBA")
                continue

            # Sauvegarde de la configuration originale
            original_block_size = model_to_test.config.get("moba_block_size", 512)
            original_top_k = model_to_test.config.get("moba_top_k", 3)

            # Appliquer la nouvelle configuration
            model_to_test.config["moba_block_size"] = block_size
            model_to_test.config["moba_top_k"] = top_k

            # Réinitialiser les couches avec la nouvelle configuration
            if hasattr(model_to_test, "switch_to_hybrid_mode"):
                hybrid_count = model_to_test.config.get("hybrid_layer_count", 3)

                # Recréer les couches MoBA avec les nouveaux paramètres
                for i in range(len(model_to_test.layers)):
                    if i < (len(model_to_test.layers) - hybrid_count):
                        # Cette couche devrait utiliser MoBA
                        model_to_test.layers[i].use_moba = True
                        model_to_test.layers[i].attention = model_to_test.MoBAAttention(
                            hidden_size=model_to_test.config["hidden_size"],
                            num_heads=model_to_test.config["num_attention_heads"],
                            block_size=block_size,
                            top_k=top_k
                        )

            # Mesurer les performances
            self.model.eval()
            with torch.no_grad():
                # Mesurer le temps et la mémoire
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    start_mem = torch.cuda.memory_allocated()

                torch.cuda.synchronize()
                start_time = time.time()

                _ = self.model(dummy_input)

                torch.cuda.synchronize()
                end_time = time.time()

                if torch.cuda.is_available():
                    peak_mem = torch.cuda.max_memory_allocated() - start_mem
                    torch.cuda.empty_cache()
                else:
                    peak_mem = None

                # Mesurer la qualité (perte MLM)
                # Créer un dataset pour cette longueur
                dataset = LongContextDataset(
                    self.tokenizer,
                    max_length=context_length,
                    max_samples=5  # Petit échantillon pour l'ablation
                )

                dataloader = DataLoader(
                    dataset,
                    batch_size=1,
                    collate_fn=dataset.collate_fn
                )

                # Mesurer la perte
                losses = []

                for batch in dataloader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)

                    # Créer des labels (décalés d'une position)
                    labels = input_ids.clone()
                    labels[:, :-1] = input_ids[:, 1:]
                    labels[:, -1] = -100  # Ignorer le dernier token

                    # Forward pass
                    outputs = self.model(input_ids, attention_mask)
                    logits = outputs["prediction_logits"]

                    # Calculer la perte
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        labels.reshape(-1),
                        ignore_index=-100,
                        reduction="mean"
                    )

                    losses.append(loss.item())

                avg_loss = np.mean(losses) if losses else None

            # Enregistrer les résultats
            config_name = f"block{block_size}_top{top_k}"
            results[config_name] = {
                "block_size": block_size,
                "top_k": top_k,
                "time": end_time - start_time,
                "memory": peak_mem / (1024 * 1024) if peak_mem is not None else None,  # MB
                "loss": avg_loss,
                "perplexity": np.exp(avg_loss) if avg_loss is not None else None
            }

            # Restaurer la configuration originale
            model_to_test.config["moba_block_size"] = original_block_size
            model_to_test.config["moba_top_k"] = original_top_k

            # Restaurer les couches originales
            if hasattr(model_to_test, "switch_to_hybrid_mode"):
                hybrid_count = model_to_test.config.get("hybrid_layer_count", 3)

                # Recréer les couches MoBA avec les paramètres originaux
                for i in range(len(model_to_test.layers)):
                    if i < (len(model_to_test.layers) - hybrid_count):
                        # Cette couche devrait utiliser MoBA
                        model_to_test.layers[i].use_moba = True
                        model_to_test.layers[i].attention = model_to_test.MoBAAttention(
                            hidden_size=model_to_test.config["hidden_size"],
                            num_heads=model_to_test.config["num_attention_heads"],
                            block_size=original_block_size,
                            top_k=original_top_k
                        )

        return results

    def ablation_hybrid_strategy(self, context_length):
        """Étudie l'impact de la stratégie hybride"""
        results = {}

        # Créer un input de la taille souhaitée
        dummy_input = torch.randint(
            0, self.model.config["vocab_size"],
            (1, context_length),
            device=self.device
        )

        model_to_test = self.model
        if hasattr(model_to_test, "module"):
            model_to_test = model_to_test.module

        # Vérifier que le modèle supporte le changement de mode
        if not hasattr(model_to_test, "switch_to_hybrid_mode"):
            logger.warning("Le modèle ne supporte pas le changement de mode hybride")
            return {"status": "not_supported"}

        # Tester différentes configurations hybrides
        total_layers = len(model_to_test.layers)
        hybrid_counts = [0, 1, 3, 5, 10, total_layers]  # Nombre de couches d'attention complète

        for hybrid_count in hybrid_counts:
            if hybrid_count > total_layers:
                continue

            logger.info(f"Test de stratégie hybride: {hybrid_count} couches d'attention complète")

            # Appliquer la configuration
            model_to_test.switch_to_hybrid_mode(hybrid_count)

            # Mesurer les performances
            self.model.eval()
            with torch.no_grad():
                # Mesurer le temps et la mémoire
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    start_mem = torch.cuda.memory_allocated()

                torch.cuda.synchronize()
                start_time = time.time()

                _ = self.model(dummy_input)

                torch.cuda.synchronize()
                end_time = time.time()

                if torch.cuda.is_available():
                    peak_mem = torch.cuda.max_memory_allocated() - start_mem
                    torch.cuda.empty_cache()
                else:
                    peak_mem = None

                # Mesurer la qualité (perte MLM)
                # Créer un dataset pour cette longueur
                dataset = LongContextDataset(
                    self.tokenizer,
                    max_length=context_length,
                    max_samples=5  # Petit échantillon pour l'ablation
                )

                dataloader = DataLoader(
                    dataset,
                    batch_size=1,
                    collate_fn=dataset.collate_fn
                )

                # Mesurer la perte
                losses = []

                for batch in dataloader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)

                    # Créer des labels (décalés d'une position)
                    labels = input_ids.clone()
                    labels[:, :-1] = input_ids[:, 1:]
                    labels[:, -1] = -100  # Ignorer le dernier token

                    # Forward pass
                    outputs = self.model(input_ids, attention_mask)
                    logits = outputs["prediction_logits"]

                    # Calculer la perte
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        labels.reshape(-1),
                        ignore_index=-100,
                        reduction="mean"
                    )

                    losses.append(loss.item())

                avg_loss = np.mean(losses) if losses else None

            # Enregistrer les résultats
            config_name = f"hybrid{hybrid_count}"
            results[config_name] = {
                "hybrid_count": hybrid_count,
                "moba_count": total_layers - hybrid_count,
                "time": end_time - start_time,
                "memory": peak_mem / (1024 * 1024) if peak_mem is not None else None,  # MB
                "loss": avg_loss,
                "perplexity": np.exp(avg_loss) if avg_loss is not None else None
            }

        # Restaurer la configuration originale
        original_hybrid_count = model_to_test.config.get("hybrid_layer_count", 3)
        model_to_test.switch_to_hybrid_mode(original_hybrid_count)

        return results

    def generate_visualizations(self, results):
        """Génère des visualisations des résultats d'évaluation"""
        logger.info("Génération des visualisations des résultats")

        # Créer le répertoire pour les visualisations
        vis_dir = os.path.join(self.args.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        # 1. Position-wise Loss
        if "long_context" in results and "position_wise_loss" in results["long_context"]:
            self._visualize_position_loss(results["long_context"]["position_wise_loss"], vis_dir)

        # 2. Trailing Token Loss
        if "long_context" in results and "trailing_token_loss" in results["long_context"]:
            self._visualize_trailing_loss(results["long_context"]["trailing_token_loss"], vis_dir)

        # 3. Needle in a Haystack
        if "long_context" in results and "needle_in_haystack" in results["long_context"]:
            self._visualize_needle_haystack(results["long_context"]["needle_in_haystack"], vis_dir)

        # 4. Efficiency metrics
        if "efficiency" in results:
            self._visualize_efficiency(results["efficiency"], vis_dir)

        # 5. Ablation results
        if "ablation" in results:
            self._visualize_ablation(results["ablation"], vis_dir)

    def _visualize_position_loss(self, position_loss_results, vis_dir):
        """Visualise les résultats de perte par position"""
        plt.figure(figsize=(12, 8))

        for context_length, data in position_loss_results.items():
            if "perplexities" not in data:
                continue

            perplexities = data["perplexities"]
            positions = [int(pos.split("-")[0]) for pos in perplexities.keys()]
            ppl_values = list(perplexities.values())

            # Trier par position
            sorted_pairs = sorted(zip(positions, ppl_values))
            sorted_positions, sorted_ppls = zip(*sorted_pairs) if sorted_pairs else ([], [])

            plt.plot(sorted_positions, sorted_ppls, marker='o', label=f"Context {context_length}")

        plt.xlabel("Position (tokens)")
        plt.ylabel("Perplexity")
        plt.title("Perplexity by Position")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(vis_dir, "position_perplexity.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def _visualize_trailing_loss(self, trailing_loss_results, vis_dir):
        """Visualise les résultats de perte sur les tokens finaux"""
        plt.figure(figsize=(10, 6))

        context_lengths = []
        perplexities = []

        for context_length, data in trailing_loss_results.items():
            if "perplexity" not in data or data["perplexity"] is None:
                continue

            context_lengths.append(int(context_length))
            perplexities.append(data["perplexity"])

        # Trier par longueur de contexte
        sorted_pairs = sorted(zip(context_lengths, perplexities))
        sorted_lengths, sorted_ppls = zip(*sorted_pairs) if sorted_pairs else ([], [])

        plt.bar(range(len(sorted_lengths)), sorted_ppls, color='skyblue')
        plt.xticks(range(len(sorted_lengths)), [str(l) for l in sorted_lengths], rotation=45)
        plt.xlabel("Context Length")
        plt.ylabel("Trailing 2K Tokens Perplexity")
        plt.title("Perplexity of Trailing 2K Tokens by Context Length")
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "trailing_perplexity.png"), dpi=300)
        plt.close()

    def _visualize_needle_haystack(self, needle_results, vis_dir):
        """Visualise les résultats de Needle in a Haystack"""
        plt.figure(figsize=(12, 8))

        quartiles = ["q1", "q2", "q3", "q4"]
        quartile_labels = ["0-25%", "25-50%", "50-75%", "75-100%"]

        for context_length, data in needle_results.items():
            success_rates = []
            for q in quartiles:
                if q in data and "success_rate" in data[q] and data[q]["success_rate"] is not None:
                    success_rates.append(data[q]["success_rate"] * 100)  # En pourcentage
                else:
                    success_rates.append(0)

            x = range(len(quartiles))
            plt.plot(x, success_rates, marker='o', label=f"Context {context_length}")

        plt.xticks(range(len(quartiles)), quartile_labels)
        plt.xlabel("Position Percentile")
        plt.ylabel("Success Rate (%)")
        plt.title("Needle in a Haystack Success Rate by Position")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(0, 100)
        plt.savefig(os.path.join(vis_dir, "needle_haystack.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def _visualize_efficiency(self, efficiency_results, vis_dir):
        """Visualise les métriques d'efficacité"""
        # 1. Visualiser la consommation mémoire
        if "memory_usage" in efficiency_results:
            plt.figure(figsize=(10, 6))

            memory_data = efficiency_results["memory_usage"]
            context_lengths = []
            moba_memory = []
            full_memory = []

            for context_length, data in memory_data.items():
                if "moba_memory" in data and "full_memory" in data:
                    context_lengths.append(int(context_length))
                    moba_memory.append(data["moba_memory"])
                    full_memory.append(data["full_memory"])

            # Trier par longueur de contexte
            sorted_data = sorted(zip(context_lengths, moba_memory, full_memory))
            if sorted_data:
                context_lengths, moba_memory, full_memory = zip(*sorted_data)

                x = range(len(context_lengths))
                width = 0.35

                plt.bar([i - width/2 for i in x], moba_memory, width, label='MOBA')
                plt.bar([i + width/2 for i in x], full_memory, width, label='Full Attention')

                plt.xticks(x, [str(l) for l in context_lengths])
                plt.xlabel("Context Length")
                plt.ylabel("Memory Usage (MB)")
                plt.title("Memory Usage: MOBA vs Full Attention")
                plt.legend()
                plt.grid(True, axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, "memory_usage.png"), dpi=300)

            plt.close()

        # 2. Visualiser les temps de calcul
        if "computation_times" in efficiency_results:
            plt.figure(figsize=(10, 6))

            time_data = efficiency_results["computation_times"]
            context_lengths = []
            moba_times = []
            full_times = []

            for context_length, data in time_data.items():
                if "moba_time" in data and "full_time" in data:
                    context_lengths.append(int(context_length))
                    moba_times.append(data["moba_time"])
                    full_times.append(data["full_time"])

            # Trier par longueur de contexte
            sorted_data = sorted(zip(context_lengths, moba_times, full_times))
            if sorted_data:
                context_lengths, moba_times, full_times = zip(*sorted_data)

                x = range(len(context_lengths))
                width = 0.35

                plt.bar([i - width/2 for i in x], moba_times, width, label='MOBA')
                plt.bar([i + width/2 for i in x], full_times, width, label='Full Attention')

                plt.xticks(x, [str(l) for l in context_lengths])
                plt.xlabel("Context Length")
                plt.ylabel("Computation Time (s)")
                plt.title("Computation Time: MOBA vs Full Attention")
                plt.legend()
                plt.grid(True, axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, "computation_time.png"), dpi=300)

            plt.close()

        # 3. Visualiser le speedup
        if "speedup_ratio" in efficiency_results:
            plt.figure(figsize=(10, 6))

            speedup_data = efficiency_results["speedup_ratio"]
            if "per_length" in speedup_data:
                context_lengths = []
                speedups = []

                for context_length, speedup in speedup_data["per_length"].items():
                    context_lengths.append(int(context_length))
                    speedups.append(speedup)

                # Trier par longueur de contexte
                sorted_data = sorted(zip(context_lengths, speedups))
                if sorted_data:
                    context_lengths, speedups = zip(*sorted_data)

                    plt.plot(context_lengths, speedups, marker='o', linestyle='-', color='blue')

                    plt.xlabel("Context Length")
                    plt.ylabel("Speedup Ratio")
                    plt.title("MOBA Speedup Ratio vs Full Attention")
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.savefig(os.path.join(vis_dir, "speedup_ratio.png"), dpi=300, bbox_inches='tight')

            plt.close()

    def _visualize_ablation(self, ablation_results, vis_dir):
        """Visualise les résultats des études d'ablation"""
        # 1. Visualiser l'impact de la granularité des blocs
        if "block_granularity" in ablation_results:
            plt.figure(figsize=(12, 10))

            block_data = ablation_results["block_granularity"]
            configs = []
            times = []
            memories = []
            perplexities = []

            for config_name, data in block_data.items():
                if "time" in data and "memory" in data and "perplexity" in data:
                    configs.append(config_name)
                    times.append(data["time"])
                    memories.append(data["memory"])
                    perplexities.append(data["perplexity"])

            if configs:
                # Sous-figure pour le temps
                plt.subplot(3, 1, 1)
                plt.bar(configs, times, color='skyblue')
                plt.ylabel("Time (s)")
                plt.title("Time by Block Configuration")
                plt.xticks(rotation=45)
                plt.grid(True, axis='y', linestyle='--', alpha=0.7)

                # Sous-figure pour la mémoire
                plt.subplot(3, 1, 2)
                plt.bar(configs, memories, color='lightgreen')
                plt.ylabel("Memory (MB)")
                plt.title("Memory by Block Configuration")
                plt.xticks(rotation=45)
                plt.grid(True, axis='y', linestyle='--', alpha=0.7)

                # Sous-figure pour la perplexité
                plt.subplot(3, 1, 3)
                plt.bar(configs, perplexities, color='salmon')
                plt.ylabel("Perplexity")
                plt.title("Perplexity by Block Configuration")
                plt.xticks(rotation=45)
                plt.grid(True, axis='y', linestyle='--', alpha=0.7)

                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, "block_granularity.png"), dpi=300)

            plt.close()

        # 2. Visualiser l'impact de la stratégie hybride
        if "hybrid_strategy" in ablation_results:
            plt.figure(figsize=(12, 10))

            hybrid_data = ablation_results["hybrid_strategy"]
            configs = []
            times = []
            memories = []
            perplexities = []

            for config_name, data in hybrid_data.items():
                if "time" in data and "memory" in data and "perplexity" in data:
                    configs.append(config_name)
                    times.append(data["time"])
                    memories.append(data["memory"])
                    perplexities.append(data["perplexity"])

            if configs:
                # Sous-figure pour le temps
                plt.subplot(3, 1, 1)
                plt.bar(configs, times, color='skyblue')
                plt.ylabel("Time (s)")
                plt.title("Time by Hybrid Configuration")
                plt.xticks(rotation=45)
                plt.grid(True, axis='y', linestyle='--', alpha=0.7)

                # Sous-figure pour la mémoire
                plt.subplot(3, 1, 2)
                plt.bar(configs, memories, color='lightgreen')
                plt.ylabel("Memory (MB)")
                plt.title("Memory by Hybrid Configuration")
                plt.xticks(rotation=45)
                plt.grid(True, axis='y', linestyle='--', alpha=0.7)

                # Sous-figure pour la perplexité
                plt.subplot(3, 1, 3)
                plt.bar(configs, perplexities, color='salmon')
                plt.ylabel("Perplexity")
                plt.title("Perplexity by Hybrid Configuration")
                plt.xticks(rotation=45)
                plt.grid(True, axis='y', linestyle='--', alpha=0.7)

                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, "hybrid_strategy.png"), dpi=300)

            plt.close()


def main():
    """Fonction principale pour exécuter l'évaluation"""
    parser = argparse.ArgumentParser(description="Évaluation de NeoBERT-MOBA")
    parser.add_argument("--model_path", type=str, required=True, help="Chemin vers le checkpoint du modèle")
    parser.add_argument("--config_path", type=str, default=None, help="Chemin vers la configuration du modèle")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Chemin vers le tokenizer")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="Répertoire de sortie")

    parser.add_argument("--run_glue", action="store_true", help="Exécuter l'évaluation GLUE")
    parser.add_argument("--run_mteb", action="store_true", help="Exécuter l'évaluation MTEB")
    parser.add_argument("--run_long_context", action="store_true", help="Exécuter les évaluations de contexte long")
    parser.add_argument("--run_efficiency", action="store_true", help="Exécuter les évaluations d'efficacité")
    parser.add_argument("--run_ablation", action="store_true", help="Exécuter les études d'ablation")

    parser.add_argument("--context_lengths", type=int, nargs="+", default=[2048, 4096, 8192, 16384, 32768], help="Longueurs de contexte à évaluer")
    parser.add_argument("--max_samples", type=int, default=100, help="Nombre maximum d'échantillons")
    parser.add_argument("--batch_size", type=int, default=1, help="Taille du batch")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device à utiliser")

    args = parser.parse_args()

    # Convertir les arguments en EvaluationArguments
    eval_args = EvaluationArguments(
        model_path=args.model_path,
        config_path=args.config_path,
        tokenizer_path=args.tokenizer_path,
        output_dir=args.output_dir,

        run_glue=args.run_glue,
        run_mteb=args.run_mteb,
        run_long_context=args.run_long_context,
        run_efficiency=args.run_efficiency,
        run_ablation=args.run_ablation,

        context_lengths=args.context_lengths,
        max_samples=args.max_samples,
        batch_size=args.batch_size,

        device=args.device,
    )

    # Créer l'évaluateur
    evaluator = NeoBERTMoBAEvaluator(eval_args)

    # Exécuter l'évaluation
    results = evaluator.evaluate_all()

    logger.info(f"Évaluation terminée. Résultats sauvegardés dans {args.output_dir}")


if __name__ == "__main__":
    main()
    
    
# python evaluation.py --model_path ./checkpoints/neobert_moba_final \
#                      --run_long_context --run_efficiency --run_ablation \
#                      --context_lengths 2048 4096 8192 16384 32768 \
#                      --output_dir ./evaluation_results