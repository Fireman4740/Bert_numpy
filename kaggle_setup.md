# Guide d'utilisation sur Kaggle avec 2 GPUs T4

Ce guide explique comment configurer et exécuter l'entraînement de NeoBERT-MOBA sur Kaggle avec 2 GPUs T4.

## 1. Configuration du notebook Kaggle

Créez un nouveau notebook avec accélérateur GPU (2xT4) et ajoutez le code suivant au début :

```python
# Installation des dépendances
!pip install -q torch==2.0.1 accelerate datasets tokenizers tensorboard

# Définir les variables d'environnement pour DDP
import os
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"

# Vérifier les GPUs disponibles
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"Nombre de GPUs: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"Mémoire totale: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

# Clone du repo (si nécessaire)
# !git clone https://github.com/votre-repo/neobert-moba.git
# %cd neobert-moba

# Télécharger les fichiers Python (ou télécharger les fichiers manuellement dans Kaggle)
!wget -q https://raw.githubusercontent.com/votre-repo/neobert-moba/main/NeoBert_MOBA.py
!wget -q https://raw.githubusercontent.com/votre-repo/neobert-moba/main/data_process.py
!wget -q https://raw.githubusercontent.com/votre-repo/neobert-moba/main/training.py
!wget -q https://raw.githubusercontent.com/votre-repo/neobert-moba/main/requirements.txt
```

## 2. Préparation des données

```python
# Création du répertoire de données
!mkdir -p data

# Importation des bibliothèques nécessaires
from data_process import DataPreparation
import logging

logging.getLogger().setLevel(logging.INFO)

# Initialisation de la préparation des données
data_prep = DataPreparation(
    base_dir="./data",
    vocab_size=30000,
    max_length=16384,  # Réduit pour T4
    mlm_probability=0.15
)

# Téléchargement des données (utilise RefinedWeb)
datasets = data_prep.download_data()

# Entraînement du tokenizer
tokenizer = data_prep.train_tokenizer(datasets["main"])

# Tokenisation et préparation des datasets
tokenized_main = data_prep.tokenize_dataset(datasets["main"])
sequence_datasets = data_prep.create_sequence_datasets(
    tokenized_main,
    lengths=[1024, 2048, 4096]  # Limité à 4096 pour T4
)
```

## 3. Lancement de l'entraînement en mode T4

### Phase 1 : Pré-entraînement initial

```python
# Exécution avec les options pour Kaggle
!python -m torch.distributed.launch --nproc_per_node=2 training.py --phase 1 --kaggle --data_dir ./data --local_rank 0
```

### Phase 2 : Extension du contexte

```python
# Après la phase 1, exécuter la phase 2
!python -m torch.distributed.launch --nproc_per_node=2 training.py --phase 2 --kaggle --data_dir ./data --continue_from_checkpoint ./outputs/phase1/checkpoints/final --local_rank 0
```

### Phase 3 : Activation MOBA

```python
# Après la phase 2, exécuter la phase 3
!python -m torch.distributed.launch --nproc_per_node=2 training.py --phase 3 --kaggle --data_dir ./data --continue_from_checkpoint ./outputs/phase2/checkpoints/final --local_rank 0
```

## 4. Suivi et visualisation

Pour visualiser les métriques d'entraînement pendant ou après l'exécution :

```python
# Chargement de TensorBoard
%load_ext tensorboard
%tensorboard --logdir ./logs
```

## 5. Sauvegarder les résultats

À la fin de l'entraînement, sauvegardez les modèles et logs vers un bucket de stockage ou téléchargez-les :

```python
# Compresser les résultats
!tar -czf neobert_moba_results.tar.gz outputs/ logs/

# Télécharger les résultats (disponibles dans l'onglet "Output" de Kaggle)
from IPython.display import FileLink
FileLink(r'neobert_moba_results.tar.gz')
```

## Notes importantes pour l'environnement Kaggle

1. **Durée de session** : Les sessions Kaggle ont une durée limitée (9h max). Prévoyez des checkpoints fréquents.
2. **Mémoire GPU** : Chaque T4 dispose d'environ 16GB de VRAM.
3. **Variables d'environnement DDP** : Assurez-vous que `MASTER_ADDR` et `MASTER_PORT` sont correctement définis pour Distributed Data Parallel.
4. **Persistance des données** : Sauvegardez régulièrement vos modèles dans le "Output" de Kaggle.
