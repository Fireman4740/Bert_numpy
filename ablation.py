import os
import json
import time
import torch
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import math

# Supposez que ces imports proviennent de vos modules précédents
from model_architecture import NeoBERTMoBA, create_neobert_moba_model, MoBAAttention, RotaryEmbedding
from data_preparation import DataPreparation

# Configuration du logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class CustomMoBAAttention(nn.Module):
    """MoBA avec différentes métriques de routage"""
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        block_size: int = 512,
        top_k: int = 3,
        routing_metric: str = "mean_pooling",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.block_size = block_size
        self.top_k = top_k
        self.routing_metric = routing_metric
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.rotary = RotaryEmbedding(self.head_dim)
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def _split_heads(self, x):
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)
    
    def _merge_heads(self, x):
        x = x.permute(0, 2, 1, 3)
        new_shape = x.size()[:-2] + (self.hidden_size,)
        return x.reshape(*new_shape)
    
    def _compute_block_representation(self, keys, block_indices):
        """
        Calcule la représentation d'un bloc selon la métrique choisie
        """
        if len(block_indices) == 0:
            return torch.zeros(keys.size(0), keys.size(1), 1, keys.size(3), device=keys.device)
            
        keys_in_block = keys[:, :, block_indices]
        
        if self.routing_metric == "mean_pooling":
            return keys_in_block.mean(dim=2, keepdim=True)
        
        elif self.routing_metric == "max_pooling":
            return keys_in_block.max(dim=2, keepdim=True)[0]
        
        elif self.routing_metric == "min_max_pooling":
            max_values = keys_in_block.max(dim=2, keepdim=True)[0]
            min_values = keys_in_block.min(dim=2, keepdim=True)[0]
            return (max_values + min_values) / 2
        
        elif self.routing_metric == "representative_token":
            # Utiliser le milieu du bloc comme token représentatif
            middle_idx = len(block_indices) // 2
            return keys[:, :, block_indices[middle_idx]:block_indices[middle_idx]+1]
        
        else:
            raise ValueError(f"Métrique de routage non supportée: {self.routing_metric}")
    
    def forward(self, hidden_states, attention_mask=None, past_key_value=None):
        batch_size, seq_len = hidden_states.shape[:2]

        # Linear projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Split into heads
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # Apply RoPE to q and k
        q = self.rotary(q, seq_len)
        k = self.rotary(k, seq_len)

        # Split sequence into blocks
        num_blocks = math.ceil(seq_len / self.block_size)

        # Compute block representations for gating
        block_reps = []
        for i in range(num_blocks):
            start_idx = i * self.block_size
            end_idx = min(start_idx + self.block_size, seq_len)
            if end_idx > start_idx:
                block_indices = list(range(start_idx, end_idx))
                block_rep = self._compute_block_representation(k, block_indices)
                block_reps.append(block_rep)

        # Stack block representations
        block_reps = torch.cat(block_reps, dim=2)

        # Initialize output tensor
        attn_output = torch.zeros_like(q)

        # Process each query token
        for i in range(seq_len):
            # Get the current query
            query = q[:, :, i:i+1]

            # Compute relevance scores between query and blocks
            block_scores = torch.matmul(query, block_reps.transpose(-1, -2)) * self.scale

            # Apply causal mask - query can only attend to blocks up to its position
            current_block = i // self.block_size
            causal_mask = torch.arange(num_blocks, device=query.device) > current_block
            block_scores = block_scores.masked_fill(
                causal_mask.view(1, 1, 1, -1), float('-inf')
            )

            # Always include the current block
            block_scores[:, :, :, current_block] = torch.finfo(block_scores.dtype).max

            # Get top-k blocks
            _, top_k_indices = torch.topk(block_scores, min(self.top_k, num_blocks), dim=-1)

            # Get keys and values for selected blocks
            selected_k = []
            selected_v = []

            for b in range(batch_size):
                for h in range(self.num_heads):
                    k_for_query = []
                    v_for_query = []

                    for block_idx in top_k_indices[b, h, 0]:
                        block_idx = block_idx.item()
                        start_idx = block_idx * self.block_size
                        end_idx = min(start_idx + self.block_size, seq_len)

                        # For the current block, apply causal masking
                        if block_idx == current_block:
                            k_block = k[b, h:h+1, start_idx:i+1]
                            v_block = v[b, h:h+1, start_idx:i+1]
                        else:
                            k_block = k[b, h:h+1, start_idx:end_idx]
                            v_block = v[b, h:h+1, start_idx:end_idx]

                        if k_block.size(2) > 0:
                            k_for_query.append(k_block)
                            v_for_query.append(v_block)

                    if k_for_query:
                        # Concatenate selected keys and values
                        k_concat = torch.cat(k_for_query, dim=2)
                        v_concat = torch.cat(v_for_query, dim=2)

                        # Compute attention scores
                        attn_scores = torch.matmul(query[b:b+1, h:h+1], k_concat.transpose(-1, -2)) * self.scale
                        attn_probs = F.softmax(attn_scores, dim=-1)

                        # Apply attention
                        attn_output[b:b+1, h:h+1, i:i+1] = torch.matmul(attn_probs, v_concat)

        # Merge heads and project back
        attn_output = self._merge_heads(attn_output)
        attn_output = self.o_proj(attn_output)

        return attn_output


class NeoBERTMoBAWithCustomRouting(NeoBERTMoBA):
    """NeoBERT-MOBA avec métrique de routage personnalisée"""
    def __init__(self, *args, routing_metric="mean_pooling", **kwargs):
        super().__init__(*args, **kwargs)
        self.routing_metric = routing_metric
        
        # Recréer les couches d'attention avec la métrique personnalisée
        hybrid_count = self.config.get("hybrid_layer_count", 3)
        
        for i in range(len(self.layers)):
            if i < (len(self.layers) - hybrid_count):
                # Cette couche utilise MoBA
                self.layers[i].attention = CustomMoBAAttention(
                    hidden_size=self.config["hidden_size"],
                    num_heads=self.config["num_attention_heads"],
                    block_size=self.config["moba_block_size"],
                    top_k=self.config["moba_top_k"],
                    routing_metric=routing_metric
                )


def run_block_granularity_ablation(
    base_model_path,
    output_dir,
    context_length=32768,
    batch_size=1,
    max_samples=5,
    device="cuda"
):
    """
    Étude d'ablation sur la granularité des blocs MOBA
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Configurations à tester
    configurations = [
        {"blocks": 8, "top_k": 2, "name": "8_blocks_top2"},
        {"blocks": 16, "top_k": 4, "name": "16_blocks_top4"},
        {"blocks": 32, "top_k": 8, "name": "32_blocks_top8"},
        {"blocks": 64, "top_k": 16, "name": "64_blocks_top16"},
        {"blocks": 128, "top_k": 32, "name": "128_blocks_top32"},
    ]
    
    # Charger la configuration du modèle
    with open(os.path.join(base_model_path, "config.json"), "r") as f:
        model_config = json.load(f)
    
    # Préparer les données d'évaluation
    data_prep = DataPreparation(max_length=context_length)
    eval_dataset = data_prep.create_evaluation_dataset(
        dataset_name="pg19", 
        split="test",
        max_samples=max_samples
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        collate_fn=eval_dataset.collate_fn
    )
    
    # Résultats
    results = {}
    
    # Tester chaque configuration
    for config in configurations:
        logger.info(f"Évaluation de la configuration: {config['name']}")
        
        # Calculer la taille du bloc pour cette configuration
        block_size = context_length // config["blocks"]
        
        # Mettre à jour la configuration
        test_config = model_config.copy()
        test_config["moba_block_size"] = block_size
        test_config["moba_top_k"] = config["top_k"]
        
        # Charger le modèle
        model = create_neobert_moba_model(test_config)
        model.load_state_dict(torch.load(os.path.join(base_model_path, "pytorch_model.bin")))
        model = model.to(device)
        model.eval()
        
        # Mesurer performance
        time_measurements = []
        memory_usage = []
        losses = []
        
        for batch in tqdm(eval_dataloader, desc=f"Évaluation {config['name']}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Créer labels (décalés d'une position)
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = -100  # Ignorer le dernier token
            
            # Mesurer mémoire et temps
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                start_mem = torch.cuda.memory_allocated()
            
            torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model(input_ids, attention_mask)
                logits = outputs["prediction_logits"]
                
                # Calculer perte
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                    reduction="mean"
                )
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            if torch.cuda.is_available():
                peak_mem = torch.cuda.max_memory_allocated() - start_mem
                memory_usage.append(peak_mem / (1024 * 1024))  # MB
            
            time_measurements.append(end_time - start_time)
            losses.append(loss.item())
        
        # Calculer moyennes
        avg_time = np.mean(time_measurements)
        avg_memory = np.mean(memory_usage) if memory_usage else None
        avg_loss = np.mean(losses)
        perplexity = np.exp(avg_loss)
        
        # Stocker résultats
        results[config['name']] = {
            "blocks": config["blocks"],
            "top_k": config["top_k"],
            "block_size": block_size,
            "sparsity": 1 - (config["top_k"] / config["blocks"]),
            "avg_time": avg_time,
            "avg_memory": avg_memory,
            "avg_loss": avg_loss,
            "perplexity": perplexity
        }
        
        logger.info(f"Résultats pour {config['name']}:")
        logger.info(f"  Temps moyen: {avg_time:.4f} s")
        logger.info(f"  Mémoire moyenne: {avg_memory:.2f} MB" if avg_memory else "  Mémoire: N/A")
        logger.info(f"  Perte moyenne: {avg_loss:.4f}")
        logger.info(f"  Perplexité: {perplexity:.4f}")
    
    # Sauvegarder résultats
    with open(os.path.join(output_dir, "block_granularity_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Créer visualisations
    create_block_granularity_visualizations(results, output_dir)
    
    return results


def create_block_granularity_visualizations(results, output_dir):
    """Crée des visualisations pour les résultats d'ablation de granularité"""
    plt.figure(figsize=(15, 10))
    
    # Extraire données
    configs = list(results.keys())
    blocks = [results[c]["blocks"] for c in configs]
    times = [results[c]["avg_time"] for c in configs]
    memories = [results[c]["avg_memory"] for c in configs if results[c]["avg_memory"] is not None]
    perplexities = [results[c]["perplexity"] for c in configs]
    
    # Sous-graphique pour le temps
    plt.subplot(2, 2, 1)
    plt.bar(configs, times, color='skyblue')
    plt.ylabel("Temps (s)")
    plt.title("Temps d'inférence par configuration")
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Sous-graphique pour la mémoire
    if memories:
        plt.subplot(2, 2, 2)
        plt.bar(configs[:len(memories)], memories, color='lightgreen')
        plt.ylabel("Mémoire (MB)")
        plt.title("Utilisation mémoire par configuration")
        plt.xticks(rotation=45)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Sous-graphique pour la perplexité
    plt.subplot(2, 2, 3)
    plt.bar(configs, perplexities, color='salmon')
    plt.ylabel("Perplexité")
    plt.title("Perplexité par configuration")
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Sous-graphique pour la relation blocks vs perplexité
    plt.subplot(2, 2, 4)
    plt.plot(blocks, perplexities, marker='o', linestyle='-', color='purple')
    plt.xlabel("Nombre de blocs")
    plt.ylabel("Perplexité")
    plt.title("Relation entre nombre de blocs et perplexité")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "block_granularity_comparison.png"), dpi=300)


def run_hybrid_strategy_ablation(
    base_model_path,
    output_dir,
    context_length=32768,
    batch_size=1,
    max_samples=5,
    device="cuda"
):
    """
    Étude d'ablation sur la stratégie hybride
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Charger configuration modèle
    with open(os.path.join(base_model_path, "config.json"), "r") as f:
        model_config = json.load(f)
    
    # Nombre total de couches
    total_layers = model_config["num_hidden_layers"]
    
    # Configurations à tester
    hybrid_configs = [
        {"hybrid_count": 0, "name": "full_moba"},
        {"hybrid_count": 1, "name": "hybrid_1"},
        {"hybrid_count": 3, "name": "hybrid_3"},
        {"hybrid_count": 5, "name": "hybrid_5"},
        {"hybrid_count": 10, "name": "hybrid_10"},
        {"hybrid_count": total_layers, "name": "full_attention"}
    ]
    
    # Préparer données évaluation
    data_prep = DataPreparation(max_length=context_length)
    eval_dataset = data_prep.create_evaluation_dataset(
        dataset_name="pg19", 
        split="test",
        max_samples=max_samples
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        collate_fn=eval_dataset.collate_fn
    )
    
    # Résultats
    results = {}
    
    # Tester chaque configuration
    for config in hybrid_configs:
        logger.info(f"Évaluation de la configuration: {config['name']}")
        
        # Charger modèle
        model = create_neobert_moba_model(model_config)
        model.load_state_dict(torch.load(os.path.join(base_model_path, "pytorch_model.bin")))
        model = model.to(device)
        model.eval()
        
        # Appliquer stratégie hybride
        model.switch_to_hybrid_mode(config["hybrid_count"])
        
        # Mesurer performance
        time_measurements = []
        memory_usage = []
        losses = []
        
        for batch in tqdm(eval_dataloader, desc=f"Évaluation {config['name']}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Créer labels
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = -100
            
            # Mesurer mémoire et temps
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                start_mem = torch.cuda.memory_allocated()
            
            torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model(input_ids, attention_mask)
                logits = outputs["prediction_logits"]
                
                # Calculer perte
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                    reduction="mean"
                )
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            if torch.cuda.is_available():
                peak_mem = torch.cuda.max_memory_allocated() - start_mem
                memory_usage.append(peak_mem / (1024 * 1024))
            
            time_measurements.append(end_time - start_time)
            losses.append(loss.item())
        
        # Calculer moyennes
        avg_time = np.mean(time_measurements)
        avg_memory = np.mean(memory_usage) if memory_usage else None
        avg_loss = np.mean(losses)
        perplexity = np.exp(avg_loss)
        
        # Stocker résultats
        results[config['name']] = {
            "hybrid_count": config["hybrid_count"],
            "moba_count": total_layers - config["hybrid_count"],
            "moba_percentage": (total_layers - config["hybrid_count"]) / total_layers * 100,
            "avg_time": avg_time,
            "avg_memory": avg_memory,
            "avg_loss": avg_loss,
            "perplexity": perplexity
        }
        
        logger.info(f"Résultats pour {config['name']}:")
        logger.info(f"  Temps moyen: {avg_time:.4f} s")
        logger.info(f"  Mémoire moyenne: {avg_memory:.2f} MB" if avg_memory else "  Mémoire: N/A")
        logger.info(f"  Perte moyenne: {avg_loss:.4f}")
        logger.info(f"  Perplexité: {perplexity:.4f}")
    
    # Sauvegarder résultats
    with open(os.path.join(output_dir, "hybrid_strategy_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Créer visualisations
    create_hybrid_strategy_visualizations(results, output_dir, total_layers)
    
    return results


def create_hybrid_strategy_visualizations(results, output_dir, total_layers):
    """Crée des visualisations pour les résultats d'ablation de stratégie hybride"""
    plt.figure(figsize=(15, 10))
    
    # Extraire données
    configs = list(results.keys())
    hybrid_counts = [results[c]["hybrid_count"] for c in configs]
    moba_percentages = [results[c]["moba_percentage"] for c in configs]
    times = [results[c]["avg_time"] for c in configs]
    memories = [results[c]["avg_memory"] for c in configs if results[c]["avg_memory"] is not None]
    perplexities = [results[c]["perplexity"] for c in configs]
    
    # Sous-graphique pour le temps
    plt.subplot(2, 2, 1)
    plt.plot(moba_percentages, times, marker='o', linestyle='-', color='blue')
    plt.xlabel("% de couches MOBA")
    plt.ylabel("Temps (s)")
    plt.title("Temps d'inférence vs % de couches MOBA")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Sous-graphique pour la mémoire
    if memories:
        plt.subplot(2, 2, 2)
        plt.plot(moba_percentages[:len(memories)], memories, marker='o', linestyle='-', color='green')
        plt.xlabel("% de couches MOBA")
        plt.ylabel("Mémoire (MB)")
        plt.title("Utilisation mémoire vs % de couches MOBA")
        plt.grid(True, linestyle='--', alpha=0.7)
    
    # Sous-graphique pour la perplexité
    plt.subplot(2, 2, 3)
    plt.plot(moba_percentages, perplexities, marker='o', linestyle='-', color='red')
    plt.xlabel("% de couches MOBA")
    plt.ylabel("Perplexité")
    plt.title("Perplexité vs % de couches MOBA")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Sous-graphique pour le compromis perplexité/temps
    plt.subplot(2, 2, 4)
    plt.scatter(times, perplexities, c=moba_percentages, cmap='viridis', s=100)
    
    # Ajouter étiquettes pour chaque point
    for i, config in enumerate(configs):
        plt.annotate(config, (times[i], perplexities[i]), 
                     xytext=(5, 5), textcoords='offset points')
    
    plt.colorbar(label="% de couches MOBA")
    plt.xlabel("Temps (s)")
    plt.ylabel("Perplexité")
    plt.title("Compromis perplexité/temps")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hybrid_strategy_comparison.png"), dpi=300)


def run_routing_metrics_ablation(
    base_model_path,
    output_dir,
    context_length=32768,
    batch_size=1,
    max_samples=5,
    device="cuda"
):
    """
    Étude d'ablation sur les métriques de routage
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Charger configuration modèle
    with open(os.path.join(base_model_path, "config.json"), "r") as f:
        model_config = json.load(f)
    
    # Métriques de routage à tester
    routing_metrics = [
        "mean_pooling",
        "max_pooling",
        "min_max_pooling",
        "representative_token"
    ]
    
    # Préparer données évaluation
    data_prep = DataPreparation(max_length=context_length)
    eval_dataset = data_prep.create_evaluation_dataset(
        dataset_name="pg19", 
        split="test",
        max_samples=max_samples
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        collate_fn=eval_dataset.collate_fn
    )
    
    # Résultats
    results = {}
    
    # Tester chaque métrique
    for metric in routing_metrics:
        logger.info(f"Évaluation de la métrique de routage: {metric}")
        
        # Créer modèle avec métrique personnalisée
        model = NeoBERTMoBAWithCustomRouting(
            vocab_size=model_config["vocab_size"],
            hidden_size=model_config["hidden_size"],
            num_hidden_layers=model_config["num_hidden_layers"],
            num_attention_heads=model_config["num_attention_heads"],
            max_position_embeddings=model_config["max_position_embeddings"],
            moba_block_size=model_config["moba_block_size"],
            moba_top_k=model_config["moba_top_k"],
            hybrid_layer_count=model_config["hybrid_layer_count"],
            routing_metric=metric
        )
        
        # Charger poids
        model.load_state_dict(torch.load(os.path.join(base_model_path, "pytorch_model.bin")))
        model = model.to(device)
        model.eval()
        
        # Mesurer performance
        time_measurements = []
        losses = []
        
        for batch in tqdm(eval_dataloader, desc=f"Évaluation {metric}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Créer labels
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = -100
            
            torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model(input_ids, attention_mask)
                logits = outputs["prediction_logits"]
                
                # Calculer perte
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                    reduction="mean"
                )
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            time_measurements.append(end_time - start_time)
            losses.append(loss.item())
        
        # Calculer moyennes
        avg_time = np.mean(time_measurements)
        avg_loss = np.mean(losses)
        perplexity = np.exp(avg_loss)
        
        # Stocker résultats
        results[metric] = {
            "avg_time": avg_time,
            "avg_loss": avg_loss,
            "perplexity": perplexity
        }
        
        logger.info(f"Résultats pour {metric}:")
        logger.info(f"  Temps moyen: {avg_time:.4f} s")
        logger.info(f"  Perte moyenne: {avg_loss:.4f}")
        logger.info(f"  Perplexité: {perplexity:.4f}")
    
    # Sauvegarder résultats
    with open(os.path.join(output_dir, "routing_metrics_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Créer visualisations
    create_routing_metrics_visualizations(results, output_dir)
    
    return results


def create_routing_metrics_visualizations(results, output_dir):
    """Crée des visualisations pour les résultats d'ablation de métriques de routage"""
    plt.figure(figsize=(15, 5))
    
    # Extraire données
    metrics = list(results.keys())
    times = [results[m]["avg_time"] for m in metrics]
    perplexities = [results[m]["perplexity"] for m in metrics]
    
    # Sous-graphique pour le temps
    plt.subplot(1, 2, 1)
    bars = plt.bar(metrics, times, color='skyblue')
    plt.ylabel("Temps (s)")
    plt.title("Temps d'inférence par métrique de routage")
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Ajouter valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}',
                 ha='center', va='bottom')
    
    # Sous-graphique pour la perplexité
    plt.subplot(1, 2, 2)
    bars = plt.bar(metrics, perplexities, color='salmon')
    plt.ylabel("Perplexité")
    plt.title("Perplexité par métrique de routage")
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Ajouter valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}',
                 ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "routing_metrics_comparison.png"), dpi=300)


def generate_development_roadmap(ablation_results_dir, output_file):
    """
    Génère un rapport sur la suite du développement basé sur les résultats d'ablation
    """
    # Charger résultats des ablations
    results = {}
    
    # Charger résultats de granularité
    granularity_file = os.path.join(ablation_results_dir, "block_granularity", "block_granularity_results.json")
    if os.path.exists(granularity_file):
        with open(granularity_file, "r") as f:
            results["granularity"] = json.load(f)
    
    # Charger résultats de stratégie hybride
    hybrid_file = os.path.join(ablation_results_dir, "hybrid_strategy", "hybrid_strategy_results.json")
    if os.path.exists(hybrid_file):
        with open(hybrid_file, "r") as f:
            results["hybrid"] = json.load(f)
    
    # Charger résultats des métriques de routage
    routing_file = os.path.join(ablation_results_dir, "routing_metrics", "routing_metrics_results.json")
    if os.path.exists(routing_file):
        with open(routing_file, "r") as f:
            results["routing"] = json.load(f)
    
    # Analyser résultats pour recommandations
    recommendations = {}
    
    # Analyser granularité des blocs
    if "granularity" in results:
        granularity_data = results["granularity"]
        
        # Trouver meilleure configuration (perplexité)
        best_perplexity = float('inf')
        best_config = None
        
        for config, data in granularity_data.items():
            if data["perplexity"] < best_perplexity:
                best_perplexity = data["perplexity"]
                best_config = config
        
        if best_config:
            recommendations["block_size"] = granularity_data[best_config]["block_size"]
            recommendations["blocks"] = granularity_data[best_config]["blocks"]
            recommendations["top_k"] = granularity_data[best_config]["top_k"]
    
    # Analyser stratégie hybride
    if "hybrid" in results:
        hybrid_data = results["hybrid"]
        
        # Trouver meilleur compromis perplexité/vitesse
        configs = []
        for config, data in hybrid_data.items():
            configs.append({
                "name": config,
                "hybrid_count": data["hybrid_count"],
                "perplexity": data["perplexity"],
                "time": data["avg_time"]
            })
        
        # Trier par perplexité
        configs.sort(key=lambda x: x["perplexity"])
        
        # Prendre configuration avec meilleure perplexité parmi les 3 premières
        # qui a aussi un bon temps d'exécution
        best_hybrid = configs[0]
        for i in range(1, min(3, len(configs))):
            if configs[i]["time"] < best_hybrid["time"] * 0.8:  # 20% plus rapide
                best_hybrid = configs[i]
        
        recommendations["hybrid_count"] = best_hybrid["hybrid_count"]
    
    # Analyser métriques de routage
    if "routing" in results:
        routing_data = results["routing"]
        
        # Trouver métrique avec meilleure perplexité
        best_perplexity = float('inf')
        best_metric = None
        
        for metric, data in routing_data.items():
            if data["perplexity"] < best_perplexity:
                best_perplexity = data["perplexity"]
                best_metric = metric
        
        if best_metric:
            recommendations["routing_metric"] = best_metric
    
    # Générer rapport
    report = f"""# Feuille de route pour le développement futur de NeoBERT-MOBA

## Recommandations basées sur les expériences d'ablation

### 1. Ajustement des hyperparamètres MOBA

Sur la base des expériences d'ablation, les paramètres optimaux recommandés sont:

"""
    
    if "blocks" in recommendations and "top_k" in recommendations:
        report += f"- **Nombre de blocs**: {recommendations['blocks']}\n"
        report += f"- **Taille de bloc**: {recommendations['block_size']} tokens\n"
        report += f"- **Paramètre top-k**: {recommendations['top_k']}\n"
    else:
        report += "- Des tests supplémentaires sont nécessaires pour déterminer la granularité optimale des blocs.\n"
    
    if "hybrid_count" in recommendations:
        report += f"- **Nombre de couches d'attention complète**: {recommendations['hybrid_count']}\n"
    else:
        report += "- Des tests supplémentaires sont nécessaires pour déterminer la stratégie hybride optimale.\n"
    
    if "routing_metric" in recommendations:
        report += f"- **Métrique de routage recommandée**: {recommendations['routing_metric']}\n"
    else:
        report += "- Des tests supplémentaires sont nécessaires pour déterminer la métrique de routage optimale.\n"

    report += """
### 2. Optimisation de l'implémentation

Pour améliorer les performances sur différents matériels:

- Optimiser l'implémentation CUDA pour améliorer l'efficacité sur GPU
- Explorer les optimisations pour CPU (AVX, OpenMP)
- Adapter l'implémentation pour hardware spécialisé (TPU, NPU)
- Optimiser la gestion mémoire pour réduire l'empreinte mémoire lors de l'inférence
- Intégrer avec les frameworks d'inférence comme ONNX Runtime, TensorRT

### 3. Extension à des contextes plus longs

Pour atteindre et dépasser 1M tokens:

- Tester des stratégies de mise à l'échelle pour adapter la taille des blocs et le paramètre top-k
- Explorer des techniques de compression mémoire pour stocker l'historique
- Développer une version hiérarchique de MOBA avec plusieurs niveaux de granularité
- Expérimenter avec des techniques d'attention récursive pour les très longs contextes
- Intégrer avec des méthodes d'indexation ou de recherche pour les contextes extrêmement longs (>10M tokens)

### 4. Adaptation à d'autres langues

Pour améliorer les performances multilingues:

- Tester MOBA avec des tokenizers spécifiques aux langues
- Ajuster la granularité des blocs pour les langues aux caractéristiques différentes
- Évaluer l'impact des différentes longueurs moyennes de mots selon les langues
- Développer des ensembles de données d'évaluation spécifiques aux langues pour les longs contextes
- Explorer des techniques d'adaptation par langue pour les métriques de routage

### 5. Prochaines étapes immédiates

Actions prioritaires à court terme:

1. Implémenter les hyperparamètres optimaux identifiés dans les expériences d'ablation
2. Réaliser des tests d'efficacité et de perplexité sur un ensemble plus large de benchmarks
3. Développer une version optimisée de l'implémentation CUDA
4. Tester la scalabilité avec des séquences jusqu'à 2M tokens
5. Intégrer les retours d'utilisateurs sur les cas d'utilisation réels

## Calendrier suggéré

- **T1**: Optimisation des hyperparamètres et de l'implémentation
- **T2**: Extension à des contextes plus longs (>1M tokens) et tests approfondis
- **T3**: Adaptation multilingue et tests spécifiques aux langues
- **T4**: Intégration avec d'autres architectures de modèles et frameworks d'inférence

"""
    
    # Écrire rapport dans fichier
    with open(output_file, "w") as f:
        f.write(report)
    
    logger.info(f"Rapport de feuille de route généré: {output_file}")


def main():
    """Fonction principale pour exécuter toutes les expériences d'ablation"""
    parser = argparse.ArgumentParser(description="Expériences d'ablation pour NeoBERT-MOBA")
    parser.add_argument("--model_path", type=str, required=True, help="Chemin vers le checkpoint du modèle")
    parser.add_argument("--output_dir", type=str, default="./ablation_results", help="Répertoire de sortie")
    parser.add_argument("--context_length", type=int, default=32768, help="Longueur du contexte pour l'évaluation")
    parser.add_argument("--max_samples", type=int, default=5, help="Nombre maximum d'échantillons")
    parser.add_argument("--batch_size", type=int, default=1, help="Taille du batch")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device à utiliser")
    
    parser.add_argument("--run_granularity", action="store_true", help="Exécuter l'ablation sur la granularité des blocs")
    parser.add_argument("--run_hybrid", action="store_true", help="Exécuter l'ablation sur la stratégie hybride")
    parser.add_argument("--run_routing", action="store_true", help="Exécuter l'ablation sur les métriques de routage")
    parser.add_argument("--run_all", action="store_true", help="Exécuter toutes les ablations")
    
    args = parser.parse_args()
    
    # Créer répertoire de sortie principal
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Exécuter expériences sélectionnées
    if args.run_all or args.run_granularity:
        granularity_dir = os.path.join(args.output_dir, "block_granularity")
        run_block_granularity_ablation(
            base_model_path=args.model_path,
            output_dir=granularity_dir,
            context_length=args.context_length,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            device=args.device
        )
    
    if args.run_all or args.run_hybrid:
        hybrid_dir = os.path.join(args.output_dir, "hybrid_strategy")
        run_hybrid_strategy_ablation(
            base_model_path=args.model_path,
            output_dir=hybrid_dir,
            context_length=args.context_length,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            device=args.device
        )
    
    if args.run_all or args.run_routing:
        routing_dir = os.path.join(args.output_dir, "routing_metrics")
        run_routing_metrics_ablation(
            base_model_path=args.model_path,
            output_dir=routing_dir,
            context_length=args.context_length,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            device=args.device
        )
    
    # Générer feuille de route pour développement futur
    roadmap_file = os.path.join(args.output_dir, "development_roadmap.md")
    generate_development_roadmap(args.output_dir, roadmap_file)
    
    logger.info("Expériences d'ablation terminées. Résultats sauvegardés dans " + args.output_dir)
    logger.info("Feuille de route pour le développement futur générée: " + roadmap_file)


if __name__ == "__main__":
    main()
    
    
# Utilisation du script
# Pour exécuter toutes les expériences d'ablation:

# bash
# Copy Code
# python ablation_experiments.py --model_path ./checkpoints/neobert_moba --output_dir ./ablation_results --run_all
# Pour exécuter une expérience spécifique:

# bash
# Copy Code
# python ablation_experiments.py --model_path ./checkpoints/neobert_moba --output_dir ./ablation_results --run_granularity
# Fonctionnalités du code
# Étude de la granularité des blocs: Teste 5 configurations de granularité différentes (8 à 128 blocs) avec un taux de sparsité constant.
# Étude de la stratégie hybride: Compare différentes combinaisons de couches MOBA et d'attention complète.
# Étude des métriques de routage: Évalue 4 méthodes de calcul des scores d'affinité:
# Mean pooling (standard)
# Max pooling
# Combinaison min/max pooling
# Token représentatif
# Visualisations automatiques: Génère des graphiques pour chaque expérience permettant d'analyser les compromis entre efficacité et performance.
# Feuille de route générée automatiquement: Produit un rapport détaillé avec des recommandations pour la suite du développement basées sur les résultats d'ablation.