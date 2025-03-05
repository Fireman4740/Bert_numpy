import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict, Any, Union
import numpy as np


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm)"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: [batch_size, seq_len, dim]
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE)"""
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.max_seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_len: int):
        # x: [batch_size, n_heads, seq_len, head_dim]
        if seq_len > self.max_seq_len_cached or self.cos_cached is None or self.sin_cached is None:
            self._update_cache(seq_len)

        cos = self.cos_cached[:seq_len].to(x.device)
        sin = self.sin_cached[:seq_len].to(x.device)
        return (x * cos) + (self._rotate_half(x) * sin)

    def _update_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]

    def _rotate_half(self, x):
        x1, x2 = x[..., : self.dim // 2], x[..., self.dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)


class SwiGLU(nn.Module):
    """SwiGLU activation function"""
    def __init__(self, dim: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or int(2/3 * 4 * dim)  # 8/3 * dim as in NeoBERT paper

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        gate = F.silu(self.w1(x))
        value = self.w2(x)
        return self.w3(gate * value)


class MoBAAttention(nn.Module):
    """
    Mixture of Block Attention (MoBA) - implémentation selon l'article de recherche
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        block_size: int = 512,
        top_k: int = 3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.block_size = block_size
        self.top_k = top_k

        # Projections pour Q, K, V
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.rotary = RotaryEmbedding(self.head_dim)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def _split_heads(self, x):
        # x: [batch_size, seq_len, hidden_size]
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_shape)  # [batch_size, seq_len, num_heads, head_dim]
        return x.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]

    def _merge_heads(self, x):
        # x: [batch_size, num_heads, seq_len, head_dim]
        x = x.permute(0, 2, 1, 3)  # [batch_size, seq_len, num_heads, head_dim]
        new_shape = x.size()[:-2] + (self.hidden_size,)
        return x.reshape(*new_shape)  # [batch_size, seq_len, hidden_size]

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape

        # Projections Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Fractionner en têtes d'attention multiples
        q = self._split_heads(q)  # [batch_size, num_heads, seq_len, head_dim]
        k = self._split_heads(k)  # [batch_size, num_heads, seq_len, head_dim]
        v = self._split_heads(v)  # [batch_size, num_heads, seq_len, head_dim]

        # Appliquer RoPE
        q = self.rotary(q, seq_len)
        k = self.rotary(k, seq_len)

        # Calculer le nombre de blocs (arrondi vers le haut si nécessaire)
        num_blocks = (seq_len + self.block_size - 1) // self.block_size

        # Partition des K et V en blocs
        k_blocks = []
        v_blocks = []
        for i in range(num_blocks):
            start_idx = i * self.block_size
            end_idx = min(start_idx + self.block_size, seq_len)
            k_blocks.append(k[:, :, start_idx:end_idx, :])
            v_blocks.append(v[:, :, start_idx:end_idx, :])

        # Calculer le bloc moyen pour chaque bloc de K (pour le routage)
        k_mean_blocks = [k_block.mean(dim=2, keepdim=True) for k_block in k_blocks]

        # Préparer le output
        output = torch.zeros_like(q)

        # Traiter chaque position de query séparément
        for query_pos in range(seq_len):
            q_pos = q[:, :, query_pos:query_pos+1, :]  # [batch_size, num_heads, 1, head_dim]

            # Déterminer le bloc courant
            current_block_idx = query_pos // self.block_size

            # Calculer les scores d'affinité entre la query et chaque bloc
            affinity_scores = []
            for i, k_mean in enumerate(k_mean_blocks):
                # Calculer le score d'affinité (produit scalaire)
                score = torch.matmul(q_pos, k_mean.transpose(-1, -2)) * self.scale

                # Appliquer le masque causal (ne pas attendre aux blocs futurs)
                if i > current_block_idx:
                    score = torch.full_like(score, float('-inf'))

                affinity_scores.append(score)

            # Concaténer les scores
            affinity_scores = torch.cat(affinity_scores, dim=-1)  # [batch_size, num_heads, 1, num_blocks]

            # Sélectionner les top-k blocs
            topk_values, topk_indices = torch.topk(affinity_scores, self.top_k, dim=-1)

            # S'assurer que le bloc courant est toujours inclus
            if self.top_k > 1:
                has_current_block = (topk_indices == current_block_idx).any(dim=-1, keepdim=True)
                if not has_current_block.all():
                    # Remplacer le bloc avec le score le plus bas par le bloc courant
                    min_value_idx = torch.argmin(topk_values, dim=-1, keepdim=True)
                    for b in range(batch_size):
                        for h in range(self.num_heads):
                            idx = min_value_idx[b, h, 0, 0]
                            topk_indices[b, h, 0, idx] = current_block_idx

            # Concaténer les valeurs K et V des blocs sélectionnés
            selected_k = []
            selected_v = []
            for b in range(batch_size):
                for h in range(self.num_heads):
                    batch_head_indices = topk_indices[b, h, 0]
                    batch_head_k = torch.cat([k_blocks[idx.item()][b:b+1, h:h+1] for idx in batch_head_indices], dim=2)
                    batch_head_v = torch.cat([v_blocks[idx.item()][b:b+1, h:h+1] for idx in batch_head_indices], dim=2)
                    selected_k.append(batch_head_k)
                    selected_v.append(batch_head_v)

            # Reshape et concaténer
            selected_k = torch.cat(selected_k, dim=0).reshape(batch_size, self.num_heads, -1, self.head_dim)
            selected_v = torch.cat(selected_v, dim=0).reshape(batch_size, self.num_heads, -1, self.head_dim)

            # Calculer l'attention avec les K et V sélectionnés
            attn_scores = torch.matmul(q_pos, selected_k.transpose(-1, -2)) * self.scale

            # Optionnel: appliquer un masque d'attention global si fourni
            if attention_mask is not None:
                # Adapter le masque d'attention aux blocs sélectionnés
                pass

            # Softmax et multiplication avec V
            attn_probs = F.softmax(attn_scores, dim=-1)
            pos_output = torch.matmul(attn_probs, selected_v)

            # Ajouter au résultat
            output[:, :, query_pos:query_pos+1, :] = pos_output

        # Fusionner les têtes d'attention
        output = self._merge_heads(output)

        # Projection finale
        return self.o_proj(output)


class FullAttention(nn.Module):
    """Standard Self-Attention with RoPE"""
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.rotary = RotaryEmbedding(self.head_dim)

        self.scale = 1.0 / math.sqrt(self.head_dim)

    def _split_heads(self, x):
        # x: [batch_size, seq_len, hidden_size]
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_shape)  # [batch_size, seq_len, num_heads, head_dim]
        return x.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]

    def _merge_heads(self, x):
        # x: [batch_size, num_heads, seq_len, head_dim]
        x = x.permute(0, 2, 1, 3)  # [batch_size, seq_len, num_heads, head_dim]
        new_shape = x.size()[:-2] + (self.hidden_size,)
        return x.reshape(*new_shape)  # [batch_size, seq_len, hidden_size]

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape

        # Projections Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Split heads
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # Apply rotary embeddings
        q = self.rotary(q, seq_len)
        k = self.rotary(k, seq_len)

        # Calculate attention scores
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Apply causal mask
        causal_mask = torch.ones((seq_len, seq_len), device=q.device).triu_(diagonal=1) * -1e9
        attn_scores = attn_scores + causal_mask.unsqueeze(0).unsqueeze(0)

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        # Softmax and apply attention
        attn_probs = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_probs, v)

        # Merge heads
        context = self._merge_heads(context)

        # Output projection
        return self.o_proj(context)


class NeoBERTMoBALayer(nn.Module):
    """NeoBERT Layer with MoBA or Full Attention"""
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        use_moba: bool = True,
        moba_block_size: int = 512,
        moba_top_k: int = 3
    ):
        super().__init__()
        self.use_moba = use_moba

        self.norm1 = RMSNorm(hidden_size)

        if use_moba:
            self.attention = MoBAAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                block_size=moba_block_size,
                top_k=moba_top_k
            )
        else:
            self.attention = FullAttention(
                hidden_size=hidden_size,
                num_heads=num_heads
            )

        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLU(hidden_size)

    def forward(self, hidden_states, attention_mask=None):
        # Pre-LN for attention
        attn_input = self.norm1(hidden_states)
        attn_output = self.attention(attn_input, attention_mask)
        hidden_states = hidden_states + attn_output

        # Pre-LN for FFN
        mlp_input = self.norm2(hidden_states)
        mlp_output = self.mlp(mlp_input)
        hidden_states = hidden_states + mlp_output

        return hidden_states


class NeoBERTEmbeddings(nn.Module):
    """Embeddings for NeoBERT"""
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.norm = RMSNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids):
        embeddings = self.word_embeddings(input_ids)
        embeddings = self.norm(embeddings)
        return self.dropout(embeddings)


class NeoBERTMoBA(nn.Module):
    """NeoBERT model with MOBA attention"""
    def __init__(
        self,
        vocab_size: int = 30000,
        hidden_size: int = 768,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 12,
        max_position_embeddings: int = 32768,
        moba_block_size: int = 512,
        moba_top_k: int = 3,
        hybrid_layer_count: int = 3,
    ):
        super().__init__()
        self.config = {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "max_position_embeddings": max_position_embeddings,
            "moba_block_size": moba_block_size,
            "moba_top_k": moba_top_k,
            "hybrid_layer_count": hybrid_layer_count,
        }

        self.embeddings = NeoBERTEmbeddings(vocab_size, hidden_size)

        # Create layers with hybrid architecture
        # Last few layers use full attention, the rest use MoBA
        self.layers = nn.ModuleList()
        for i in range(num_hidden_layers):
            # Use MoBA for early layers, full attention for last hybrid_layer_count layers
            use_moba = i < (num_hidden_layers - hybrid_layer_count)
            self.layers.append(
                NeoBERTMoBALayer(
                    hidden_size=hidden_size,
                    num_heads=num_attention_heads,
                    use_moba=use_moba,
                    moba_block_size=moba_block_size,
                    moba_top_k=moba_top_k
                )
            )

        self.final_norm = RMSNorm(hidden_size)

        # MLM head
        self.mlm_dense = nn.Linear(hidden_size, hidden_size)
        self.mlm_norm = RMSNorm(hidden_size)
        self.mlm_head = nn.Linear(hidden_size, vocab_size)

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, return_hidden_states=False):
        batch_size, seq_len = input_ids.size()

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=input_ids.device)

        # Convert to extended attention mask (1.0 for tokens to attend, 0.0 for masked tokens)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Process embeddings
        hidden_states = self.embeddings(input_ids)

        # Store all hidden states if requested
        all_hidden_states = [] if return_hidden_states else None

        # Forward through all layers with gradient checkpointing
        for i, layer in enumerate(self.layers):
            if return_hidden_states:
                all_hidden_states.append(hidden_states)

            # Apply gradient checkpointing if in training mode
            if self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    extended_attention_mask
                )
            else:
                hidden_states = layer(hidden_states, extended_attention_mask)

        # Final layer norm
        hidden_states = self.final_norm(hidden_states)

        if return_hidden_states:
            all_hidden_states.append(hidden_states)

        # MLM head
        mlm_output = self.mlm_dense(hidden_states)
        mlm_output = F.gelu(mlm_output)
        mlm_output = self.mlm_norm(mlm_output)
        prediction_scores = self.mlm_head(mlm_output)

        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Calculate MLM loss - ignore padding tokens (-100)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # Flatten the tokens
            active_loss = labels.view(-1) != -100
            active_logits = prediction_scores.view(-1, self.config["vocab_size"])[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)

        outputs = {
            'last_hidden_state': hidden_states,
            'prediction_logits': prediction_scores,
        }

        if loss is not None:
            outputs['loss'] = loss

        if return_hidden_states:
            outputs['hidden_states'] = all_hidden_states

        return outputs

    def switch_attention_mode(self, layer_idx, use_moba):
        """
        Switches a specific layer between MoBA and Full Attention
        Useful for training strategies that transition between attention types
        """
        if layer_idx >= len(self.layers):
            raise ValueError(f"Layer index {layer_idx} is out of range (0-{len(self.layers)-1})")

        hidden_size = self.config["hidden_size"]
        num_heads = self.config["num_attention_heads"]
        moba_block_size = self.config["moba_block_size"]
        moba_top_k = self.config["moba_top_k"]

        # Create new attention module
        if use_moba:
            new_attention = MoBAAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                block_size=moba_block_size,
                top_k=moba_top_k
            )
        else:
            new_attention = FullAttention(
                hidden_size=hidden_size,
                num_heads=num_heads
            )

        # Transfer weights if dimensions match
        old_attention = self.layers[layer_idx].attention
        new_attention.q_proj.weight.data.copy_(old_attention.q_proj.weight.data)
        new_attention.k_proj.weight.data.copy_(old_attention.k_proj.weight.data)
        new_attention.v_proj.weight.data.copy_(old_attention.v_proj.weight.data)
        new_attention.o_proj.weight.data.copy_(old_attention.o_proj.weight.data)

        # Replace attention module
        self.layers[layer_idx].attention = new_attention
        self.layers[layer_idx].use_moba = use_moba

    def switch_to_hybrid_mode(self, hybrid_layer_count):
        """
        Switches the model to hybrid mode with the last 'hybrid_layer_count' layers
        using Full Attention and the rest using MoBA
        """
        for i in range(len(self.layers)):
            use_moba = i < (len(self.layers) - hybrid_layer_count)
            if use_moba != self.layers[i].use_moba:
                self.switch_attention_mode(i, use_moba)

        # Update config
        self.config["hybrid_layer_count"] = hybrid_layer_count

    def switch_to_full_attention(self):
        """Switches all layers to Full Attention"""
        for i in range(len(self.layers)):
            if self.layers[i].use_moba:
                self.switch_attention_mode(i, False)

        # Update config
        self.config["hybrid_layer_count"] = len(self.layers)

    def switch_to_full_moba(self):
        """Switches all layers to MoBA Attention"""
        for i in range(len(self.layers)):
            if not self.layers[i].use_moba:
                self.switch_attention_mode(i, True)

        # Update config
        self.config["hybrid_layer_count"] = 0


def create_neobert_moba_model(config):
    """
    Helper function to create a NeoBERT-MOBA model from a configuration
    """
    model = NeoBERTMoBA(
        vocab_size=config.get("vocab_size", 30000),
        hidden_size=config.get("hidden_size", 768),
        num_hidden_layers=config.get("num_hidden_layers", 28),
        num_attention_heads=config.get("num_attention_heads", 12),
        max_position_embeddings=config.get("max_position_embeddings", 32768),
        moba_block_size=config.get("moba_block_size", 512),
        moba_top_k=config.get("moba_top_k", 3),
        hybrid_layer_count=config.get("hybrid_layer_count", 3),
    )
    return model


def create_neobert_moba_t4_model():
    """Crée une version plus petite du modèle adaptée aux GPUs T4"""
    model_config = {
        # Configuration NeoBERT réduite pour T4
        "vocab_size": 30000,
        "hidden_size": 512,  # Réduit de 768 à 512
        "num_hidden_layers": 16,  # Réduit de 28 à 16
        "num_attention_heads": 8,  # Réduit de 12 à 8
        "max_position_embeddings": 16384,  # Réduit de 32768 à 16384

        # Configuration MOBA
        "moba_block_size": 512,
        "moba_top_k": 3,
        "hybrid_layer_count": 2,  # Réduit de 3 à 2
    }
    return create_neobert_moba_model(model_config)