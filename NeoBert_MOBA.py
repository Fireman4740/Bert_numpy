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
    """Mixture of Block Attention (MoBA)"""
    def __init__(
        self, 
        hidden_size: int, 
        num_heads: int, 
        block_size: int = 512, 
        top_k: int = 3,
        rotary_dim: int = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.block_size = block_size
        self.top_k = top_k
        
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
    
    def forward(self, hidden_states, attention_mask=None, past_key_value=None):
        batch_size, seq_len = hidden_states.shape[:2]
        
        # Linear projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Split into heads
        q = self._split_heads(q)  # [batch_size, num_heads, seq_len, head_dim]
        k = self._split_heads(k)  # [batch_size, num_heads, seq_len, head_dim]
        v = self._split_heads(v)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Apply RoPE to q and k
        q = self.rotary(q, seq_len)
        k = self.rotary(k, seq_len)
        
        # Split sequence into blocks
        num_blocks = math.ceil(seq_len / self.block_size)
        
        # Compute mean representation for each block
        block_means = []
        for i in range(num_blocks):
            start_idx = i * self.block_size
            end_idx = min(start_idx + self.block_size, seq_len)
            if end_idx > start_idx:  # Ensure non-empty block
                # Mean pooling of keys in this block for all heads
                # Shape: [batch_size, num_heads, 1, head_dim]
                block_mean = k[:, :, start_idx:end_idx].mean(dim=2, keepdim=True)
                block_means.append(block_mean)
        
        # Stack block means
        # Shape: [batch_size, num_heads, num_blocks, head_dim]
        block_means = torch.cat(block_means, dim=2)
        
        # Initialize output tensor
        attn_output = torch.zeros_like(q)
        
        # Process each query token
        for i in range(seq_len):
            # Get the current query
            # Shape: [batch_size, num_heads, 1, head_dim]
            query = q[:, :, i:i+1]
            
            # Compute relevance scores between query and block means
            # Shape: [batch_size, num_heads, 1, num_blocks]
            block_scores = torch.matmul(query, block_means.transpose(-1, -2)) * self.scale
            
            # Apply causal mask - query can only attend to blocks up to its position
            current_block = i // self.block_size
            causal_mask = torch.arange(num_blocks, device=query.device) > current_block
            block_scores = block_scores.masked_fill(
                causal_mask.view(1, 1, 1, -1), float('-inf')
            )
            
            # Always include the current block
            # Set a large positive value to ensure selection
            block_scores[:, :, :, current_block] = torch.finfo(block_scores.dtype).max
            
            # Get top-k blocks (always including current block)
            # Shape: [batch_size, num_heads, 1, top_k]
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
                            k_block = k[b, h:h+1, start_idx:i+1]  # Up to current position
                            v_block = v[b, h:h+1, start_idx:i+1]
                        else:
                            k_block = k[b, h:h+1, start_idx:end_idx]
                            v_block = v[b, h:h+1, start_idx:end_idx]
                        
                        if k_block.size(2) > 0:  # Ensure non-empty block
                            k_for_query.append(k_block)
                            v_for_query.append(v_block)
                    
                    if k_for_query:
                        # Concatenate selected keys and values
                        k_concat = torch.cat(k_for_query, dim=2)  # [1, seq_len_selected, head_dim]
                        v_concat = torch.cat(v_for_query, dim=2)  # [1, seq_len_selected, head_dim]
                        
                        # Compute attention scores
                        attn_scores = torch.matmul(query[b:b+1, h:h+1], k_concat.transpose(-1, -2)) * self.scale
                        attn_probs = F.softmax(attn_scores, dim=-1)
                        
                        # Apply attention
                        attn_output[b:b+1, h:h+1, i:i+1] = torch.matmul(attn_probs, v_concat)
        
        # Merge heads and project back
        attn_output = self._merge_heads(attn_output)
        attn_output = self.o_proj(attn_output)
        
        return attn_output


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
    
    def forward(self, hidden_states, attention_mask=None, past_key_value=None):
        batch_size, seq_len = hidden_states.shape[:2]
        
        # Linear projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Split into heads
        q = self._split_heads(q)  # [batch_size, num_heads, seq_len, head_dim]
        k = self._split_heads(k)  # [batch_size, num_heads, seq_len, head_dim]
        v = self._split_heads(v)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Apply RoPE to q and k
        q = self.rotary(q, seq_len)
        k = self.rotary(k, seq_len)
        
        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=q.device),
            diagonal=1
        )
        attention_scores = attention_scores.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0), float('-inf')
        )
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Calculate attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Apply attention
        context = torch.matmul(attention_probs, v)
        
        # Merge heads and project back
        context = self._merge_heads(context)
        attn_output = self.o_proj(context)
        
        return attn_output


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
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, return_hidden_states=False):
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
        
        outputs = {
            'last_hidden_state': hidden_states,
            'prediction_logits': prediction_scores,
        }
        
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


# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration du modèle
    model_config = {
        # Configuration NeoBERT
        "vocab_size": 30000,
        "hidden_size": 768,
        "num_hidden_layers": 28,
        "num_attention_heads": 12,
        "max_position_embeddings": 32768,
        
        # Configuration MOBA
        "moba_block_size": 512,  # Taille des blocs
        "moba_top_k": 3,         # Nombre de blocs à sélectionner
        "hybrid_layer_count": 3, # Dernières couches en attention complète
    }
    
    # Création du modèle
    model = create_neobert_moba_model(model_config)
    
    # Test avec un input factice
    batch_size = 2
    seq_len = 1024
    input_ids = torch.randint(0, 30000, (batch_size, seq_len))
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids)
    
    print(f"Last hidden state shape: {outputs['last_hidden_state'].shape}")
    print(f"Prediction logits shape: {outputs['prediction_logits'].shape}")
    
    # Test de changement de mode d'attention
    print(f"Avant: {sum(int(layer.use_moba) for layer in model.layers)} couches MOBA")
    model.switch_to_hybrid_mode(5)
    print(f"Après: {sum(int(layer.use_moba) for layer in model.layers)} couches MOBA")
    
    # Test avec le nouveau mode d'attention
    with torch.no_grad():
        outputs = model(input_ids)
    
    print(f"Last hidden state shape: {outputs['last_hidden_state'].shape}")

# Configuration du modèle pour T4 (à ajouter à la fin du fichier)
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