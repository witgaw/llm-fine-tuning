import torch.nn.functional as F
from modules.attention import CausalSelfAttention
from torch import nn


class GPT2Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Multi-head attention.
        self.self_attention = CausalSelfAttention(config)
        # Add-norm for multi-head attention.
        self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        # Feed forward.
        self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.interm_af = F.gelu
        # Add-norm for feed forward.
        self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

    def add(self, input, output, dense_layer, dropout):
        """
        Helper method for the forward function.
        - This function is applied after the multi-head attention layer as well as after the feed forward layer.
        - GPT-2 layer applies dropout to the transformed output of each sub-layer,
          before it is added to the sub-layer input. WE DO NOT APPLY THE LAYER NORM
          IN THIS FUNCTION.
        """
        output = dense_layer(output)
        output = dropout(output)
        combined = input + output

        return combined

    def forward(self, hidden_states, attention_mask):
        """
        Forward pass through the GPT-2 layer:
        - A multi-head attention layer (CausalSelfAttention) that computes self-attention based on masked inputs.
        - Layer normalization applied *before* the attention layer and feed-forward layer.
        - Apply dropout, residual connection, and layer normalization according to the plot in the assignment. (Use self.add)
        - A feed-forward layer that applies transformations to further refine the hidden states.
        """
        norm_pre_att = self.attention_layer_norm(hidden_states)
        att = self.self_attention(norm_pre_att, attention_mask)

        att_with_skip = self.add(hidden_states, att, self.attention_dense, self.attention_dropout)

        norm_pre_mlp = self.out_layer_norm(att_with_skip)

        mlp_pre_af = self.interm_dense(norm_pre_mlp)
        mlp = self.interm_af(mlp_pre_af)

        return self.add(att_with_skip, mlp, self.out_dense, self.out_dropout)
