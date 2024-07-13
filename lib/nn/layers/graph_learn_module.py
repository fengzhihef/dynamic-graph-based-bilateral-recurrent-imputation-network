import torch
from torch import nn
from .layer_module import *

class graph_constructor(nn.Module):
    def __init__(self, nodes, dim, device, time_step, cout=16, heads=4, head_dim=8, eta=1, gamma=0.0001, dropout=0.5, m=0.9, batch_size=64, in_dim=2, is_add1=True):
        super(graph_constructor, self).__init__()

        self.attn_static = nn.LayerNorm(nodes)  # Normalization layer for static attention
        self.time_step = time_step + 1 if is_add1 else time_step

        # Convolutional layer to merge time steps into a single node feature
        self.trans_Merge_line = nn.Conv2d(in_dim, dim, kernel_size=(1, self.time_step), bias=True)

        self.time_norm = nn.LayerNorm(dim)

    def spearman_rank_coefficient_matrix(self, matrix):
        """
        Computes the Spearman rank correlation coefficient matrix for input features.
        """
        rank_matrix = matrix.argsort(dim=-1).argsort(dim=-1).float()
        matrix_centered = rank_matrix - rank_matrix.mean(dim=-1, keepdim=True)
        norm = matrix_centered.pow(2).sum(dim=-1, keepdim=True).sqrt()
        correlation_matrix = torch.mm(matrix_centered, matrix_centered.transpose(-1, -2))
        return torch.abs(correlation_matrix / (norm * norm.transpose(-1, -2) + 1e-8))
    
    def static_graph_spearman(self, input_embed):
        """
        Constructs a static graph using Spearman rank correlation based on node embeddings.
        """
        B, N, D = input_embed.shape
        result_static = torch.zeros(B, N, N).to(input_embed.device)
        for b in range(B):
            result_static[b] = self.spearman_rank_coefficient_matrix(input_embed[b])
        result_static = F.relu(self.attn_static(result_static))
        return result_static

    def forward(self, input):
        """
        Forward pass that processes the input through various transformations to produce an adjacency matrix.
        """
        input_transformed = self.trans_Merge_line(input).squeeze(-1).transpose(1, 2)
        input_normalized = self.time_norm(input_transformed)
        adj = self.static_graph_spearman(input_normalized)
        return adj



