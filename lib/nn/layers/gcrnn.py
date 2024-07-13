import torch
import torch.nn as nn
from .spatial_conv import SpatialConvOrderK

class GCGRUCell(nn.Module):
    """
    Graph Convolution Gated Recurrent Unit (GCGRU) Cell, an adaptation of GRU for use with graph-structured data.
    """
    def __init__(self, d_in, num_units, support_len, order, activation='tanh'):
        super(GCGRUCell, self).__init__()
        self.activation_fn = getattr(torch, activation)
        self.forget_gate = SpatialConvOrderK(c_in=d_in + num_units, c_out=num_units, support_len=support_len, order=order)
        self.update_gate = SpatialConvOrderK(c_in=d_in + num_units, c_out=num_units, support_len=support_len, order=order)
        self.c_gate = SpatialConvOrderK(c_in=d_in + num_units, c_out=num_units, support_len=support_len, order=order)

    def forward(self, x, h, adj):
        """
        Performs a single forward pass through the GCGRU cell.
        
        Args:
            x (Tensor): Input features with shape (batch_size, input_dim, num_nodes).
            h (Tensor): Hidden state with shape (batch_size, num_units, num_nodes).
            adj (Tensor): Adjacency matrix with shape (num_nodes, num_nodes).
        
        Returns:
            Tensor: Updated hidden state.
        """
        x_gates = torch.cat([x, h], dim=1)
        r = torch.sigmoid(self.forget_gate(x_gates, adj))
        u = torch.sigmoid(self.update_gate(x_gates, adj))
        x_c = torch.cat([x, r * h], dim=1)
        c = self.c_gate(x_c, adj)
        c = self.activation_fn(c)
        return u * h + (1.0 - u) * c

class GCRNN(nn.Module):
    """
    Graph Convolutional Recurrent Neural Network (GCRNN) using the GCGRUCell for graph-based sequence processing.
    """
    def __init__(self, d_in, d_model, d_out, n_layers, support_len, kernel_size=2):
        super(GCRNN, self).__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.d_out = d_out
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.support_len = support_len
        self.rnn_cells = nn.ModuleList([
            GCGRUCell(d_in=self.d_in if i == 0 else self.d_model, num_units=self.d_model, 
                      support_len=self.support_len, order=self.kernel_size) for i in range(n_layers)
        ])
        self.output_layer = nn.Conv2d(self.d_model, self.d_out, kernel_size=1)

    def init_hidden_states(self, x):
        """
        Initializes hidden states for each layer to zero.

        Args:
            x (Tensor): Input tensor to infer batch size and device from.
        
        Returns:
            list[Tensor]: List of zero-initialized hidden states for each layer.
        """
        return [torch.zeros(size=(x.shape[0], self.d_model, x.shape[2]), device=x.device) for _ in range(self.n_layers)]

    def single_pass(self, x, h, adj):
        """
        Processes input through all RNN layers for a single time step.
        
        Args:
            x (Tensor): Input features for a single time step.
            h (list[Tensor]): List of hidden states for each layer.
            adj (Tensor): Adjacency matrix.
        
        Returns:
            tuple: Output of the last layer and updated hidden states.
        """
        out = x
        for l, layer in enumerate(self.rnn_cells):
            out = h[l] = layer(out, h[l], adj)
        return out, h

    def forward(self, x, adj, h=None):
        """
        Forward pass through the GCRNN.

        Args:
            x (Tensor): Input tensor with shape (batch, features, nodes, steps).
            adj (Tensor): Adjacency matrix.
            h (list[Tensor], optional): Initial hidden states for each layer.
        
        Returns:
            Tensor: The output of the network.
        """
        _, _, _, steps = x.size()
        if h is None:
            h = self.init_hidden_states(x)
        for step in range(steps):
            x_step = x[..., step]
            out, h = self.single_pass(x_step, h, adj)

        return self.output_layer(out[..., None])
