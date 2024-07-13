# 引入所需的库和模块
import torch
import torch.nn as nn
from einops import rearrange

from .spatial_conv import SpatialConvOrderK
from .gcrnn import GCGRUCell 
from .spatial_attention import SpatialAttention
from ..utils.ops import reverse_tensor


class SpatialDecoder(nn.Module):
    """Temporal decoder for integrating neighbor information."""
    def __init__(self, d_in, d_model, d_out, support_len, order=1, attention_block=False, nheads=2, dropout=0.):
        super(SpatialDecoder, self).__init__()
        self.order = order  # Order of graph convolution
        self.lin_in = nn.Conv1d(d_in, d_model, kernel_size=1)  # Fully connected layer for extracting single-dimensional temporal correlations
        self.graph_conv = SpatialConvOrderK(c_in=d_model, c_out=d_model, support_len=support_len * order, order=1, include_self=False)
        if attention_block:
            self.spatial_att = SpatialAttention(d_in=d_model, d_model=d_model, nheads=nheads, dropout=dropout)
            self.lin_out = nn.Conv1d(3 * d_model, d_model, kernel_size=1)
        else:
            self.register_parameter('spatial_att', None)
            self.lin_out = nn.Conv1d(2 * d_model, d_model, kernel_size=1)
        self.read_out = nn.Conv1d(2 * d_model, d_out, kernel_size=1)
        self.activation = nn.PReLU()
        self.adj = None

    def forward(self, x, m, h, u, adj, cached_support=False):
        x_in = [x, m, h] if u is None else [x, m, u, h]
        x_in = torch.cat(x_in, 1)
        if self.order > 1:
            if cached_support and (self.adj is not None):
                adj = self.adj
            else:
                adj = SpatialConvOrderK.compute_support_orderK(adj, self.order, include_self=False, device=x_in.device)
                self.adj = adj if cached_support else None

        x_in = self.lin_in(x_in)
        out = self.graph_conv(x_in, adj)
        if self.spatial_att is not None:
            x_in = rearrange(x_in, 'b f n -> b 1 n f')
            out_att = self.spatial_att(x_in, torch.eye(x_in.size(2), dtype=torch.bool, device=x_in.device))
            out_att = rearrange(out_att, 'b s n f -> b f (n s)')
            out = torch.cat([out, out_att], 1)
        out = torch.cat([out, h], 1)
        out = self.activation(self.lin_out(out))
        out = torch.cat([out, h], 1)
        return self.read_out(out), out

class TRAE_layer(nn.Module):
    """A layer for affine transformation with threshold adjustment."""
    def __init__(self, dim):
        super(TRAE_layer, self).__init__()
        self.weights = nn.Parameter(torch.randn(dim, dim))
        self.threshold = nn.Parameter(torch.randn(dim, 1))
        self.weights.data.fill_diagonal_(0)

    def forward(self, x):
        x = torch.matmul(x, self.weights)
        x = x - self.threshold.T
        return x


class GRIL(nn.Module):
    """
    Modified version of GRU for forward propagation with additional functionalities.
    """
    def __init__(self, input_size, hidden_size, u_size=None, n_layers=1, dropout=0.,
                 kernel_size=2, decoder_order=1, global_att=False, support_len=2,
                 n_nodes=None, layer_norm=False, use_linear_input=False, use_gat=True):
        super(GRIL, self).__init__()
        self.use_gat = use_gat
        self.use_linear_input = use_linear_input
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.u_size = int(u_size) if u_size is not None else 0  # Purpose of 'u' is unclear
        self.n_layers = int(n_layers)

        # Calculation of the input size for the RNN layers.
        rnn_input_size = 2 * self.input_size + self.u_size  # input + mask + exogenous if available
        self.trae = TRAE_layer(8)

        # Definition of cells and normalization layers for RNN based on 'use_linear_input'
        if not self.use_linear_input:
            self.cells = nn.ModuleList()
            self.norms = nn.ModuleList()
            for i in range(self.n_layers):
                self.cells.append(GCGRUCell(d_in=rnn_input_size if i == 0 else self.hidden_size,
                                            num_units=self.hidden_size, support_len=support_len, order=kernel_size))
                self.norms.append(nn.GroupNorm(num_groups=1, num_channels=self.hidden_size) if layer_norm else nn.Identity())
        else:
            self.cells_with_linear = nn.ModuleList()
            self.norms_with_linear = nn.ModuleList()
            for i in range(self.n_layers):
                self.cells_with_linear.append(GCGRUCell(d_in=int(rnn_input_size*2) if i == 0 else self.hidden_size,
                                                        num_units=self.hidden_size, support_len=support_len, order=kernel_size))
                self.norms_with_linear.append(nn.GroupNorm(num_groups=1, num_channels=self.hidden_size) if layer_norm else nn.Identity())

        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        self.first_stage = nn.Conv1d(in_channels=self.hidden_size, out_channels=self.input_size, kernel_size=1)
        self.spatial_decoder = SpatialDecoder(d_in=rnn_input_size + self.hidden_size,
                                              d_model=self.hidden_size,
                                              d_out=self.input_size,
                                              support_len=2,
                                              order=decoder_order,
                                              attention_block=global_att)

        # Initialization of hidden states if n_nodes are provided.
        if n_nodes is not None:
            self.h0 = self.init_hidden_states(n_nodes)
        else:
            self.register_parameter('h0', None)

    def init_hidden_states(self, n_nodes):
        """
        Initializes hidden states, trainable, with output dimension equal to the attribute dimension n_nodes.
        """
        h0 = []
        for l in range(self.n_layers):
            std = 1. / torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float))
            vals = torch.distributions.Normal(0, std).sample((self.hidden_size, n_nodes))
            h0.append(nn.Parameter(vals))
        return nn.ParameterList(h0)

    def get_h0(self, x):
        """
        Initializes hidden states to ensure size compatibility.
        """
        if self.h0 is not None:
            return [h.expand(x.shape[0], -1, -1) for h in self.h0]
        return [torch.zeros(size=(x.shape[0], self.hidden_size, x.shape[2])).to(x.device)] * self.n_layers

    def update_state(self, x, h, adj):
        """
        Aggregates information and updates hidden neuron states using the key function 'cell'.
        """
        rnn_in = x
        for layer, (cell, norm) in enumerate(zip(self.cells, self.norms)):
            rnn_in = h[layer] = norm(cell(rnn_in, h[layer], adj))
            if self.dropout is not None and layer < (self.n_layers - 1):
                rnn_in = self.dropout(rnn_in)
        return h

    def update_state_with_linear(self, x, h, adj):
        """
        Updates hidden states using linear input considerations.
        """
        rnn_in = x
        for layer, (cell_with_linear, norm_with_linear) in enumerate(zip(self.cells_with_linear, self.norms_with_linear)):
            rnn_in = h[layer] = norm_with_linear(cell_with_linear(rnn_in, h[layer], adj))
            if self.dropout is not None and layer < (self.n_layers - 1):
                rnn_in = self.dropout(rnn_in)
        return h

    def forward(self, x, adj, mask=None, u=None, h=None, cached_support=False):
        """
        Forward propagation, with 'adj' being the normalized adjacency matrix.
        """
        use_linear_input = self.use_linear_input
        *_, steps = x.size()

        if mask is None:
            mask = torch.ones_like(x, dtype=torch.uint8)

        if h is None:
            h = self.get_h0(x)
        elif not isinstance(h, list):
            h = [*h]

        predictions, imputations, states, representations = [], [], [], []
        for step in range(steps):
            x_s = x[..., step]
            x_input = x_s.clone()
            m_s = mask[..., step]
            h_s = h[-1]
            u_s = u[..., step] if u is not None else None
            xs_hat_1 = self.first_stage(h_s)
            m_s_byte = m_s.byte()
            x_s = torch.where(m_s_byte, x_s, xs_hat_1)
            xs_hat_2, repr_s = self.spatial_decoder(x=x_s, m=m_s, h=h_s, u=u_s, adj=adj, cached_support=cached_support)
            x_s = torch.where(m_s_byte, x_s, xs_hat_2)

            if use_linear_input:
                m_from0_to1 = (m_s > 0) & (m_s < 1)
                m_reliable_grin = m_s.clone()
                m_reliable_grin[m_from0_to1] = 1 - m_reliable_grin[m_from0_to1]
                inputs = [x_input, m_s, x_s, m_reliable_grin]
                if u_s is not None:
                    inputs.append(u_s)
                inputs = torch.cat(inputs, dim=1)
                h = self.update_state_with_linear(inputs, h, adj)
            else:
                inputs = [x_s, m_s]
                if u_s is not None:
                    inputs.append(u_s)
                inputs = torch.cat(inputs, dim=1)
                h = self.update_state(inputs, h, adj)

            if u_s is not None:
                inputs.append(u_s)
            imputations.append(xs_hat_2)
            predictions.append(xs_hat_1)
            states.append(torch.stack(h, dim=0))
            representations.append(repr_s)

        imputations = torch.stack(imputations, dim=-1)
        predictions = torch.stack(predictions, dim=-1)
        states = torch.stack(states, dim=-1)
        representations = torch.stack(representations, dim=-1)

        return imputations, predictions, representations, states


class BiGRIL(nn.Module):
    """
    Bidirectional GRIL integrates forward and backward GRIL layers for enhanced learning by combining outputs.
    """
    def __init__(self, input_size, hidden_size, ff_size, ff_dropout, n_layers=1, dropout=0.,
                 n_nodes=None, support_len=2, kernel_size=2, decoder_order=1, global_att=False,
                 u_size=0, embedding_size=0, layer_norm=False, merge='mlp'):
        super(BiGRIL, self).__init__()
        # Initialize forward and backward GRIL layers
        self.fwd_rnn = GRIL(input_size=input_size, hidden_size=hidden_size, n_layers=n_layers,
                            dropout=dropout, n_nodes=n_nodes, support_len=support_len,
                            kernel_size=kernel_size, decoder_order=decoder_order,
                            global_att=global_att, u_size=u_size, layer_norm=layer_norm)
        self.bwd_rnn = GRIL(input_size=input_size, hidden_size=hidden_size, n_layers=n_layers,
                            dropout=dropout, n_nodes=n_nodes, support_len=support_len,
                            kernel_size=kernel_size, decoder_order=decoder_order,
                            global_att=global_att, u_size=u_size, layer_norm=layer_norm)

        # Initialize embedding layer if embedding size is specified
        if embedding_size > 0:
            self.emb = nn.Parameter(torch.empty(embedding_size, n_nodes))
            nn.init.kaiming_normal_(self.emb, nonlinearity='relu')
        else:
            self.register_parameter('emb', None)

        # Configure merging strategy
        if merge == 'mlp':
            self._impute_from_states = True
            self.out = nn.Sequential(
                nn.Conv2d(in_channels=4 * hidden_size + input_size + embedding_size,
                          out_channels=ff_size, kernel_size=1),
                nn.ReLU(),
                nn.Dropout(ff_dropout),
                nn.Conv2d(in_channels=ff_size, out_channels=input_size, kernel_size=1)
            )
        elif merge in ['mean', 'sum', 'min', 'max']:
            self._impute_from_states = False
            self.out = getattr(torch, merge)
        else:
            raise ValueError(f"Merge option {merge} not allowed.")

        self.supp = None

    def forward(self, x, adj, mask=None, u=None, cached_support=False):
        """
        Forward pass computes the bi-directional GRIL outputs and merges them.
        """
        if cached_support and self.supp is not None:
            supp = self.supp
        else:
            supp = SpatialConvOrderK.compute_support(adj, x.device)
            self.supp = supp if cached_support else None

        # Process with forward GRIL
        fwd_out, fwd_pred, fwd_repr, _ = self.fwd_rnn(x, supp, mask=mask, u=u, cached_support=cached_support)
        
        # Process with backward GRIL, reversing input tensors
        rev_x, rev_mask, rev_u = [reverse_tensor(tens) for tens in (x, mask, u)]
        bwd_out, bwd_pred, bwd_repr, _ = self.bwd_rnn(rev_x, supp, mask=rev_mask, u=rev_u, cached_support=cached_support)
        bwd_out, bwd_pred, bwd_repr = [reverse_tensor(res) for res in (bwd_out, bwd_pred, bwd_repr)]

        # Combine forward and backward representations
        if self._impute_from_states:
            inputs = [fwd_repr, bwd_repr, mask]
            if self.emb is not None:
                b, *_, s = fwd_repr.shape
                inputs.append(self.emb.view(1, *self.emb.shape, 1).expand(b, -1, -1, s))
            imputation = torch.cat(inputs, dim=1)
            imputation = self.out(imputation)
        else:
            imputation = torch.stack([fwd_out, bwd_out], dim=1)
            imputation = self.out(imputation, dim=1)

        predictions = torch.stack([fwd_out, bwd_out, fwd_pred, bwd_pred], dim=0)

        return imputation, predictions

