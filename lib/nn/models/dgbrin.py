import torch
from einops import rearrange
from torch import nn

from ..layers import BiGRIL
from ..layers.graph_learn_module import graph_constructor

# Model definition includes data accommodation, attribute accommodation, and the definition of imputation layers,
# where the imputation layers are used for filling missing values, forming the system.
class DGBRIN(nn.Module):
    def __init__(self,
                 n_nodes,
                 d_in,
                 d_hidden,
                 d_ff,
                 ff_dropout=0.0,
                 n_layers=1,
                 kernel_size=2,
                 decoder_order=1,
                 global_att=False,
                 d_u=0,
                 d_emb=0,
                 layer_norm=False,
                 merge='mlp',
                 impute_only_holes=True,
                 batch_size=128,
                 number_of_samples=0,
                 time_step=30,
                 fuse_dim=10):
        super(DGBRIN, self).__init__()
        self.batch_size = batch_size
        self.d_in = d_in  # Input dimension
        self.d_hidden = d_hidden  # Hidden layer dimension
        self.d_u = int(d_u) if d_u is not None else 0  
        self.d_emb = int(d_emb) if d_emb is not None else 0  
        self.fuse_dim = fuse_dim
        self.dynamic_adj = torch.zeros(n_nodes,n_nodes).float()
        
        self.impute_only_holes = impute_only_holes  # Custom parameter to decide whether to output full values + imputed values or just imputed values
        
        self.bigrill = BiGRIL(input_size=self.d_in,
                              ff_size=d_ff,
                              ff_dropout=ff_dropout,
                              hidden_size=self.d_hidden,
                              embedding_size=self.d_emb,
                              n_nodes=n_nodes,
                              n_layers=n_layers,
                              kernel_size=kernel_size,
                              decoder_order=decoder_order,
                              global_att=global_att,
                              u_size=self.d_u,
                              layer_norm=layer_norm,
                              merge=merge)

        self.graph_construct = graph_constructor(nodes=n_nodes, dim=self.fuse_dim, device='cuda:0', time_step=time_step, eta=1, in_dim=1, gamma=0.001, dropout=ff_dropout, m=0.9, batch_size=batch_size, is_add1=False)
        
        self.fusion_linear = nn.Linear(2 * n_nodes, n_nodes)  # For dynamic-static graph fusion

    def forward(self, x, mask=None, u=None, index=None, **kwargs):
        # Rearrange x and mask dimensions for processing
        x = rearrange(x, 'b s n c -> b c n s')
        mask = rearrange(mask, 'b s n c -> b c n s')
        n_batches, n_channels, n_nodes, n_steps = x.shape

        # Obtain static adjacency matrix using graph constructor
        dynamic_adj = self.graph_construct(x)
        self.dynamic_adj.data = dynamic_adj
        # Set diagonal elements to zero and normalize rows to sum to 1
        mask_ones = torch.eye(n_nodes, n_nodes).cuda().bool()
        self.dynamic_adj.masked_fill_(mask_ones, 0)
        self.dynamic_adj /= (self.dynamic_adj.sum(2, keepdims=True) + 1e-8)
        
        # Set values below a threshold to zero for normalization
        threshold_value = 1 / n_nodes
        self.dynamic_adj[self.dynamic_adj < threshold_value] = 0
        self.dynamic_adj /= (self.dynamic_adj.sum(2, keepdims=True) + 1e-8)

        # Compute imputation and prediction using BiGRIL
        imputation, prediction = self.bigrill(x, self.dynamic_adj, mask=mask, u=u, cached_support=self.training)
        if self.impute_only_holes and not self.training:
            imputation = torch.where(mask.byte(), x, imputation)

        # Transpose the last and third-to-last dimensions
        imputation = torch.transpose(imputation, -3, -1)
        prediction = torch.transpose(prediction, -3, -1)

        if self.training:
            return imputation, prediction, 0
        return imputation

    # Method to define additional parameters specific to the DGBRIN model
    @staticmethod
    def add_model_specific_args(namespace):
        namespace.d_hidden = 64
        namespace.d_ff = 64
        namespace.ff_dropout = 0.
        namespace.n_layers = 1
        namespace.kernel_size = 2
        namespace.decoder_order = 1
        namespace.d_u = 0
        namespace.d_emb = 8
        namespace.layer_norm = False
        namespace.global_att = False
        namespace.merge = 'mlp'
        namespace.impute_only_holes = True
        return namespace
