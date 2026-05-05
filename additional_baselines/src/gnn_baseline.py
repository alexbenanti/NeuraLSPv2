import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add  # Requires pytorch_scatter


class MLP(nn.Module):
    """Simple Multi-Layer Perceptron helper."""
    def __init__(self, in_dim, hidden_dim, out_dim, layers=2):
        super().__init__()
        mod_list = []
        curr_dim = in_dim
        for _ in range(layers - 1):
            mod_list.append(nn.Linear(curr_dim, hidden_dim))
            mod_list.append(nn.ReLU())
            curr_dim = hidden_dim
        mod_list.append(nn.Linear(curr_dim, out_dim))
        self.net = nn.Sequential(*mod_list)

    def forward(self, x):
        return self.net(x)

class GraphNetworkBlock(nn.Module):
    """
    A Message Passing Layer as described in Luz et al. (2020).
    Updates node and edge features.
    """
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        # Functions to compute messages and updates
        self.edge_update_mlp = MLP(edge_dim + 2 * node_dim, hidden_dim, edge_dim)
        self.node_update_mlp = MLP(node_dim + edge_dim, hidden_dim, node_dim)

    def forward(self, x, edge_index, edge_attr):
        """
        x: (num_nodes, node_dim)
        edge_index: (2, num_edges) [src, dst]
        edge_attr: (num_edges, edge_dim)
        """
        row, col = edge_index
        
        # 1. Edge Update Step
        edge_in = torch.cat([edge_attr, x[row], x[col]], dim=1)
        edge_attr_new = self.edge_update_mlp(edge_in) # (num_edges, edge_dim)
        
        # 2. Node Update Step
        # Aggregate messages from edges to destination nodes
        # Message = Updated Edge Feature
        messages = scatter_add(edge_attr_new, col, dim=0, dim_size=x.size(0))
        
        # Concatenate: [Node_Feat | Aggregated_Messages]
        node_in = torch.cat([x, messages], dim=1)
        x_new = self.node_update_mlp(node_in) # (num_nodes, node_dim)
        
        return x_new, edge_attr_new

class AMG_GNN(nn.Module):
    def __init__(self, input_node_dim, output_dim, hidden_dim=64, num_layers=3):
        super().__init__()
        self.output_dim = output_dim # Rank r
        
        # 1. Encoder
        self.node_encoder = MLP(input_node_dim, hidden_dim, hidden_dim)
        
        self.edge_encoder = MLP(1, hidden_dim, hidden_dim)
        
        # 2. Processor (Message Passing Layers)
        self.layers = nn.ModuleList([
            GraphNetworkBlock(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # 3. Decoder (Node-wise prediction of vectors)
        # Maps latent node features -> r-dimensional vector
        self.decoder = MLP(hidden_dim, hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_attr):
        """
        x: (B * N, input_dim) - Batching handled by concatenating graphs (PyG style)
        edge_index: (2, B * E)
        edge_attr: (B * E, 1)
        """
        # Encode
        x = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)
        
        # Process
        
        for layer in self.layers:
            x, e = layer(x, edge_index, e)
            x = F.relu(x)
            e = F.relu(e)
            
        # Decode to get Vectors Y
        y = self.decoder(x) # (Total_Nodes, r)
        
        
        
        return y