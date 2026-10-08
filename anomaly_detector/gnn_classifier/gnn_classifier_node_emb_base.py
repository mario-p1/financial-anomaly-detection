import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.typing import Metadata


class GNNClassifierNodeEmbeddingsBase(nn.Module):
    """Base class for GNN classifiers that use node embeddings for non-company nodes."""

    def __init__(
        self,
        metadata: Metadata,
        in_channels: int,
        hidden_channels: int,
    ):
        super().__init__()

        self.gnn_company_encoder = nn.Linear(in_channels, hidden_channels)

        self.type_embeddings = nn.ParameterDict(
            {
                node_type: nn.Parameter(torch.randn(hidden_channels))
                for node_type in metadata[0]
                if node_type != "L"
            },
        )

        # self.classification_head = nn.Linear(hidden_channels + graph["L"].x.size(-1), 1)
        self.classification_head = nn.Linear(hidden_channels, 1)

    def build_x_dict(self, graph: HeteroData) -> dict[str, torch.Tensor]:
        x_dict = {"L": self.gnn_company_encoder(graph["L"].x)}

        for node_type, embedding in self.type_embeddings.items():
            x_dict[node_type] = embedding.expand(graph[node_type].num_nodes, -1)

        return x_dict

    def forward(self, graph: HeteroData) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement the forward method.")
