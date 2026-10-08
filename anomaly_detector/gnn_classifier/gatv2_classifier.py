import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv, to_hetero
from torch_geometric.typing import Metadata

from anomaly_detector.gnn_classifier.gnn_classifier_node_emb_base import (
    GNNClassifierNodeEmbeddingsBase,
)


class GATv2Model(nn.Module):
    def __init__(
        self, in_channels: int, hidden_channels: int, dropout: float, heads: int
    ):
        super().__init__()
        assert hidden_channels % heads == 0, (
            "hidden_channels must be divisible by heads"
        )
        self.conv1 = GATv2Conv(
            in_channels, hidden_channels // heads, heads=heads, add_self_loops=False
        )
        self.conv2 = GATv2Conv(
            hidden_channels, hidden_channels // heads, heads=heads, add_self_loops=False
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index).relu()
        return x


class GatV2Classifier(GNNClassifierNodeEmbeddingsBase):
    def __init__(
        self,
        metadata: Metadata,
        in_channels: int,
        hidden_channels: int,
        dropout: float,
        heads: int,
    ):
        super().__init__(metadata, in_channels, hidden_channels)
        self.gnn = to_hetero(
            GATv2Model(
                in_channels=hidden_channels,
                hidden_channels=hidden_channels,
                dropout=dropout,
                heads=heads,
            ),
            metadata,
            aggr="mean",
        )

    def forward(self, graph: HeteroData) -> torch.Tensor:
        x_dict = self.build_x_dict(graph)

        gnn_output = self.gnn(x_dict, graph.edge_index_dict)

        company_x = gnn_output["L"]

        return self.classification_head(company_x).squeeze(-1)
