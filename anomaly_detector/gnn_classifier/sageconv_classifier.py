import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.typing import Metadata

from anomaly_detector.gnn_classifier.gnn_classifier_node_emb_base import (
    GNNClassifierNodeEmbeddingsBase,
)


class SageConvModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        dropout: float,
    ):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index).relu()
        return x


class SageConvGNNClassifier(GNNClassifierNodeEmbeddingsBase):
    def __init__(
        self,
        metadata: Metadata,
        in_channels: int,
        hidden_channels: int,
        dropout: float,
    ):
        super().__init__(metadata, in_channels, hidden_channels)
        self.gnn = to_hetero(
            SageConvModel(
                in_channels=hidden_channels,
                hidden_channels=hidden_channels,
                dropout=dropout,
            ),
            metadata,
            aggr="mean",
        )

    def forward(self, graph: HeteroData) -> torch.Tensor:
        x_dict = self.build_x_dict(graph)

        gnn_output = self.gnn(x_dict, graph.edge_index_dict)

        company_x = gnn_output["L"]

        return self.classification_head(company_x).squeeze(-1)
