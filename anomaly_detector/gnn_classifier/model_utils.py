import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.typing import Metadata

from anomaly_detector.gnn_classifier.gatv2_classifier import GatV2Classifier
from anomaly_detector.gnn_classifier.mlp_classifier import MLPClassifier
from anomaly_detector.gnn_classifier.sageconv_classifier import SageConvGNNClassifier
from anomaly_detector.utils import AVAILABLE_GNN_MODELS_TYPE


def build_model(
    metadata: Metadata,
    in_channels: int,
    hidden_channels: int,
    dropout: float,
    heads: int | None,  # Only used for GATv2 model, ignored for other models
    layer_type: AVAILABLE_GNN_MODELS_TYPE,
) -> nn.Module:
    if layer_type == "mlp":
        model = MLPClassifier(
            input_dim=in_channels,
            hidden_dim=hidden_channels,
            dropout=dropout,
        )
    elif layer_type == "sageconv":
        model = SageConvGNNClassifier(
            metadata=metadata,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            dropout=dropout,
        )
    elif layer_type == "gatv2":
        model = GatV2Classifier(
            metadata=metadata,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            dropout=dropout,
            heads=heads,
        )
    else:
        raise ValueError(f"Invalid layer_type: {layer_type}")

    return model


@torch.no_grad()
def predict(
    model: nn.Module,
    graph: HeteroData,
) -> tuple[torch.Tensor, torch.Tensor]:
    is_mlp = isinstance(model, MLPClassifier)

    model.eval()

    y_true = graph["L"].y.cpu().numpy()

    if is_mlp:
        y_score = torch.sigmoid(model(graph["L"].x)).cpu().numpy()
    else:
        y_score = torch.sigmoid(model(graph)).cpu().numpy()

    return y_true, y_score
