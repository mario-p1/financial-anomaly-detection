from itertools import chain, repeat

import mlflow
import numpy as np
import torch
import typer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from torch import nn
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData
from torch_geometric.typing import Metadata

from anomaly_detector.gnn_classifier.dataset import (
    build_gnn_graphs,
    compute_pos_weight,
)
from anomaly_detector.gnn_classifier.mlp_classifier import MLPClassifier
from anomaly_detector.gnn_classifier.model_utils import build_model, predict
from anomaly_detector.metrics import (
    compute_metrics,
    find_best_threshold,
    mlflow_log_metrics,
    print_metrics,
)
from anomaly_detector.utils import AVAILABLE_GNN_MODELS_TYPE


class GraphsDataModule(LightningDataModule):
    """Serves one full graph per step: a batch is a whole year's graph."""

    def __init__(
        self,
        train_year_start: int,
        train_year_end: int,
        valid_year_start: int,
        valid_year_end: int,
        test_year_start: int | None = None,
        test_year_end: int | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.graphs = None

    def setup(self, stage: str | None = None) -> None:
        if self.graphs is not None:
            return
        self.graphs = build_gnn_graphs(
            train_year_start=self.hparams.train_year_start,
            train_year_end=self.hparams.train_year_end,
            valid_year_start=self.hparams.valid_year_start,
            valid_year_end=self.hparams.valid_year_end,
            test_year_start=self.hparams.test_year_start,
            test_year_end=self.hparams.test_year_end,
        )

    def train_dataloader(self) -> DataLoader:
        # batch_size=None disables batching: each graph is yielded as it is.
        return DataLoader(
            list(self.graphs.train_graphs.values()),
            batch_size=None,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            list(self.graphs.valid_graphs.items()),
            batch_size=None,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            list(self.graphs.test_graphs.items()),
            batch_size=None,
        )


class GNNClassifier(LightningModule):
    def __init__(
        self,
        metadata: Metadata,
        in_channels: int,
        pos_weight: float,
        learning_rate: float,
        hidden_channels: int,
        dropout: float,
        layer_type: AVAILABLE_GNN_MODELS_TYPE,
        heads: int | None,  # Only used for GATv2 model, ignored for other models
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = build_model(
            metadata=metadata,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            dropout=dropout,
            layer_type=layer_type,
            heads=heads,
        )
        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight))

    def forward(self, graph: HeteroData) -> torch.Tensor:
        if isinstance(self.model, MLPClassifier):
            return self.model(graph["L"].x)

        return self.model(graph)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def training_step(self, graph: HeteroData, batch_idx: int) -> torch.Tensor:
        loss = self.loss_fn(self(graph), graph["L"].y)
        self.log("Loss/train", loss, on_epoch=True,
                 batch_size=1, prog_bar=True)

        return loss

    def validation_step(
        self,
        data: tuple[int, HeteroData],
    ) -> None:
        year, graph = data

        logits = self(graph)
        metrics = compute_metrics(
            graph["L"].y.cpu().numpy(),
            torch.sigmoid(logits).cpu().numpy(),
        )

        self.log_dict(
            {
                f"Loss/eval_{year}": self.loss_fn(logits, graph["L"].y),
                f"Au_prc/eval_{year}": metrics["au_prc"],
            },
            batch_size=1,
            add_dataloader_idx=False,
        )

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self.checkpoint_epoch = checkpoint["epoch"]

    def predict_graph(self, graph: HeteroData) -> tuple[np.ndarray, np.ndarray]:
        """Returns the true labels and the predicted scores for a whole graph."""
        return predict(self.model, graph.to(self.device))


def main(
    train_year_start: int = typer.Option(),
    train_year_end: int = typer.Option(),
    valid_year_start: int = typer.Option(),
    valid_year_end: int = typer.Option(),
    test_year_start: int | None = typer.Option(None),
    test_year_end: int | None = typer.Option(None),
    layer_type: AVAILABLE_GNN_MODELS_TYPE = typer.Option(),
    epochs: int = typer.Option(),
    learning_rate: float = typer.Option(),
    hidden_channels: int = typer.Option(),
    dropout: float = typer.Option(),
    heads: int | None = typer.Option(
        None, help="Number of attention heads for GATv2 model"
    ),
):
    seed_everything(42)

    datamodule = GraphsDataModule(
        train_year_start=train_year_start,
        train_year_end=train_year_end,
        valid_year_start=valid_year_start,
        valid_year_end=valid_year_end,
        test_year_start=test_year_start,
        test_year_end=test_year_end,
    )
    datamodule.setup()

    tuning_year = valid_year_start

    train_graph = datamodule.graphs.train_graphs[train_year_start]
    module = GNNClassifier(
        metadata=train_graph.metadata(),
        in_channels=train_graph["L"].x.size(-1),
        pos_weight=compute_pos_weight(
            list(datamodule.graphs.train_graphs.values())
        ).item(),
        learning_rate=learning_rate,
        hidden_channels=hidden_channels,
        dropout=dropout,
        layer_type=layer_type,
        heads=heads,
    )

    early_stopping = EarlyStopping(
        monitor=f"Au_prc/eval_{tuning_year}",
        patience=10 * len(datamodule.graphs.train_graphs),
        mode="max",
    )

    checkpointer = ModelCheckpoint(
        monitor=f"Au_prc/eval_{tuning_year}",
        dirpath="checkpoints",
        filename="best_model",
        save_top_k=1,
        mode="max",
    )

    trainer = Trainer(
        max_epochs=epochs,
        logger=TensorBoardLogger(save_dir=".", name="tensorboard_logs"),
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        devices=1,
        callbacks=[early_stopping, checkpointer],
    )

    with mlflow.start_run():
        trainer.fit(module, datamodule.train_dataloader(),
                    datamodule.val_dataloader())

        threshold = find_best_threshold(
            *module.predict_graph(datamodule.graphs.valid_graphs[tuning_year])
        )

        mlflow.log_params(
            {
                "tuning_year": tuning_year,
                "train_year_start": train_year_start,
                "train_year_end": train_year_end,
                "valid_year_start": valid_year_start,
                "valid_year_end": valid_year_end,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "hidden_channels": hidden_channels,
                "dropout": dropout,
                "model": layer_type,
            }
        )

        print("Loading best model from checkpoint...")
        module = GNNClassifier.load_from_checkpoint(
            checkpointer.best_model_path)
        mlflow.log_param("best_epoch", module.checkpoint_epoch)

        for dset, (year, graph) in chain(
            zip(repeat("valid"), datamodule.graphs.valid_graphs.items()),
            zip(repeat("test"), datamodule.graphs.test_graphs.items()),
        ):
            with mlflow.start_run(nested=True):
                mlflow.log_param("dataset", f"{dset}_{year}")
                y_true, y_score = module.predict_graph(graph)
                metrics = compute_metrics(
                    y_true, y_score, at_threshold=threshold)
                mlflow_log_metrics(metrics)

                print(f"\nMetrics for year {year}:")
                print_metrics(metrics)


if __name__ == "__main__":
    typer.run(main)
