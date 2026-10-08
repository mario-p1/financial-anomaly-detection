from pathlib import Path
from typing import Literal

base_project_path = Path(__file__).parent.parent
base_data_path = base_project_path / "data_files"
AVAILABLE_GNN_MODELS_TYPE = Literal["mlp", "sageconv", "gatv2"]
