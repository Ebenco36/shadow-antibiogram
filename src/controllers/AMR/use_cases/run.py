from pathlib import Path
import pandas as pd

from src.controllers.AMR.use_cases.UseCase1 import run_temporal_use_case
from src.controllers.AMR.use_cases.UseCase2 import run_context_divergence_use_case


def run_two_key_use_cases(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    run_temporal_use_case(df, output_dir / "use_case1_temporal")
    run_context_divergence_use_case(df, output_dir / "use_case2_context")
