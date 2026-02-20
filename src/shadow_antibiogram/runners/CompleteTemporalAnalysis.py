from pathlib import Path
from shadow_antibiogram.runners.Phases.Phase_I2A import run_temp_basic
# from shadow_antibiogram.runners.Phases.Phase_I3 import run_phase_I3
from shadow_antibiogram.runners.Phases.Phase_I2 import run_phase_I2
from shadow_antibiogram.controllers.DataLoader import DataLoader
from shadow_antibiogram.runners.Phases.Phase_I1 import run_phase_I1

def ensure_output_dirs(base_dir="./outputs", verbose=True):
    base = Path(base_dir)
    visuals = base / "descriptive_visuals"

    for p in (base, visuals):
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
            if verbose:
                print(f"Created directory: {p}")
        elif verbose:
            print(f"Directory exists: {p}")

    return base, visuals


if __name__ == "__main__":
    #################################################################################
    ##################### LOAD DATASETS FOR ALL PROCESSES BELOW #####################
    #################################################################################
    ensure_output_dirs()
    loader = DataLoader("./datasets/WHO_Aware_data")
    df_combined = loader.get_combined()
    df = df_combined

    # ===========Visual Analytics Pipeline for Antibiotic Testing Coverage============
    run_phase_I1(df)
    # # ==========Analysis Pipeline for Antibiotic Testing Disparities & Trends=========
    run_phase_I2(main_df=df)
    # # =====================Pipeline for AST Panel Breadth Analysis====================
    # run_phase_I3(main_df=df) # might not be needed anymore

    run_temp_basic(data_loader=loader, df=df)
