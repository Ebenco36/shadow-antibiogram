from src.controllers.AMR.experiments.temporal_analysis_extended import temporal_complement

from src.controllers.DataLoader import DataLoader
from src.runners.Phases.Phase_I1 import run_phase_I1
from src.runners.Phases.Phase_I2A import run_temp_basic

loader = DataLoader("./datasets/WHO_Aware_data")
df = loader.get_combined()

run_temp_basic(data_loader=loader, df=df)
# temporal_complement()
# run_phase_I1(main_df=df)