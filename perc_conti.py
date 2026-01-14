from src.controllers.AMR.config.experiment_config import ExperimentConfig
from src.controllers.AMR.experiments.grid_search_runner import GridSearchRunner
from src.controllers.AMR.use_cases.helper import filter_continuous_organisations
config = ExperimentConfig.default()
runner = GridSearchRunner(config)

new_df = runner.data_loader.get_combined()
df = new_df
if "NumberOrganisation" in df.columns.to_list():
    res = filter_continuous_organisations(
        df,
        org_col="NumberOrganisation",
        year_col="Year",       # will use if present
        date_col="Date",       # used only if Year missing
        min_year=2020,
        max_year=2022,
        verbose=True,
    )

    df_cont = res.df_continuous
    orgs = res.continuous_orgs

    print(f"Continuous organisations: {len(orgs)}")
    print(f"Isolates retained: {len(df_cont):,}")
    continuous_participation_perctage = len(df_cont)/len(new_df) * 100
    print(f"Percentage of isolates retained after filtering: {continuous_participation_perctage:.2f}%")
else:
    print("NumberOrganisation column not found in dataframe.")