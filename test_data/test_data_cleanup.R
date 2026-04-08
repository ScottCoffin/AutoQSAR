library(tidyverse)

# import test data from QACs_OEHHA repo
starting_data <- read_csv("test_data/comprehensive_dataset.csv")

# isolate repro/dev as target and QSAR_Ready_smiles
repro_dev_data <- starting_data %>%
  select(SMILES = QSAR_READY_SMILES, TARGET = `Repro/Dev`) |>
  drop_na()

#save csv for processing in autoQSAR
write_csv(repro_dev_data, "test_data/repro_dev_data.csv")

#### TK dataset cleanup from PFAS_QSAR_TK repo ####
tk_complete <- read_csv("test_data/complete_tk.csv") |>
  mutate(SMILES = case_when(is.na(QSAR_READY_SMILES) ~ SMILES, T ~ SMILES)) |>
  drop_na(SMILES, standard_value)

skimr::skim(tk_complete)
VDss <- tk_complete |>
  filter(standard_endpoint == "VDss", species_name == "Human") |>
  mutate(TARGET = standard_value)

HLe_invivo <- tk_complete |>
  filter(standard_endpoint == "HLe_invivo", species_name == "Human") |>
  mutate(TARGET = standard_value)

fu <- tk_complete |>
  filter(standard_endpoint == "fu", species_name == "Human") |>
  mutate(TARGET = standard_value)

# save all to csv
write_csv(VDss, "test_data/VDss.csv")
write_csv(HLe_invivo, "test_data/HLe_invivo.csv")
write_csv(fu, "test_data/fu.csv")


## examine how many PFAS are in each dataset
VDss |> group_by(group) |> summarize(n_distinct(SMILES))
