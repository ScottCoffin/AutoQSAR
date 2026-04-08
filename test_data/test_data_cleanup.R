library(tidyverse)

# import test data from QACs_OEHHA repo
starting_data <- read_csv("test_data/comprehensive_dataset.csv")

# isolate repro/dev as target and QSAR_Ready_smiles
repro_dev_data <- starting_data %>%
  select(SMILES = QSAR_READY_SMILES, TARGET = `Repro/Dev`) |>
  drop_na()

#save csv for processing in autoQSAR
write_csv(repro_dev_data, "test_data/repro_dev_data.csv")
