"""Shared benchmark dataset registries for notebook + CLI workflows."""

from __future__ import annotations

from typing import Any

PYTDC_SOURCE_URL = (
    "https://files.pythonhosted.org/packages/db/bf/"
    "db7525f0e9c48d340a66ae11ed46bbb1966234660a6882ce47d1e1d52824/pytdc-1.1.15.tar.gz"
)

CHEMML_EXAMPLE_OPTIONS = ["organic_density", "cep_homo", "xyz_polarizability"]

TDC_QSAR_OPTIONS: dict[str, dict[str, Any]] = {
    # ADME tasks (https://tdcommons.ai/single_pred_tasks/adme/)
    "caco2_wang": {"task": "ADME", "label": "Caco-2 permeability"},
    "pampa_ncats": {"task": "ADME", "label": "PAMPA permeability"},
    "hia_hou": {"task": "ADME", "label": "Human intestinal absorption"},
    "pgp_broccatelli": {"task": "ADME", "label": "P-gp inhibition"},
    "bioavailability_ma": {"task": "ADME", "label": "Bioavailability"},
    "lipophilicity_astrazeneca": {"task": "ADME", "label": "Lipophilicity"},
    "solubility_aqsoldb": {"task": "ADME", "label": "AqSolDB solubility"},
    "hydrationfreeenergy_freesolv": {"task": "ADME", "label": "Hydration free energy"},
    "bbb_martins": {"task": "ADME", "label": "Blood-brain barrier penetration"},
    "ppbr_az": {"task": "ADME", "label": "Plasma protein binding"},
    "vdss_lombardo": {"task": "ADME", "label": "Volume of distribution"},
    "cyp2c19_veith": {"task": "ADME", "label": "CYP2C19 inhibition"},
    "cyp2d6_veith": {"task": "ADME", "label": "CYP2D6 inhibition"},
    "cyp3a4_veith": {"task": "ADME", "label": "CYP3A4 inhibition"},
    "cyp1a2_veith": {"task": "ADME", "label": "CYP1A2 inhibition"},
    "cyp2c9_veith": {"task": "ADME", "label": "CYP2C9 inhibition"},
    "cyp2c9_substrate_carbonmangels": {"task": "ADME", "label": "CYP2C9 substrate"},
    "cyp2d6_substrate_carbonmangels": {"task": "ADME", "label": "CYP2D6 substrate"},
    "cyp3a4_substrate_carbonmangels": {"task": "ADME", "label": "CYP3A4 substrate"},
    "half_life_obach": {"task": "ADME", "label": "Half-life"},
    "clearance_hepatocyte_az": {"task": "ADME", "label": "Hepatocyte clearance"},
    "clearance_microsome_az": {"task": "ADME", "label": "Microsome clearance"},
    # Tox tasks (https://tdcommons.ai/single_pred_tasks/tox/)
    "ld50_zhu": {"task": "Tox", "label": "Acute toxicity LD50"},
    "herg": {"task": "Tox", "label": "hERG blockers"},
    "herg_central": {"task": "Tox", "label": "hERG Central", "auto_label_name": True},
    "herg_karim": {"task": "Tox", "label": "hERG Karim"},
    "ames": {"task": "Tox", "label": "Ames mutagenicity"},
    "dili": {"task": "Tox", "label": "Drug-induced liver injury"},
    "skin_reaction": {"task": "Tox", "label": "Skin reaction"},
    "carcinogens_lagunin": {"task": "Tox", "label": "Carcinogens"},
    "tox21": {"task": "Tox", "label": "Tox21", "auto_label_name": True},
    "toxcast": {"task": "Tox", "label": "ToxCast", "auto_label_name": True},
    "clintox": {"task": "Tox", "label": "ClinTox"},
}

TDC_LEADERBOARD_URLS: dict[str, str] = {
    "caco2_wang": "https://tdcommons.ai/benchmark/admet_group/01caco2/",
    "lipophilicity_astrazeneca": "https://tdcommons.ai/benchmark/admet_group/05lipo/",
    "solubility_aqsoldb": "https://tdcommons.ai/benchmark/admet_group/06aqsol/",
    "ppbr_az": "https://tdcommons.ai/benchmark/admet_group/08ppbr/",
    "vdss_lombardo": "https://tdcommons.ai/benchmark/admet_group/09vdss/",
    "half_life_obach": "https://tdcommons.ai/benchmark/admet_group/16halflife/",
    "clearance_hepatocyte_az": "https://tdcommons.ai/benchmark/admet_group/17clhepa/",
    "clearance_microsome_az": "https://tdcommons.ai/benchmark/admet_group/18clmicro/",
    "ld50_zhu": "https://tdcommons.ai/benchmark/admet_group/19ld50/",
}

MOLECULENET_LEADERBOARD_README_URL = "https://raw.githubusercontent.com/deepchem/moleculenet/master/README.md"
MOLECULENET_LEADERBOARD_REPO_URL = "https://github.com/deepchem/moleculenet"

MOLECULENET_PHYSCHEM_OPTIONS: dict[str, dict[str, Any]] = {
    "esol_delaney": {
        "option_label": "MoleculeNet | PhysChem | ESOL (Delaney)",
        "dataset_name": "esol_delaney",
        "source_label": "MoleculeNet PhysChem benchmark: ESOL (Delaney)",
        "source_url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
        "smiles_column": "smiles",
        "target_candidates": [
            "measured log solubility in mols per litre",
            "measured log(solubility:mol/L)",
            "target",
        ],
        "recommended_split": "scaffold",
        "recommended_metric": "rmse",
        "leaderboard_section": "Delaney (ESOL)",
        "leaderboard_url": MOLECULENET_LEADERBOARD_REPO_URL,
    },
    "freesolv_sampl": {
        "option_label": "MoleculeNet | PhysChem | FreeSolv (SAMPL)",
        "dataset_name": "freesolv_sampl",
        "source_label": "MoleculeNet PhysChem benchmark: FreeSolv (SAMPL)",
        "source_url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv",
        "smiles_column": "smiles",
        "target_candidates": ["expt", "target"],
        "recommended_split": "random",
        "recommended_metric": "rmse",
        "leaderboard_section": "SAMPL",
        "leaderboard_section_candidates": ["SAMPL", "FreeSolv (SAMPL)", "FreeSolv"],
        "leaderboard_url": MOLECULENET_LEADERBOARD_REPO_URL,
    },
    "lipophilicity": {
        "option_label": "MoleculeNet | PhysChem | Lipophilicity",
        "dataset_name": "lipophilicity",
        "source_label": "MoleculeNet PhysChem benchmark: Lipophilicity",
        "source_url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
        "smiles_column": "smiles",
        "target_candidates": ["exp", "target"],
        "recommended_split": "scaffold",
        "recommended_metric": "rmse",
        "leaderboard_section": "Lipo",
        "leaderboard_url": MOLECULENET_LEADERBOARD_REPO_URL,
    },
}

POLARIS_ADME_OPTIONS: dict[str, dict[str, Any]] = {
    "perm": {
        "option_label": "Polaris | ADME | adme-fang-perm-1",
        "dataset_name": "polaris_adme_fang_perm_1",
        "source_label": "Polaris ADME benchmark mirror: adme-fang-perm-1",
        "benchmark_id": "polaris/adme-fang-perm-1",
        "benchmark_url": "https://polarishub.io/benchmarks/polaris/adme-fang-perm-1",
        "train_url": "https://raw.githubusercontent.com/molecularinformatics/Computational-ADME/main/MPNN/ADME_MDR1_ER_train.csv",
        "test_url": "https://raw.githubusercontent.com/molecularinformatics/Computational-ADME/main/MPNN/ADME_MDR1_ER_test.csv",
        "smiles_column": "smiles",
        "target_column": "activity",
        "recommended_split": "predefined",
        "recommended_metric": "mean_squared_error",
    },
    "solu": {
        "option_label": "Polaris | ADME | adme-fang-solu-1",
        "dataset_name": "polaris_adme_fang_solu_1",
        "source_label": "Polaris ADME benchmark mirror: adme-fang-solu-1",
        "benchmark_id": "polaris/adme-fang-solu-1",
        "benchmark_url": "https://polarishub.io/benchmarks/polaris/adme-fang-solu-1",
        "train_url": "https://raw.githubusercontent.com/molecularinformatics/Computational-ADME/main/MPNN/ADME_Sol_train.csv",
        "test_url": "https://raw.githubusercontent.com/molecularinformatics/Computational-ADME/main/MPNN/ADME_Sol_test.csv",
        "smiles_column": "smiles",
        "target_column": "activity",
        "recommended_split": "predefined",
        "recommended_metric": "mean_squared_error",
    },
    "rclint": {
        "option_label": "Polaris | ADME | adme-fang-rclint-1",
        "dataset_name": "polaris_adme_fang_rclint_1",
        "source_label": "Polaris ADME benchmark mirror: adme-fang-rclint-1",
        "benchmark_id": "polaris/adme-fang-rclint-1",
        "benchmark_url": "https://polarishub.io/benchmarks/polaris/adme-fang-rclint-1",
        "train_url": "https://raw.githubusercontent.com/molecularinformatics/Computational-ADME/main/MPNN/ADME_RLM_train.csv",
        "test_url": "https://raw.githubusercontent.com/molecularinformatics/Computational-ADME/main/MPNN/ADME_RLM_test.csv",
        "smiles_column": "smiles",
        "target_column": "activity",
        "recommended_split": "predefined",
        "recommended_metric": "mean_squared_error",
    },
    "hppb": {
        "option_label": "Polaris | ADME | adme-fang-hppb-1",
        "dataset_name": "polaris_adme_fang_hppb_1",
        "source_label": "Polaris ADME benchmark mirror: adme-fang-hppb-1",
        "benchmark_id": "polaris/adme-fang-hppb-1",
        "benchmark_url": "https://polarishub.io/benchmarks/polaris/adme-fang-hppb-1",
        "train_url": "https://raw.githubusercontent.com/molecularinformatics/Computational-ADME/main/MPNN/ADME_hPPB_train.csv",
        "test_url": "https://raw.githubusercontent.com/molecularinformatics/Computational-ADME/main/MPNN/ADME_hPPB_test.csv",
        "smiles_column": "smiles",
        "target_column": "activity",
        "recommended_split": "predefined",
        "recommended_metric": "mean_squared_error",
    },
    "rppb": {
        "option_label": "Polaris | ADME | adme-fang-rppb-1",
        "dataset_name": "polaris_adme_fang_rppb_1",
        "source_label": "Polaris ADME benchmark mirror: adme-fang-rppb-1",
        "benchmark_id": "polaris/adme-fang-rppb-1",
        "benchmark_url": "https://polarishub.io/benchmarks/polaris/adme-fang-rppb-1",
        "train_url": "https://raw.githubusercontent.com/molecularinformatics/Computational-ADME/main/MPNN/ADME_rPPB_train.csv",
        "test_url": "https://raw.githubusercontent.com/molecularinformatics/Computational-ADME/main/MPNN/ADME_rPPB_test.csv",
        "smiles_column": "smiles",
        "target_column": "activity",
        "recommended_split": "predefined",
        "recommended_metric": "mean_squared_error",
    },
}

PODUAM_POD_OPTIONS: dict[str, dict[str, Any]] = {
    "poduam_pod_nc_std": {
        "option_label": "PODUAM | POD | nc-std",
        "dataset_name": "poduam_pod_nc_std",
        "source_label": "PODUAM benchmark: POD non-cancer standardized set (Aurisano et al., Nature Communications 2025)",
        "source_url": "https://raw.githubusercontent.com/kejbo/PODUAM/main/data/data_pod_nc-std.csv",
        "source_smiles_column": "Canonical_QSARr",
        "source_target_column": "y",
        "smiles_column": "SMILES",
        "target_column": "POD_logmol",
        "recommended_split": "stratified",
        "recommended_metric": "rmse",
        "benchmark_suite": "literature",
        "benchmark_id": "poduam_pod_nc_std",
        "benchmark_url": "https://www.nature.com/articles/s41467-025-67374-4#data-availability",
        "github_repo_url": "https://github.com/kejbo/PODUAM",
        "leaderboard_url": "https://www.nature.com/articles/s41467-025-67374-4#data-availability",
        "leaderboard_summary": {
            "url": "https://www.nature.com/articles/s41467-025-67374-4#data-availability",
            "rank": "1",
            "model": "PODUAM BNN (PODnc)",
            "metric_name": "RMSE",
            "metric_value": "0.73",
            "dataset_split": "10-fold CV (paper-reported)",
            "top10": [
                {"rank": "1", "model": "PODUAM BNN (PODnc)", "metric_name": "RMSE", "metric_value": "0.73"},
                {"rank": "2", "model": "PODUAM BNN (PODnc)", "metric_name": "R2", "metric_value": "0.55"},
            ],
            "source": "literature_static",
            "github_repo": "https://github.com/kejbo/PODUAM",
        },
    },
    "poduam_pod_rd_std": {
        "option_label": "PODUAM | POD | rd-std",
        "dataset_name": "poduam_pod_rd_std",
        "source_label": "PODUAM benchmark: POD reproductive/developmental standardized set (Aurisano et al., Nature Communications 2025)",
        "source_url": "https://raw.githubusercontent.com/kejbo/PODUAM/main/data/data_pod_rd-std.csv",
        "source_smiles_column": "Canonical_QSARr",
        "source_target_column": "y",
        "smiles_column": "SMILES",
        "target_column": "POD_logmol",
        "recommended_split": "stratified",
        "recommended_metric": "rmse",
        "benchmark_suite": "literature",
        "benchmark_id": "poduam_pod_rd_std",
        "benchmark_url": "https://www.nature.com/articles/s41467-025-67374-4#data-availability",
        "github_repo_url": "https://github.com/kejbo/PODUAM",
        "leaderboard_url": "https://www.nature.com/articles/s41467-025-67374-4#data-availability",
        "leaderboard_summary": {
            "url": "https://www.nature.com/articles/s41467-025-67374-4#data-availability",
            "rank": "1",
            "model": "PODUAM BNN (PODrd)",
            "metric_name": "RMSE",
            "metric_value": "0.63",
            "dataset_split": "10-fold CV (paper-reported)",
            "top10": [
                {"rank": "1", "model": "PODUAM BNN (PODrd)", "metric_name": "RMSE", "metric_value": "0.63"},
                {"rank": "2", "model": "PODUAM BNN (PODrd)", "metric_name": "R2", "metric_value": "0.41"},
            ],
            "source": "literature_static",
            "github_repo": "https://github.com/kejbo/PODUAM",
        },
    },
}

FREESOLV_EXPANDED_SCALED_OPTION: dict[str, Any] = {
    "dataset_name": "freesolv_expanded_scaled_2025",
    "source_label": "Expanded Free Solvation Energy dataset (Marques & Muller 2025, scaled benchmark set)",
    "source_file": "data/free_solv/1-s2.0-S0378381225000068-mmc1.xlsx",
    "sheet_name": "Model Benchmarking",
    "split_sheet_name": "Temperature-Dependent Model",
    "benchmark_id": "freesolv_expanded_scaled_2025",
    "benchmark_suite": "literature",
    "benchmark_url": "https://www.sciencedirect.com/science/article/pii/S0378381225000068",
    "recommended_split": "predefined",
    "recommended_metric": "mae",
}


def notebook_example_dataset_options() -> list[str]:
    options: list[str] = [
        "ChemML | organic_density",
        "ChemML | cep_homo",
        "ChemML | xyz_polarizability",
        "ChemML | comp_energy",
        "ChemML | crystal_structures",
    ]
    options.extend([f"TDC | {cfg['task']} | {name}" for name, cfg in TDC_QSAR_OPTIONS.items()])
    options.extend(item["option_label"] for item in MOLECULENET_PHYSCHEM_OPTIONS.values())
    options.extend(item["option_label"] for item in POLARIS_ADME_OPTIONS.values())
    options.extend(item["option_label"] for item in PODUAM_POD_OPTIONS.values())
    return options
