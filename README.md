# SALSA: Salience-Based Switching Attack for Adversarial Perturbations in Fake News Detection Models

This repository contains anonymized code for our submission to ECIR 2024.

## Requirements - External libraries

Clone the repository and create a virtual environment with the following
libraries from pypi and a python version >= 3.6 to execute all the files with
full functionality.

```bash
numpy
pandas
matplotlib
seaborn
tqdm
transformers
torch
scikit-learn
scipy
shap
spacy
swifter
nltk
```

## Reproduction steps

The data folder should contain the NELA and TFG datasets. Each dataset has its
own folder, and each of those folders should have three files : ```train.csv```,
```val.csv``` and ```test.csv```. (The data is commonly available online. Please
reach out to us if you need access to the processed splits of the dataset)

The code is contained in the ```src``` directory.

- ```data_load.py``` defines data loading utils for the NELA and TFG datasets
  using PyTorch syntax.
- ```bert_model.py``` defines a wrapper class for BERT and RoBERTa, that can be
  used as a classifier.
- ```train.py``` defines utils for fine-tuning the models on the NELA and TFG
  datasets, and evaluating validation and testing performance. Trained
  checkpoints would get saved in the ```models``` folder.
- ```shapipy.py``` defines utils for computing SHAP values for the models on the
  validation and testing sets. Outputs would get stored in the test and val
  splits directly.
- ```attack_prep.py``` defines utils for preparing the data for the attack by
  applying relevant filters to SHAP outputs for validation and test sets to get
  the attack candidates and important words. Outputs are stored in
  ```outputs/shap_outputs``` folder for attack candidates. Important words would
  get directly stored in the test split itself.
- ```attack.py``` defines utils for all attacks in our paper. It loads the test
  split and perturbs it.
- ```attack_helper.py``` defines utils for predicting labels for the perturbed data.
- ```attack_eval.py``` defines utils for evaluating the attack performance using
  ERI, ATR and ARR metrics from our paper.
- The three attack files use configurations defined in the configs folder which
  can be controlled using the meta config file.

First finetune a model on a dataset. Then run shap on validation and test splits
for that dataset and model. Then run attack prep on the validation and test to
get required inputs for the attacks. To run and evaluate the attacks,execute the
following command after setting relevant config in the
```configs/meta_config.yaml``` file.

```python attack.py && python attack_helper.py && python attack_eval.py```

Output excel file with results corresponding to this configuration would then get saved to
the outputs folder. Intermediate attack files would get saved to ```data/attack_files```.

## Results

Results for all experiments referred to in the paper are given in the
```outputs/excel_outputs``` folder. It includes xlsx files organized into
subfolders for each dataset.

The main structure of the repository is as follows :
```bash
.
├── README.md
├── __init__.py
├── configs
│   ├── meta_config.yaml
│   ├── nela.yaml
│   └── tfg.yaml
├── data
│   ├── attack_files
│   ├── nela
│   └── tfg
├── models
├── outputs
│   ├── excel_outputs
│   │   ├── nela
│   │   │   ├── ads
│   │   │   │   ├── ...
│   │   │   ├── default
│   │   │   │   ├── transfer study
│   │   │   │   ├── hyperparameter study
│   │   │   │   │   ├── ...
│   │   │   │   ├── ...
│   │   │   ├── noun
│   │   │   │   ├── ...
│   │   │   └── verb
│   │   │       ├── ...
│   │   └── tfg
│   │       ├── ads
│   │       │   ├── ...
│   │       ├── default
│   │       │   ├── hyperparameter study
│   │       │   │   ├── ...
│   │       │   ├── ...
│   │       ├── noun
│   │       │   ├── ...
│   │       └── verb
│   │           ├── ...
│   └── shap_outputs
│       ├── ...
│       ├── ...
└── src
    ├── __init__.py
    ├── attack.py
    ├── attack_eval.py
    ├── attack_helper.py
    ├── attack_prep.py
    ├── bert_model.py
    ├── data_load.py
    ├── shapipy.py
    └── train.py
```
