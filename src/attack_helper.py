import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoTokenizer

from bert_model import BertClassifier
# from bert_model_2 import BertClassifier

warnings.filterwarnings("ignore")

with open("../configs/meta_config.yaml", "r") as f:
    meta_config = yaml.load(f, Loader=yaml.FullLoader)

CONFIG_PATH = meta_config["CONFIG_PATH"][0]
EXPERIMENT = meta_config["EXPERIMENT"][0]
NER = meta_config["NER"][0]

# Set seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Load config file
with open(CONFIG_PATH, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# get experiment config
experiment_config = config[EXPERIMENT]
print("Experiment config: ", experiment_config)

# Arguments  - nela is only implemented for negation and adverb attack here
DATASET_NAME = experiment_config[0]["DATASET_NAME"][
    0
]  # "kaggle_fake_news", "liar", "nela", "tfg", "isot"
INSERT_POSITION = experiment_config[2]["INSERT_POSITION"][0]  # "random", "head", "tail"
FALSE_CATEGORY = experiment_config[3]["FALSE_CATEGORY"][0]  # "fp", "fn"
METHOD = experiment_config[1]["METHOD"][
    0
]  # "salience", "freq", "adverb", "negation", "sim_switch_freq", "sim_switch_salience", "perplexity_freq", "perplexity_salience"

# Model weights
MODELS_PATH = "../models"
if DATASET_NAME == "kaggle_fake_news":
    MODEL_PATH = f"{MODELS_PATH}/kaggle_fake_news_14-07-23-02_35_fine-tune_distilbert-base-uncased_model_final.pt"
elif DATASET_NAME == "liar":
    MODEL_PATH = f"{MODELS_PATH}/liar_14-07-23-04_19_fine-tune_distilbert-base-uncased_model_final.pt"
elif DATASET_NAME == "nela":
    MODEL_PATH = f"{MODELS_PATH}/nela_04-08-23-21_48_fine-tune_distilbert-base-uncased_model_final.pt"
    # MODEL_PATH = f"{MODELS_PATH}/nela_04-08-23-23_03_fine-tune_roberta-base_model_final.pt"
elif DATASET_NAME == "tfg":
    MODEL_PATH = f"{MODELS_PATH}/tfg_19-07-23-06_11_fine-tune_distilbert-base-uncased_model_final.pt"
elif DATASET_NAME == "isot":
    MODEL_PATH = f"{MODELS_PATH}/isot_19-07-23-20_24_fine-tune_distilbert-base-uncased_model_final.pt"
elif DATASET_NAME == "ti_cnn":
    MODEL_PATH = f"{MODELS_PATH}/ticnn_19-07-23-22_05_fine-tune_distilbert-base-uncased_model_final.pt"

# Paths
DATA_PATH = "../data/attack_files"
if METHOD in ["salience", "freq"]:
    if NER == "default":
        TEST_DATA_PATH = f"{DATA_PATH}/{DATASET_NAME}/{INSERT_POSITION}_{FALSE_CATEGORY}_inject_test_data_{METHOD}.csv"
        OUTPUT_PATH = f"{DATA_PATH}/{DATASET_NAME}/{INSERT_POSITION}_{FALSE_CATEGORY}_inject_test_data_{METHOD}_with_preds.csv"
    elif NER in ["noun", "verb", "ads"]:
        TEST_DATA_PATH = f"{DATA_PATH}/{DATASET_NAME}/{INSERT_POSITION}_{FALSE_CATEGORY}_inject_test_data_{METHOD}_{NER}.csv"
        OUTPUT_PATH = f"{DATA_PATH}/{DATASET_NAME}/{INSERT_POSITION}_{FALSE_CATEGORY}_inject_test_data_{METHOD}_{NER}_with_preds.csv"
elif METHOD in [
    "adverb",
    "negation",
    "sim_switch_freq",
    "sim_switch_salience",
    "perplexity_freq",
    "perplexity_salience",
    "stss",
]:
    if NER == "default":
        TEST_DATA_PATH = (
            f"{DATA_PATH}/{DATASET_NAME}/{FALSE_CATEGORY}_test_data_{METHOD}.csv"
        )
        OUTPUT_PATH = f"{DATA_PATH}/{DATASET_NAME}/{FALSE_CATEGORY}_test_data_{METHOD}_with_preds.csv"
    elif NER in ["noun", "verb", "ads"]:
        TEST_DATA_PATH = (
            f"{DATA_PATH}/{DATASET_NAME}/{FALSE_CATEGORY}_test_data_{METHOD}_{NER}.csv"
        )
        OUTPUT_PATH = f"{DATA_PATH}/{DATASET_NAME}/{FALSE_CATEGORY}_test_data_{METHOD}_{NER}_with_preds.csv"

print(MODEL_PATH, TEST_DATA_PATH, OUTPUT_PATH)

# Tokenizer and model settings
MAX_SEQUENCE_LENGTH = 128 if DATASET_NAME == "liar" else 512
BATCH_SIZE = 64 if DATASET_NAME == "liar" else 32
# MODEL_NAME = "distilbert-base-uncased"
MODEL_NAME = "roberta-base"
MODE = "fine-tune"

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # Load the attack test data
    df = pd.read_csv(TEST_DATA_PATH)
    df = df[df["modified_content"].apply(lambda x: isinstance(x, str))]

    # Tokenize the attack test data
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoded = tokenizer.batch_encode_plus(
        df["modified_content"].tolist(),
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        padding="max_length",
    )
    input_ids = torch.tensor(encoded["input_ids"]).to(device)
    attention_mask = torch.tensor(encoded["attention_mask"]).to(device)

    # Create a DataLoader for the attack test data
    attack_test_dataset = torch.utils.data.TensorDataset(input_ids, attention_mask)
    attack_test_dataloader = torch.utils.data.DataLoader(
        attack_test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    # Load the model
    model = BertClassifier(name=MODEL_NAME, mode=MODE, pretrained_checkpoint=None)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)
    model.eval()

    # Use the model to make predictions on the attack test data
    predicted_labels = []
    with torch.no_grad():
        for batch in attack_test_dataloader:
            input_ids, attention_mask = batch
            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predicted_labels.extend(preds.cpu().numpy())

    # Add the predicted labels to the DataFrame as a new column
    df["predicted_label_after_attack"] = predicted_labels

    # find label flips in df
    flips = df[df["predicted_labels"] != df["predicted_label_after_attack"]]
    print(
        f"\nNumber of label flips in test set is {len(flips)} out of {len(df)} items\n"
    )

    # Save the modified DataFrame to a new CSV file
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")
