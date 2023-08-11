import json
import random
import re

import nltk  # for word tokenization during preprocessing
import numpy as np
import pandas as pd
import shap
import spacy  # for NER
import swifter
import torch
from nltk.corpus import stopwords
from tqdm import tqdm
from transformers import AutoTokenizer

from bert_model import BertClassifier

# NLTK stopwords and spacy NER
stop_words = set(stopwords.words("english"))
nlp = spacy.load("en_core_web_sm")

# random seed settings
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Arguments
DATASET_NAME = "nela"  # "liar" # "kaggle_fake_news", "tfg", "isot", "ti_cnn", "nela"
STRATEGY = "test"  # "valid", "test"

# Tokenizer and Model settings
MAX_SEQUENCE_LENGTH = 128 if DATASET_NAME == "liar" else 512
NAME = "distilbert-base-uncased"
MODE = "fine-tune"

# Model checkpoint
if DATASET_NAME == "liar":
    MODEL_CKPT = (
        "../models/liar_14-07-23-04_19_fine-tune_distilbert-base-uncased_model_final.pt"
    )
elif DATASET_NAME == "kaggle_fake_news":
    MODEL_CKPT = "../models/kaggle_fake_news_14-07-23-02_35_fine-tune_distilbert-base-uncased_model_final.pt"
elif DATASET_NAME == "tfg":
    MODEL_CKPT = (
        "../models/tfg_19-07-23-06_11_fine-tune_distilbert-base-uncased_model_final.pt"
    )
elif DATASET_NAME == "isot":
    MODEL_CKPT = (
        "../models/isot_19-07-23-20_24_fine-tune_distilbert-base-uncased_model_final.pt"
    )
elif DATASET_NAME == "ti_cnn":
    MODEL_CKPT = "../models/ticnn_19-07-23-22_05_fine-tune_distilbert-base-uncased_model_final.pt"
elif DATASET_NAME == "nela":
    MODEL_CKPT = (
        "../models/nela_04-08-23-21_48_fine-tune_distilbert-base-uncased_model_final.pt"
    )


# Data path - save results in input df
if STRATEGY == "valid":
    INPUTS = "../outputs"
    if DATASET_NAME == "liar":
        TEST_PATH = f"{INPUTS}/liar_14-07-23-05_46_fine-tune_distilbert-base-uncased_valid_results_all.csv"
    elif DATASET_NAME == "kaggle_fake_news":
        TEST_PATH = f"{INPUTS}/kaggle_fake_news_14-07-23-05_37_fine-tune_distilbert-base-uncased_valid_results_all.csv"
    elif DATASET_NAME == "tfg":
        TEST_PATH = f"{INPUTS}/tfg_19-07-23-06_12_fine-tune_distilbert-base-uncased_valid_results_all.csv"
    elif DATASET_NAME == "isot":
        TEST_PATH = f"{INPUTS}/isot_19-07-23-20_25_fine-tune_distilbert-base-uncased_valid_results_all.csv"
    elif DATASET_NAME == "ti_cnn":
        TEST_PATH = f"{INPUTS}/ticnn_19-07-23-22_05_fine-tune_distilbert-base-uncased_valid_results_all.csv"
    elif DATASET_NAME == "nela":
        TEST_PATH = f"{INPUTS}/nela_04-08-23-21_51_fine-tune_distilbert-base-uncased_valid_results_all.csv"

elif STRATEGY == "test":
    INPUTS = "../outputs"
    if DATASET_NAME == "liar":
        TEST_PATH = f"{INPUTS}/liar_14-07-23-04_19_fine-tune_distilbert-base-uncased_test_results_all.csv"
    elif DATASET_NAME == "kaggle_fake_news":
        TEST_PATH = f"{INPUTS}/kaggle_fake_news_14-07-23-02_35_fine-tune_distilbert-base-uncased_test_results_all.csv"
    elif DATASET_NAME == "tfg":
        TEST_PATH = f"{INPUTS}/tfg_19-07-23-06_12_fine-tune_distilbert-base-uncased_test_results_all.csv"
    elif DATASET_NAME == "isot":
        TEST_PATH = f"{INPUTS}/isot_19-07-23-20_25_fine-tune_distilbert-base-uncased_test_results_all.csv"
    elif DATASET_NAME == "ti_cnn":
        TEST_PATH = f"{INPUTS}/ticnn_19-07-23-22_05_fine-tune_distilbert-base-uncased_test_results_all.csv"
    elif DATASET_NAME == "nela":
        TEST_PATH = f"{INPUTS}/nela_04-08-23-21_50_fine-tune_distilbert-base-uncased_test_results_all.csv"


# torch gpu settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(NAME, use_fast=True)
model = BertClassifier(name=NAME, mode=MODE, pretrained_checkpoint=None)
model.load_state_dict(torch.load(MODEL_CKPT, map_location=device))
model.eval()
model.to(device)


def tokenize(tokenizer, sentences, padding="max_length"):
    encoded = tokenizer.batch_encode_plus(
        sentences, max_length=MAX_SEQUENCE_LENGTH, truncation=True, padding=padding
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    return torch.tensor(input_ids).to(device), torch.tensor(attention_mask).to(device)


def get_model_output(sentences):
    sentences = list(sentences)
    input_ids, attention_mask = tokenize(tokenizer, sentences)
    with torch.no_grad():
        output = model(input_ids, attention_mask)
        probabilities = torch.softmax(output, dim=-1)
    return probabilities.cpu().numpy()


def preprocess_text(text):
    text = re.sub(r"#", "", text.lower())
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    text = " ".join(tokens)
    return text


def shapper(sentence, output_class):
    explainer = shap.Explainer(
        lambda x: get_model_output(x),
        shap.maskers.Text(tokenizer),
        silent=False,
    )
    shap_values = explainer([sentence])
    importance_values = shap_values[:, :, output_class].values
    tokenized_sentence = tokenizer.tokenize(sentence)
    token_importance = list(zip(tokenized_sentence, importance_values[0]))

    # Perform NER
    doc = nlp(sentence)

    # Aggregate salience scores for named entities
    aggregated_token_importance = []
    token_scores = {
        token: score for token, score in token_importance if not token.startswith("##")
    }

    token_scores_aggregated = token_scores.copy()

    for ent in doc.ents:
        scores = [token_scores.get(token, 0) for token in ent.text.split()]
        aggregated_score = sum(scores)
        average_score = aggregated_score / len(scores) if scores else 0
        aggregated_token_importance.append((ent.text, average_score))

        for token in ent.text.split():
            if token in token_scores_aggregated:
                del token_scores_aggregated[token]

    for token, score in token_scores_aggregated.items():
        aggregated_token_importance.append((token, score))

    # Split positive and negative scores
    shap_neg_outs = [item for item in aggregated_token_importance if item[1] < 0]
    # sort by score - largest mod value first
    shap_neg_outs = sorted(shap_neg_outs, key=lambda x: x[1])

    shap_pos_outs = [item for item in aggregated_token_importance if item[1] > 0]
    # sort by score - largest value first
    shap_pos_outs = sorted(shap_pos_outs, key=lambda x: x[1], reverse=True)
    return shap_neg_outs, shap_pos_outs


if __name__ == "__main__":
    df = pd.read_csv(TEST_PATH)
    df = df.dropna(subset=["content"])

    df["processed_content"] = df["content"].apply(preprocess_text)
    print(f"Running shap on {len(df)} samples...")
    print("-" * 80)

    df[["shap_neg_outs", "shap_pos_outs"]] = df.swifter.apply(
        lambda row: pd.Series(
            shapper(row["processed_content"], row["predicted_labels"])
        ),
        axis=1,
    )

    print(df.head())
    print("-" * 80)
    # save df to csv at TEST_PATH
    df.to_csv(TEST_PATH, index=False)
