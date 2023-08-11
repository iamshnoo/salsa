import ast
import calendar
import json
import random

import numpy as np
import pandas as pd
import spacy
import swifter
from nltk.corpus import words
from tqdm import tqdm

# random seed settings
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

nlp = spacy.load("en_core_web_lg", disable=["parser", "ner"])

# Arguments
DATASET_NAME = "nela"  # "isot", "nela", "liar", "kaggle_fake_news", "tfg", "ti_cnn"
STRATEGY = "test"  # "valid", "test"
NER = "default"  # "default", "other"
CANDIDATES_COUNT = 100 # 25, 50, 100, 150, 200
IMPORTANT_WORDS_COUNT = 20 # 5, 10, 20, 30, 40


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
        # TEST_PATH = f"{INPUTS}/nela_04-08-23-21_51_fine-tune_distilbert-base-uncased_valid_results_all_roberta.csv"
        # TEST_PATH = f"{INPUTS}/nela_04-08-23-23_08_fine-tune_roberta-base_valid_results_all_distilbert.csv"
        # TEST_PATH = f"{INPUTS}/nela_04-08-23-23_08_fine-tune_roberta-base_valid_results_all.csv"

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
        # TEST_PATH = f"{INPUTS}/nela_04-08-23-21_50_fine-tune_distilbert-base-uncased_test_results_all.csv"
        # TEST_PATH = f"{INPUTS}/nela_04-08-23-21_50_fine-tune_distilbert-base-uncased_test_results_all_roberta.csv"
        # TEST_PATH = f"{INPUTS}/nela_04-08-23-23_05_fine-tune_roberta-base_test_results_all_distilbert.csv"
        TEST_PATH = f"{INPUTS}/nela_04-08-23-23_05_fine-tune_roberta-base_test_results_all.csv"


def compute_attack_candidates(row, candidates, labels, shap_score):
    if row["predicted_labels"] == labels:
        for word, score in ast.literal_eval(row[shap_score]):
            if word not in candidates:
                candidates[word] = abs(score)
            else:
                if score > candidates[word]:
                    candidates[word] = abs(score)


# Define filters
filters = set(calendar.day_name).union(set(calendar.month_name)).union({""})
# also include day names in lower case
filters = filters.union({day.lower() for day in calendar.day_name}).union(
    {month.lower() for month in calendar.month_name}
)
# also include short forms of day names and months
filters = filters.union({day[:3].lower() for day in calendar.day_name}).union(
    {month[:3].lower() for month in calendar.month_name}
)

# Convert list of words to set for faster lookup
english_words = set(words.words())


def is_valid(word):
    if word in filters:
        return False
    if len(word) <= 3:
        if word in ["cnn"]:
            return True
        return False
    if word in english_words:
        return False
    return True


def filter_candidates(candidates):
    df = pd.Series(candidates).to_frame().reset_index()
    df.columns = ["word", "count"]
    df["is_valid"] = df["word"].swifter.apply(is_valid)
    valid_candidates = df[df["is_valid"]].set_index("word")["count"].to_dict()
    return valid_candidates


def extract_pos(words):
    noun_dict = {}
    verb_dict = {}
    adj_adv_dict = {}

    for word, score in words.items():
        # get the pos tag of each word
        token = nlp(word)[0]
        if token.pos_ == "NOUN" or token.pos_ == "PROPN":
            noun_dict[word] = score
        elif token.pos_ == "VERB":
            verb_dict[word] = score
        elif token.pos_ == "ADJ" or token.pos_ == "ADV":
            adj_adv_dict[word] = score

    return noun_dict, verb_dict, adj_adv_dict


if __name__ == "__main__":
    df = pd.read_csv(TEST_PATH)
    # print(df.columns)
    # df = df.rename(columns={"content": "original_content"})
    # df = df.rename(columns={"processed_content": "content"})
    # print(df.columns)
    # df.to_csv(TEST_PATH, index=False)

    if STRATEGY == "valid":
        # real attack candidates = (pos scores from 0) + (neg scores from 1)
        # we want true label real to be predicted as fake
        real_attack_candidates = {}
        df.swifter.apply(
            compute_attack_candidates,
            args=(real_attack_candidates, 0, "shap_pos_outs"),
            axis=1,
        )
        df.swifter.apply(
            compute_attack_candidates,
            args=(real_attack_candidates, 1, "shap_neg_outs"),
            axis=1,
        )
        # sort and store by abs shap score
        real_attack_candidates = {
            k: v
            for k, v in sorted(
                real_attack_candidates.items(), key=lambda item: item[1], reverse=True
            )
        }

        # fake attack candidates = (pos scores from 1) + (neg scores from 0)
        # we want true label fake to be predicted as real
        fake_attack_candidates = {}
        df.swifter.apply(
            compute_attack_candidates,
            args=(fake_attack_candidates, 1, "shap_pos_outs"),
            axis=1,
        )
        df.swifter.apply(
            compute_attack_candidates,
            args=(fake_attack_candidates, 0, "shap_neg_outs"),
            axis=1,
        )
        # sort and store by abs shap score
        fake_attack_candidates = {
            k: v
            for k, v in sorted(
                fake_attack_candidates.items(), key=lambda item: item[1], reverse=True
            )
        }

        # Assuming 'fake_attack_candidates' and 'real_attack_candidates' are your input dictionaries
        fake_attack_candidates_filtered = filter_candidates(fake_attack_candidates)
        real_attack_candidates_filtered = filter_candidates(real_attack_candidates)

        if NER == "default":
            # store first 100 candidates of each
            fake_attack_candidates = {
                k: v
                for k, v in list(fake_attack_candidates_filtered.items())[
                    :CANDIDATES_COUNT
                ]
            }
            real_attack_candidates = {
                k: v
                for k, v in list(real_attack_candidates_filtered.items())[
                    :CANDIDATES_COUNT
                ]
            }

            # remove the symbol "ƒ†" and "Ġ" from each word
            fake_attack_candidates = {
                word.replace("ƒ†", "").replace("Ġ", ""): score
                for word, score in fake_attack_candidates.items()
            }
            real_attack_candidates = {
                word.replace("ƒ†", "").replace("Ġ", ""): score
                for word, score in real_attack_candidates.items()
            }

            with open(
                f"../outputs/shap_outputs/{DATASET_NAME}_fake_attack_candidates.json",
                "w",
            ) as f:
                json.dump(fake_attack_candidates, f)
            with open(
                f"../outputs/shap_outputs/{DATASET_NAME}_real_attack_candidates.json",
                "w",
            ) as f:
                json.dump(real_attack_candidates, f)

        else:
            print("Extracting POS tags from fake attack candidates ...")
            fake_nouns, fake_verbs, fake_adj_adv = extract_pos(
                fake_attack_candidates_filtered
            )
            print("Extracting POS tags from real attack candidates ...")
            real_nouns, real_verbs, real_adj_adv = extract_pos(
                real_attack_candidates_filtered
            )
            print("Storing POS tags ...")
            # store first 100 candidates of each
            fake_nouns = {k: v for k, v in list(fake_nouns.items())[:CANDIDATES_COUNT]}
            fake_verbs = {k: v for k, v in list(fake_verbs.items())[:CANDIDATES_COUNT]}
            fake_adj_adv = {
                k: v for k, v in list(fake_adj_adv.items())[:CANDIDATES_COUNT]
            }
            real_nouns = {k: v for k, v in list(real_nouns.items())[:CANDIDATES_COUNT]}
            real_verbs = {k: v for k, v in list(real_verbs.items())[:CANDIDATES_COUNT]}
            real_adj_adv = {
                k: v for k, v in list(real_adj_adv.items())[:CANDIDATES_COUNT]
            }

            with open(
                f"../outputs/shap_outputs/{DATASET_NAME}_fake_attack_candidates_noun.json",
                "w",
            ) as f:
                json.dump(fake_nouns, f)
            with open(
                f"../outputs/shap_outputs/{DATASET_NAME}_fake_attack_candidates_verb.json",
                "w",
            ) as f:
                json.dump(fake_verbs, f)
            with open(
                f"../outputs/shap_outputs/{DATASET_NAME}_fake_attack_candidates_ads.json",
                "w",
            ) as f:
                json.dump(fake_adj_adv, f)
            with open(
                f"../outputs/shap_outputs/{DATASET_NAME}_real_attack_candidates_noun.json",
                "w",
            ) as f:
                json.dump(real_nouns, f)
            with open(
                f"../outputs/shap_outputs/{DATASET_NAME}_real_attack_candidates_verb.json",
                "w",
            ) as f:
                json.dump(real_verbs, f)
            with open(
                f"../outputs/shap_outputs/{DATASET_NAME}_real_attack_candidates_ads.json",
                "w",
            ) as f:
                json.dump(real_adj_adv, f)

    elif STRATEGY == "test":
        df["important_words"] = None

        # storing switching tokens
        for idx, row in tqdm(
            df.iterrows(), total=len(df), desc="Storing switching tokens"
        ):
            pos_scores = ast.literal_eval(
                row["shap_pos_outs"]
            )  # already sorted in order of abs score
            neg_scores = ast.literal_eval(
                row["shap_neg_outs"]
            )  # already sorted in order of abs score

            # apply ths is_valid function to filter out switching tokens
            pos_scores = [(word, score) for word, score in pos_scores if is_valid(word)]
            neg_scores = [(word, score) for word, score in neg_scores if is_valid(word)]

            # keep unique words in each list
            pos_scores = list(dict.fromkeys(pos_scores))
            neg_scores = list(dict.fromkeys(neg_scores))

            # fake attack
            if row["true_labels"] == 0:
                if row["predicted_labels"] == 0:
                    # need to reverse neg scores
                    neg_scores.reverse()

                    important_words = []
                    for word, score in pos_scores:
                        if len(important_words) < IMPORTANT_WORDS_COUNT:
                            important_words.append(word)
                        else:
                            break
                    for word, score in neg_scores:
                        if len(important_words) < IMPORTANT_WORDS_COUNT:
                            important_words.append(word)
                        else:
                            break

                elif row["predicted_labels"] == 1:
                    # need to reverse pos scores
                    pos_scores.reverse()

                    important_words = []
                    for word, score in neg_scores:
                        if len(important_words) < IMPORTANT_WORDS_COUNT:
                            important_words.append(word)
                        else:
                            break

                    for word, score in pos_scores:
                        if len(important_words) < IMPORTANT_WORDS_COUNT:
                            important_words.append(word)
                        else:
                            break

            elif row["true_labels"] == 1:
                if row["predicted_labels"] == 0:
                    # need to reverse pos scores
                    pos_scores.reverse()

                    important_words = []
                    for word, score in neg_scores:
                        if len(important_words) < IMPORTANT_WORDS_COUNT:
                            important_words.append(word)
                        else:
                            break

                    for word, score in pos_scores:
                        if len(important_words) < IMPORTANT_WORDS_COUNT:
                            important_words.append(word)
                        else:
                            break

                elif row["predicted_labels"] == 1:
                    # need to reverse neg scores
                    neg_scores.reverse()

                    important_words = []
                    for word, score in pos_scores:
                        if len(important_words) < IMPORTANT_WORDS_COUNT:
                            important_words.append(word)
                        else:
                            break
                    for word, score in neg_scores:
                        if len(important_words) < IMPORTANT_WORDS_COUNT:
                            important_words.append(word)
                        else:
                            break

            # some sentences may have less than specified number of important words
            # use all tokens in the sentence if this is the case
            if len(important_words) < IMPORTANT_WORDS_COUNT:
                important_words = str(row["content"]).split()
                if len(important_words) > IMPORTANT_WORDS_COUNT:
                    important_words = important_words[:IMPORTANT_WORDS_COUNT]
            if len(important_words) > IMPORTANT_WORDS_COUNT:
                important_words = important_words[:IMPORTANT_WORDS_COUNT]

            df.at[idx, "important_words"] = important_words
            # remove the symbol "ƒ†" and "Ġ" from each word in the list of important
            # words - weirdly only happened only for roberta preds, roberta shap
            df.at[idx, "important_words"] = [
                word.replace("ƒ†", "").replace("Ġ", "") for word in important_words
            ]

        df.to_csv(TEST_PATH, index=False)
