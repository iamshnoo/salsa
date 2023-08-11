import os
import random
import warnings

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

from sklearn.metrics import confusion_matrix

with open("../configs/meta_config.yaml", "r") as f:
    meta_config = yaml.load(f, Loader=yaml.FullLoader)

CONFIG_PATH = meta_config["CONFIG_PATH"][0]
EXPERIMENT = meta_config["EXPERIMENT"][0]
NER = meta_config["NER"][0]

# Set seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Load config file
with open(CONFIG_PATH, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# get experiment config
experiment_config = config[EXPERIMENT]
print("Experiment config: ", experiment_config)

# Arguments - nela is only implemented for negation and adverb attack here
DATASET_NAME = experiment_config[0]["DATASET_NAME"][
    0
]  # "kaggle_fake_news", "liar", "nela", "tfg", "isot"
MODE = experiment_config[4]["MODE"][0]  # "real", "fake"
INSERT_POSITION = experiment_config[2]["INSERT_POSITION"][
    0
]  # "random", "head", "tail", "adverb", "negation", "sim_switch_freq", "sim_switch_salience", "perplexity_freq", "perplexity_salience"
METHOD = experiment_config[1]["METHOD"][
    0
]  # "salience", "freq", "adverb", "negation", "sim_switch_freq", "sim_switch_salience", "perplexity_freq", "perplexity_salience"

FALSE_CATEGORY = "fn" if MODE == "real" else "fp"
assert FALSE_CATEGORY == experiment_config[3]["FALSE_CATEGORY"][0]

# Paths
if DATASET_NAME == "liar":
    OG_DATA = "../outputs/liar_14-07-23-04_19_fine-tune_distilbert-base-uncased_test_results_all.csv"
elif DATASET_NAME == "kaggle_fake_news":
    OG_DATA = "../outputs/kaggle_fake_news_14-07-23-02_35_fine-tune_distilbert-base-uncased_test_results_all.csv"
elif DATASET_NAME == "nela":
    # OG_DATA = (
    #     "../outputs/nela_04-08-23-21_50_fine-tune_distilbert-base-uncased_test_results_all.csv"
    # )
    # OG_DATA = (
    #     "../outputs/nela_04-08-23-21_50_fine-tune_distilbert-base-uncased_test_results_all_roberta.csv"
    # )
    # OG_DATA = (
    #     "../outputs/nela_04-08-23-23_05_fine-tune_roberta-base_test_results_all_distilbert.csv"
    # )
    OG_DATA = (
        "../outputs/nela_04-08-23-23_05_fine-tune_roberta-base_test_results_all.csv"
    )
elif DATASET_NAME == "tfg":
    OG_DATA = "../outputs/tfg_19-07-23-06_12_fine-tune_distilbert-base-uncased_test_results_all.csv"
elif DATASET_NAME == "isot":
    OG_DATA = "../outputs/isot_19-07-23-20_25_fine-tune_distilbert-base-uncased_test_results_all.csv"
elif DATASET_NAME == "ti_cnn":
    OG_DATA = "../outputs/ticnn_19-07-23-22_05_fine-tune_distilbert-base-uncased_test_results_all.csv"

DATA_FOLDER = "../data/attack_files"
if NER == "default":
    if METHOD in ["salience", "freq"]:
        DATA_PATH = f"{DATA_FOLDER}/{DATASET_NAME}/{INSERT_POSITION}_{FALSE_CATEGORY}_inject_test_data_{METHOD}_with_preds.csv"
    elif METHOD in [
        "adverb",
        "negation",
        "sim_switch_freq",
        "sim_switch_salience",
        "perplexity_freq",
        "perplexity_salience",
        "stss",
    ]:
        DATA_PATH = f"{DATA_FOLDER}/{DATASET_NAME}/{FALSE_CATEGORY}_test_data_{METHOD}_with_preds.csv"


elif NER in ["noun", "verb", "ads"]:
    if METHOD in ["salience", "freq"]:
        DATA_PATH = f"{DATA_FOLDER}/{DATASET_NAME}/{INSERT_POSITION}_{FALSE_CATEGORY}_inject_test_data_{METHOD}_{NER}_with_preds.csv"
    elif METHOD in [
        "adverb",
        "negation",
        "sim_switch_freq",
        "sim_switch_salience",
        "perplexity_freq",
        "perplexity_salience",
        "stss",
    ]:
        DATA_PATH = f"{DATA_FOLDER}/{DATASET_NAME}/{FALSE_CATEGORY}_test_data_{METHOD}_{NER}_with_preds.csv"

OUTPUT_PATH = (
    f"../outputs/excel_outputs/{DATASET_NAME}/{NER}/results_{METHOD}_{MODE}.xlsx"
)
# OUTPUT_PATH = (
#     f"../outputs/excel_outputs/{DATASET_NAME}/{NER}/1_transfer_results_{METHOD}_{MODE}.xlsx"
# )
# OUTPUT_PATH = (
#     f"../outputs/excel_outputs/{DATASET_NAME}/{NER}/2_transfer_results_{METHOD}_{MODE}.xlsx"
# )
# OUTPUT_PATH = (
#     f"../outputs/excel_outputs/{DATASET_NAME}/{NER}/roberta_preds_results_{METHOD}_{MODE}.xlsx"
# )


def calculate_metrics(
    df,
    mode,
    when,
    original_accuracy=None,
    fn_override=None,
    tp_override=None,
    tn_override=None,
    fp_override=None,
):
    y_labels = np.array(df["true_labels"])
    if when == "before":
        y_preds = np.array(df["predicted_labels"])
    else:
        y_preds = np.array(df["predicted_label_after_attack"])

    cm = confusion_matrix(y_labels, y_preds, labels=[0, 1])
    if DATASET_NAME == "liar" and METHOD == "negation":
        if mode == "real":
            if fn_override is not None:
                cm[1][0] = fn_override
            if tp_override is not None:
                cm[1][1] = tp_override
        else:
            if tn_override is not None:
                cm[0][0] = tn_override
            if fp_override is not None:
                cm[0][1] = fp_override
    else:
        if mode == "real":
            if tn_override is not None:
                cm[0][0] = tn_override
            if fp_override is not None:
                cm[0][1] = fp_override
        else:
            if fn_override is not None:
                cm[1][0] = fn_override
            if tp_override is not None:
                cm[1][1] = tp_override

    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    precision_classwise = [tn / (tn + fn), precision]
    recall_classwise = [tn / (tn + fp), recall]

    if mode == "real":
        miss_rate = fn / (fn + tp) * 100
    else:
        miss_rate = fp / (fp + tn) * 100

    if original_accuracy:
        delta_accuracy = (original_accuracy - accuracy) * 100

    result = {
        "True negatives": tn,
        "False positives": fp,
        "False negatives": fn,
        "True positives": tp,
        "Accuracy": accuracy * 100,
        "Precision": precision * 100,
        "Recall": recall * 100,
        "F1-score": f1 * 100,
        "Precision - class 0": precision_classwise[0] * 100,
        "Precision - class 1": precision_classwise[1] * 100,
        "Recall - class 0": recall_classwise[0] * 100,
        "Recall - class 1": recall_classwise[1] * 100,
        "Miss Rate": miss_rate,
    }

    return result


def metric_2(df, mode="real"):
    if mode == "real":
        numerator = len(
            df[
                (df["predicted_label_after_attack"] == 0)
                & (df["predicted_labels"] == 1)
            ]
        )
        denominator = len(df[df["predicted_labels"] == 1])
        metric_2_val = numerator / denominator
        return metric_2_val
    else:
        numerator = len(
            df[
                (df["predicted_label_after_attack"] == 1)
                & (df["predicted_labels"] == 0)
            ]
        )
        denominator = len(df[df["predicted_labels"] == 0])
        metric_2_val = numerator / denominator
        return metric_2_val


def metric_3(df, mode="real"):
    if mode == "real":
        numerator = len(
            df[
                (df["predicted_label_after_attack"] == 0)
                & (df["predicted_labels"] == 0)
            ]
        )
        denominator = len(df[df["predicted_labels"] == 0])
        metric_3_val = numerator / denominator
        return metric_3_val
    else:
        numerator = len(
            df[
                (df["predicted_label_after_attack"] == 1)
                & (df["predicted_labels"] == 1)
            ]
        )
        denominator = len(df[df["predicted_labels"] == 1])
        metric_3_val = numerator / denominator
        return metric_3_val


def write_to_excel(results, results1, path, extra_metrics=None):
    if os.path.isfile(OUTPUT_PATH):
        df = pd.read_excel(OUTPUT_PATH, index_col=0)
        df1 = pd.DataFrame.from_dict(
            results1, orient="index", columns=[INSERT_POSITION]
        )
        df = df.join(df1)
    else:
        df = pd.DataFrame.from_dict(
            results, orient="index", columns=["Original Test Set"]
        )
        df1 = pd.DataFrame.from_dict(
            results1, orient="index", columns=[INSERT_POSITION]
        )
        df = df.join(df1)

    if extra_metrics is not None:
        for k, v in extra_metrics.items():
            df.loc[k, INSERT_POSITION] = v

    df.to_excel(path)


if __name__ == "__main__":
    # original metrics
    df = pd.read_csv(OG_DATA)
    results = calculate_metrics(df=df, mode=MODE, when="before")
    print("Original metrics: ", results)

    df1 = pd.read_csv(DATA_PATH)
    print(df1.head())

    if MODE == "real":
        if DATASET_NAME == "kaggle_fake_news":
            results_df1 = calculate_metrics(
                df1, mode=MODE, when="after", tn_override=1560, fp_override=2
            )
        elif DATASET_NAME == "liar":
            if METHOD == "negation":
                results_df1 = calculate_metrics(
                    df1, mode=MODE, when="after", fn_override=283, tp_override=270
                )
            else:
                results_df1 = calculate_metrics(
                    df1, mode=MODE, when="after", tn_override=283, fp_override=270
                )
        elif DATASET_NAME == "nela":
            results_df1 = calculate_metrics(
                df1, mode=MODE, when="after", tn_override=11973, fp_override=1067
            )
            # results_df1 = calculate_metrics(
            #     df1, mode=MODE, when="after", tn_override=12106, fp_override=934
            # )
            # results_df1 = calculate_metrics(
            #     df1, mode=MODE, when="after", tn_override=12150, fp_override=890
            # )
        elif DATASET_NAME == "tfg":
            results_df1 = calculate_metrics(
                df1, mode=MODE, when="after", tn_override=3710, fp_override=43
            )
        elif DATASET_NAME == "isot":
            results_df1 = calculate_metrics(
                df1, mode=MODE, when="after", tn_override=2597, fp_override=8
            )
        elif DATASET_NAME == "ti_cnn":
            results_df1 = calculate_metrics(
                df1, mode=MODE, when="after", tn_override=1725, fp_override=44
            )
    else:
        if DATASET_NAME == "kaggle_fake_news":
            results_df1 = calculate_metrics(
                df1, mode=MODE, when="after", fn_override=2, tp_override=1556
            )
        elif DATASET_NAME == "liar":
            if METHOD == "negation":
                results_df1 = calculate_metrics(
                    df1, mode=MODE, when="after", tn_override=172, fp_override=542
                )
            else:
                results_df1 = calculate_metrics(
                    df1, mode=MODE, when="after", fn_override=172, tp_override=542
                )
        elif DATASET_NAME == "nela":
            results_df1 = calculate_metrics(
                df1, mode=MODE, when="after", fn_override=987, tp_override=10452
            )
            # results_df1 = calculate_metrics(
            #     df1, mode=MODE, when="after", fn_override=1062, tp_override=9807
            # )
            # results_df1 = calculate_metrics(
            #     df1, mode=MODE, when="after", fn_override=890, tp_override=9979
            # )
        elif DATASET_NAME == "tfg":
            results_df1 = calculate_metrics(
                df1, mode=MODE, when="after", fn_override=35, tp_override=4329
            )
        elif DATASET_NAME == "isot":
            results_df1 = calculate_metrics(
                df1, mode=MODE, when="after", fn_override=3, tp_override=3189
            )
        elif DATASET_NAME == "ti_cnn":
            results_df1 = calculate_metrics(
                df1, mode=MODE, when="after", fn_override=18, tp_override=1216
            )

    delta_miss_rate = results_df1["Miss Rate"] - results["Miss Rate"]
    metric2 = metric_2(df1, mode=MODE)
    metric3 = metric_3(df1, mode=MODE)

    # New dictionary to store extra metrics
    extra_metrics = {
        "Delta Miss Rate": delta_miss_rate,
        "Metric 2": metric2,
        "Metric 3": metric3,
    }

    results_df1["Delta Miss Rate"] = delta_miss_rate
    results_df1["Metric 2"] = metric2
    results_df1["Metric 3"] = metric3

    print()
    print("New metrics: ", results_df1)

    write_to_excel(results, results_df1, OUTPUT_PATH, extra_metrics)
    print(f"Results written to excel file at {OUTPUT_PATH}")
