import ast
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import swifter
import torch
import yaml
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    logging,
)

logging.set_verbosity_error()

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

# Tokenizer and Model for word embeddings used for similarity switch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased").to(device)
perplexity_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
perplexity_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
model.eval()
perplexity_model.eval()

# Load config file
with open(CONFIG_PATH, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# get experiment config
experiment_config = config[EXPERIMENT]
print("Experiment config: ", experiment_config)

# Arguments - nela is only implemented for negation and adverb attack here
METHOD = experiment_config[1]["METHOD"][
    0
]  # "salience", "freq", "adverb", "negation", "sim_switch_freq", "sim_switch_salience", "perplexity_freq", "perplexity_salience"
DATASET_NAME = experiment_config[0]["DATASET_NAME"][
    0
]  # "kaggle_fake_news", "liar", "nela", "tfg", "isot"
INSERT_POSITION = experiment_config[2]["INSERT_POSITION"][0]  # "random", "head", "tail"
FALSE_CATEGORY = experiment_config[3]["FALSE_CATEGORY"][0]  # "fp", "fn"
BATCH_SIZE = 16  # for batching in parallel threads in sim_switch, perplexity

# Input path
if DATASET_NAME == "liar":
    DATA_PATH = "../outputs/liar_14-07-23-04_19_fine-tune_distilbert-base-uncased_test_results_all.csv"
elif DATASET_NAME == "kaggle_fake_news":
    DATA_PATH = "../outputs/kaggle_fake_news_14-07-23-02_35_fine-tune_distilbert-base-uncased_test_results_all.csv"
elif DATASET_NAME == "nela":
    DATA_PATH = "../outputs/nela_04-08-23-21_50_fine-tune_distilbert-base-uncased_test_results_all.csv"
    # DATA_PATH = (
    #     "../outputs/nela_04-08-23-21_50_fine-tune_distilbert-base-uncased_test_results_all_roberta.csv"
    # )
    # DATA_PATH = (
    #     "../outputs/nela_04-08-23-23_05_fine-tune_roberta-base_test_results_all_distilbert.csv"
    # )
    # DATA_PATH = (
    #     "../outputs/nela_04-08-23-23_05_fine-tune_roberta-base_test_results_all.csv"
    # )
elif DATASET_NAME == "tfg":
    DATA_PATH = "../outputs/tfg_19-07-23-06_12_fine-tune_distilbert-base-uncased_test_results_all.csv"
elif DATASET_NAME == "isot":
    DATA_PATH = "../outputs/isot_19-07-23-20_25_fine-tune_distilbert-base-uncased_test_results_all.csv"
elif DATASET_NAME == "ti_cnn":
    DATA_PATH = "../outputs/ticnn_19-07-23-22_05_fine-tune_distilbert-base-uncased_test_results_all.csv"

# Output path
DATA_FOLDER = "../data/attack_files"
if METHOD in ["salience", "freq"]:
    if NER == "default":
        OUTPUT_PATH = f"{DATA_FOLDER}/{DATASET_NAME}/{INSERT_POSITION}_{FALSE_CATEGORY}_inject_test_data_{METHOD}.csv"
    elif NER in ["noun", "verb", "ads"]:
        OUTPUT_PATH = f"{DATA_FOLDER}/{DATASET_NAME}/{INSERT_POSITION}_{FALSE_CATEGORY}_inject_test_data_{METHOD}_{NER}.csv"
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
        OUTPUT_PATH = (
            f"{DATA_FOLDER}/{DATASET_NAME}/{FALSE_CATEGORY}_test_data_{METHOD}.csv"
        )
    elif NER in ["noun", "verb", "ads"]:
        OUTPUT_PATH = f"{DATA_FOLDER}/{DATASET_NAME}/{FALSE_CATEGORY}_test_data_{METHOD}_{NER}.csv"

# SHAP outputs path
if METHOD in [
    "salience",
    "freq",
    "sim_switch_salience",
    "sim_switch_freq",
    "perplexity_salience",
    "perplexity_freq",
    "stss",
]:
    if NER == "default":
        if FALSE_CATEGORY == "fp":
            TOKENS_JSON_PATH = (
                f"../outputs/shap_outputs/{DATASET_NAME}_fake_attack_candidates.json"
            )
        elif FALSE_CATEGORY == "fn":
            TOKENS_JSON_PATH = (
                f"../outputs/shap_outputs/{DATASET_NAME}_real_attack_candidates.json"
            )
    elif NER in ["noun", "verb", "ads"]:
        if FALSE_CATEGORY == "fp":
            TOKENS_JSON_PATH = f"../outputs/shap_outputs/{DATASET_NAME}_fake_attack_candidates_{NER}.json"
        elif FALSE_CATEGORY == "fn":
            TOKENS_JSON_PATH = f"../outputs/shap_outputs/{DATASET_NAME}_real_attack_candidates_{NER}.json"


# Constants
LABEL_TO_FILTER = (
    0 if FALSE_CATEGORY == "fp" else 1
)  # Filter news items (0 for fp, 1 for fn)
NUM_WORDS_TO_INJECT = 10  # Number of words to inject in each article
RATIO_TO_MODIFY = 1.0  # Randomly select x% of the news items to modify
NUMBER_OF_CANDIDATE_TOKENS = 100  # chosen from TOKENS_JSON_PATH

# Adverbs
BOOSTER_DICT = [
    "absolutely",
    "amazingly",
    "awfully",
    "barely",
    "completely",
    "considerably",
    "decidedly",
    "deeply",
    "enormously",
    "entirely",
    "especially",
    "exceptionally",
    "exclusively",
    "extremely",
    "fully",
    "greatly",
    "hardly",
    "hella",
    "highly",
    "hugely",
    "incredibly",
    "intensely",
    "majorly",
    "overwhelmingly",
    "really",
    "remarkably",
    "substantially",
    "thoroughly",
    "totally",
    "tremendously",
    "unbelievably",
    "unusually",
    "utterly",
    "very",
]

# Negation dictionary
negate_dict = {
    "isn't": "is",
    "isn't": "is",
    "is not ": "is ",
    "is ": "is not ",
    "didn't": "did",
    "didn't": "did",
    "did not ": "did",
    "does not have": "has",
    "doesn't have": "has",
    "doesn't have": "has",
    "has ": "does not have ",
    "shouldn't": "should",
    "shouldn't": "should",
    "should not": "should",
    "should": "should not",
    "wouldn't": "would",
    "wouldn't": "would",
    "would not": "would",
    "would": "would not",
    "mustn't": "must",
    "mustn't": "must",
    "must not": "must",
    "must ": "must not ",
    "can't": "can",
    "can't": "can",
    "cannot": "can",
    " can ": " cannot ",
}

IRREGULAR_ES_VERB_ENDINGS = ["ss", "x", "ch", "sh", "o"]

# Cache for embeddings (stss)
embedding_cache = {}


# STSS
def switch_words_stss(row, candidate_tokens):
    sentence = row["content"]
    important_words = row["important_words"]

    if not important_words:
        return sentence

    switched_sentence = sentence
    all_words = important_words + candidate_tokens

    # Generate embeddings for important words and candidate tokens
    # Use cached embeddings if available
    all_embeddings = [embedding_cache.get(word) for word in all_words]
    words_to_encode = [
        word for word, embedding in zip(all_words, all_embeddings) if embedding is None
    ]

    if words_to_encode:
        new_embeddings = get_embedding_batch(words_to_encode)
        embedding_cache.update(
            {
                word: embedding
                for word, embedding in zip(words_to_encode, new_embeddings)
            }
        )
        all_embeddings = [embedding_cache[word] for word in all_words]

    important_words_embeddings = all_embeddings[: len(important_words)]
    candidate_tokens_embeddings = all_embeddings[len(important_words) :]

    # Calculate cosine similarities
    similarities = cosine_similarity(
        candidate_tokens_embeddings, important_words_embeddings
    )

    assert similarities.shape[0] == len(candidate_tokens)
    assert similarities.shape[1] == len(important_words)

    # iterate over the important words
    replacements = []
    for idx_sim, word in enumerate(important_words):
        # get the most similar candidate token
        similar_word_indices = np.argsort(similarities[:, idx_sim])[::-1]
        similar_word_index = similar_word_indices[0]
        candidate_token = candidate_tokens[similar_word_index]
        # switch the word
        if word.lower() != candidate_token.lower():
            replacements.append((word, candidate_token))
            switched_sentence = re.sub(
                r"\b" + re.escape(word) + r"\b", candidate_token, switched_sentence
            )
        elif word.lower() == candidate_token.lower():
            # find the next most similar candidate token which is not the same
            # as the word
            for similar_word_index in similar_word_indices:
                candidate_token = candidate_tokens[similar_word_index]
                if word.lower() != candidate_token.lower():
                    replacements.append((word, candidate_token))
                    switched_sentence = re.sub(
                        r"\b" + re.escape(word) + r"\b",
                        candidate_token,
                        switched_sentence,
                    )
                    break

    assert len(replacements) == len(important_words)
    return replacements, switched_sentence


# Measure perplexity of a sentence
def measure_perplexity(sentence):
    with torch.no_grad():
        inputs = perplexity_tokenizer(sentence, return_tensors="pt").to(device)
        outputs = perplexity_model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    return torch.exp(loss).item()


# Inject a token at all possible positions and keep the one with lowest perplexity
def inject_word(sentence, word_to_inject):
    tokens = sentence.split()
    min_perplexity = float("inf")
    best_sentence = sentence
    for i in range(len(tokens) + 1):
        new_tokens = tokens[:i] + [word_to_inject] + tokens[i:]
        new_sentence = " ".join(new_tokens)
        perplexity = measure_perplexity(new_sentence)
        if perplexity < min_perplexity:
            min_perplexity = perplexity
            best_sentence = new_sentence
    return best_sentence


# Function to modify a single article
def modify_article(article):
    sentences = article.split(".")
    with ThreadPoolExecutor() as executor:
        future_to_sentence = {
            executor.submit(
                inject_word, sentence, random.choice(tokens_to_inject)
            ): sentence
            for sentence in sentences
        }
        for future in as_completed(future_to_sentence):
            sentence = future_to_sentence[future]
            try:
                data = future.result()
            except Exception as exc:
                print("%r generated an exception: %s" % (sentence, exc))
    return ". ".join([future.result() for future in as_completed(future_to_sentence)])


def modify_articles_batch(df_batch):
    df_batch = df_batch.copy()
    for idx, row in df_batch.iterrows():
        article = row["content"]
        if isinstance(article, str):  # Check if the article is a string
            df_batch.loc[idx, "modified_content"] = modify_article(article)
    return df_batch


# Function to get BERT embeddings
def get_embedding_batch(words):
    input_ids = [tokenizer.encode(word, add_special_tokens=True) for word in words]
    max_len = max([len(i) for i in input_ids])
    padded = torch.tensor([i + [0] * (max_len - len(i)) for i in input_ids]).to(device)
    attention_mask = torch.where(padded != 0, 1, 0).to(device)
    with torch.no_grad():
        last_hidden_states = model(padded, attention_mask=attention_mask)
    features = last_hidden_states[0][:, 0, :].cpu().numpy()
    return features


def switch_words(sentence, words_to_switch):
    if not isinstance(sentence, str):
        raise ValueError(f"Expected string, got {type(sentence)}")

    tokens = word_tokenize(sentence)
    switched_sentence = sentence

    try:
        # Generate embeddings for all tokens and candidate words at once
        all_embeddings = get_embedding_batch(tokens + words_to_switch)
        tokens_embeddings = all_embeddings[: len(tokens)]
        words_to_switch_embeddings = all_embeddings[len(tokens) :]

        assert len(tokens) > 0, "Tokens is empty"
        assert len(words_to_switch) > 0, "Words_to_switch is empty"
        assert all_embeddings.shape[0] > 0, "All embeddings is empty"
        assert tokens_embeddings.shape[0] > 0, "Tokens_embeddings is empty"
        assert (
            words_to_switch_embeddings.shape[0] > 0
        ), "Words_to_switch_embeddings is empty"

        # Calculate cosine similarities in a vectorized way
        similarities = cosine_similarity(words_to_switch_embeddings, tokens_embeddings)

    except Exception as e:
        return switched_sentence
    # Pair tokens with their similarity scores for each candidate
    token_similarity_pairs = []
    for idx, candidate in enumerate(words_to_switch):
        token_similarity_pairs.extend(
            [(token, candidate, sim) for token, sim in zip(tokens, similarities[idx])]
        )

    # Sort pairs by similarity
    token_similarity_pairs.sort(key=lambda x: x[2], reverse=True)

    count = 0  # Initialize count here to limit per sentence switches

    # Pick the token-candidate pairs with the highest similarity
    for token, candidate, sim in token_similarity_pairs:
        if count >= 10:  # Stop after switching twenty words
            break
        if 0.5 < sim < 0.9:  # Threshold for similarity
            if (
                token.lower() != candidate.lower()
            ):  # Avoid replacing with the same token
                switched_sentence = re.sub(
                    r"\b" + re.escape(token) + r"\b", candidate, switched_sentence
                )
                count += 1

    return switched_sentence


def switch_words_in_batch(batch, words_to_switch):
    # Now each item in the iterable is itself an iterable (a tuple)
    return list(executor.map(switch_words, batch, [words_to_switch] * len(batch)))


# Negation attack
def negate(sentence):
    for key in negate_dict.keys():
        if sentence.find(key) > -1:
            return sentence.replace(key, negate_dict[key])
    doesnt_regex = r"(doesn't|doesn\\'t|does not) (?P<verb>\w+)"
    if re.search(doesnt_regex, sentence):
        return re.sub(doesnt_regex, replace_doesnt, sentence, 1)
    return sentence


def __is_consonant(letter):
    return letter not in ["a", "e", "i", "o", "u", "y"]


def replace_doesnt(matchobj):
    verb = matchobj.group(2)
    if verb.endswith("y") and __is_consonant(verb[-2]):
        return "{0}ies".format(verb[0:-1])
    for ending in IRREGULAR_ES_VERB_ENDINGS:
        if verb.endswith(ending):
            return "{0}es".format(verb)
    return "{0}s".format(verb)


# Adverb intensity attack
def reduce_intensity(sentence):
    return " ".join([w for w in sentence.split() if w.lower() not in BOOSTER_DICT])


# Injection attack
def inject_words(sentence, words_to_inject, num_words_to_inject, mode):
    if mode == "random":
        # print("Random insertion ...")
        return inject_words_random(sentence, words_to_inject, num_words_to_inject)
    elif mode == "head":
        # print("Head insertion ...")
        return inject_words_head(sentence, words_to_inject, num_words_to_inject)
    elif mode == "tail":
        # print("Tail insertion ...")
        return inject_words_tail(sentence, words_to_inject, num_words_to_inject)


# inject at random locations
def inject_words_random(sentence, words_to_inject, num_words_to_inject):
    tokens = sentence.split()
    words_to_inject = random.sample(words_to_inject, num_words_to_inject)
    for word in words_to_inject:
        position = random.randint(0, len(tokens))
        tokens.insert(position, word)
    return " ".join(tokens)


# inject at head
def inject_words_head(sentence, words_to_inject, num_words_to_inject):
    tokens = sentence.split()
    words_to_inject = random.sample(words_to_inject, num_words_to_inject)
    for word in words_to_inject:
        tokens.insert(0, word)  # Insert at the beginning of the sentence
    return " ".join(tokens)


# inject at tail
def inject_words_tail(sentence, words_to_inject, num_words_to_inject):
    tokens = sentence.split()
    words_to_inject = random.sample(words_to_inject, num_words_to_inject)
    for word in words_to_inject:
        tokens.append(word)  # Append at the end of the sentence
    return " ".join(tokens)


def preprocess_text(text):
    text = re.sub(r"#", "", text.lower())
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    text = " ".join(tokens)
    return text


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    df = df[df["content"].apply(lambda x: isinstance(x, str))]
    df_label_filtered = df[df["true_labels"] == LABEL_TO_FILTER]
    num_items_to_modify = int(RATIO_TO_MODIFY * len(df_label_filtered))
    items_to_modify = df_label_filtered.sample(num_items_to_modify)

    if METHOD in ["sim_switch_freq", "sim_switch_salience"]:
        with open(TOKENS_JSON_PATH, "r") as f:
            tokens = json.load(f)
        switching_candidates = sorted(tokens, key=tokens.get, reverse=True)[
            : min(NUMBER_OF_CANDIDATE_TOKENS, len(tokens.keys()))
        ]
        with ThreadPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
            switched_content = []
            for i in tqdm(range(0, len(items_to_modify), BATCH_SIZE)):
                batch = items_to_modify["content"][i : i + BATCH_SIZE].apply(str)
                switched_content.extend(
                    switch_words_in_batch(batch, switching_candidates)
                )
        items_to_modify["modified_content"] = switched_content
        modified_df = items_to_modify
        modified_df.to_csv(OUTPUT_PATH, index=False)
        print("Saved to: ", OUTPUT_PATH)

    elif METHOD in ["stss"]:
        with open(TOKENS_JSON_PATH, "r") as f:
            tokens = json.load(f)

        words_to_switch = list(tokens.keys())

        # convert str to list
        items_to_modify["important_words"] = items_to_modify["important_words"].apply(
            lambda x: ast.literal_eval(x)
        )

        items_to_modify["len_important_words"] = items_to_modify[
            "important_words"
        ].apply(lambda x: len(x))
        sns.displot(items_to_modify["len_important_words"], kde=False)
        plt.savefig(f"important_words_dist_{DATASET_NAME}_{FALSE_CATEGORY}.png")

        result = items_to_modify.swifter.apply(
            lambda row: switch_words_stss(row, words_to_switch), axis=1
        )
        items_to_modify["replacements"], items_to_modify["modified_content"] = zip(
            *result
        )

        modified_df = items_to_modify
        print(modified_df.head())
        modified_df.to_csv(OUTPUT_PATH, index=False)
        print(f"Saved to: {OUTPUT_PATH}")

    elif "perplexity" in METHOD:
        with open(TOKENS_JSON_PATH, "r") as f:
            tokens = json.load(f)
        tokens_to_inject = list(tokens.keys())[
            : min(NUMBER_OF_CANDIDATE_TOKENS, len(tokens.keys()))
        ]
        with ThreadPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
            modified_content = []
            for i in tqdm(range(0, len(items_to_modify), BATCH_SIZE)):
                batch = items_to_modify["content"][i : i + BATCH_SIZE].apply(str)
                df_batch = pd.DataFrame(batch)
                modified_content.extend(
                    modify_articles_batch(df_batch)["modified_content"]
                )
        items_to_modify["modified_content"] = modified_content
        modified_df = items_to_modify
        modified_df.to_csv(OUTPUT_PATH, index=False)
        print("Saved to: ", OUTPUT_PATH)

    else:
        for idx, row in items_to_modify.iterrows():
            original_text = row["content"]
            if not isinstance(original_text, str):
                continue

            if METHOD in ["salience", "freq"]:
                with open(TOKENS_JSON_PATH, "r") as f:
                    tokens = json.load(f)

                tokens_to_inject = list(tokens.keys())[
                    : min(NUMBER_OF_CANDIDATE_TOKENS, len(tokens.keys()))
                ]

                modified_text = inject_words(
                    original_text,
                    tokens_to_inject,
                    num_words_to_inject=min(NUM_WORDS_TO_INJECT, len(tokens_to_inject)),
                    mode=INSERT_POSITION,
                )
                df.loc[idx, "modified_content"] = modified_text
            elif METHOD == "adverb":
                original_text = original_text.lower().replace("’", "'")
                polar_word_present = any(
                    w in original_text.split() for w in BOOSTER_DICT
                )
                if polar_word_present:
                    modified_text = reduce_intensity(original_text)
                    df.loc[idx, "modified_content"] = modified_text
                else:
                    df.loc[idx, "modified_content"] = original_text
            elif METHOD == "negation":
                original_text = original_text.lower().replace("’", "'")
                modified_text = negate(original_text)
                df.loc[idx, "modified_content"] = modified_text
                if DATASET_NAME == "liar":
                    # FLIP THE TRUE LABEL
                    df.loc[idx, "true_labels"] = 1 - df.loc[idx, "true_labels"]

        modified_df = df.loc[items_to_modify.index.values]
        print(modified_df.head())
        modified_df.to_csv(OUTPUT_PATH, index=False)
