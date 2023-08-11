import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# Set seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# demo tokenization settings
MAX_SEQUENCE_LENGTH = 512
BATCH_SIZE = 32
NAME = "distilbert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "../data/"
NUM_SAMPLES = -1  # -1 to run on all data

# flores and hao settings for LIAR 2-class
new_label_dict = {
    "true": 1,
    "mostly-true": 1,
    "half-true": 1,
    "barely-true": 0,
    "pants-fire": 0,
    "false": 0,
}


class LIARDataset:
    # set num_samples = -1 to run on all data
    def __init__(
        self, path=f"{DATA_PATH}/liar", num_samples=NUM_SAMPLES, split="train"
    ):
        self.path = path
        self.num_samples = num_samples
        self.split = split
        self.df = pd.read_csv(
            f"{path}/{split}.tsv", sep="\t", header=None, usecols=[1, 2]
        )
        self.df.columns = ["label", "content"]
        self.df = self.df[["content", "label"]]
        self.df["label"] = self.df["label"].map(new_label_dict)

        if self.num_samples != -1:
            self.df = self.df[: self.num_samples]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        return {
            "content": sample["content"],
            "label": sample["label"],
        }


class KaggleFakeNewsDataset:
    def __init__(
        self,
        path=f"{DATA_PATH}/kaggle_fake_news",
        num_samples=NUM_SAMPLES,
        split="train",
    ):
        self.path = path
        self.num_samples = num_samples
        self.split = split
        self.df = pd.read_csv(f"{path}/{split}.csv", usecols=[3, 4])
        self.df.columns = ["content", "label"]
        self.df["content"] = self.df["content"].map(str)
        self.df["label"] = self.df["label"].map(int)
        self.df = self.df.dropna()
        self.df = self.df[self.df["content"] != ""]
        # flip labels as originally [1 is unreliable and 0 is reliable]
        # but for us [1 is real and 0 is fake]
        self.df["label"] = self.df["label"].map({0: 1, 1: 0})

        if num_samples != -1:
            self.df = self.df[:num_samples]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        return {
            "content": sample["content"],
            "label": sample["label"],
        }


class TFGDataset:
    def __init__(self, path=f"{DATA_PATH}/tfg", num_samples=NUM_SAMPLES, split="train"):
        self.path = path
        self.num_samples = num_samples
        self.split = split
        self.df = pd.read_csv(f"{path}/{split}.csv", sep=";")
        self.df = self.df[["text", "label"]]
        self.df.columns = ["content", "label"]
        self.df = self.df[["content", "label"]]
        if self.num_samples != -1:
            self.df = self.df[: self.num_samples]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        return {
            "content": sample["content"],
            "label": sample["label"],
        }


class ISOTDataset:
    def __init__(
        self, path=f"{DATA_PATH}/isot", num_samples=NUM_SAMPLES, split="train"
    ):
        self.path = path
        self.num_samples = num_samples
        self.split = split
        self.df = pd.read_csv(f"{path}/{split}.csv")
        self.df = self.df[["content", "label"]]
        if self.num_samples != -1:
            self.df = self.df[: self.num_samples]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        return {
            "content": sample["content"],
            "label": sample["label"],
        }


class TICNNDataset:
    def __init__(
        self, path=f"{DATA_PATH}/ti_cnn", num_samples=NUM_SAMPLES, split="train"
    ):
        self.path = path
        self.num_samples = num_samples
        self.split = split
        self.df = pd.read_csv(f"{path}/{split}.csv")
        self.df = self.df[["content", "label"]]
        if self.num_samples != -1:
            self.df = self.df[: self.num_samples]
        # map labels to int and content to str
        self.df["label"] = self.df["label"].map(int)
        self.df["content"] = self.df["content"].map(str)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        return {
            "content": sample["content"],
            "label": sample["label"],
        }


# different from the other classes. Uses older style of code.
class NELADataset:
    def __init__(
        self, path=f"{DATA_PATH}/nela", num_samples=NUM_SAMPLES, split="train"
    ):
        self.path = path
        self.num_samples = num_samples
        self.split = split
        self.df = pd.read_csv(f"{path}/{split}.csv")
        self.df = self.df[["content", "label"]]
        if self.num_samples != -1:
            self.df = self.df[: self.num_samples]
        self.df = self.df.dropna(subset=["content"])
        self.df = self.df[~self.df["content"].str.contains("403 forbidden")]
        # map labels to int and content to str
        self.df["label"] = self.df["label"].map(int)
        self.df["content"] = self.df["content"].map(str)
        print(f"{split} dataset size: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        return {
            "content": sample["content"],
            "label": sample["label"],
            "leaning": sample["leaning"],
        }


class LIARDataProcessor:
    def __init__(
        self, name=NAME, max_seq_length=MAX_SEQUENCE_LENGTH, batch_size=BATCH_SIZE
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def tokenize(self, sentences, padding="max_length"):
        encoded = self.tokenizer.batch_encode_plus(
            sentences, max_length=self.max_seq_length, truncation=True, padding=padding
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        return torch.tensor(input_ids).to(device), torch.tensor(attention_mask).to(
            device
        )

    def process(self, data_path=f"{DATA_PATH}/liar", num_samples=NUM_SAMPLES):
        train_data = LIARDataset(path=data_path, num_samples=num_samples, split="train")
        test_data = LIARDataset(path=data_path, num_samples=num_samples, split="test")
        valid_data = LIARDataset(path=data_path, num_samples=num_samples, split="valid")
        self.dataset = {
            "train": train_data,
            "test": test_data,
            "valid": valid_data,
        }
        X_train, y_train = (
            train_data.df["content"].values.tolist(),
            train_data.df["label"].values.tolist(),
        )
        X_test, y_test = (
            test_data.df["content"].values.tolist(),
            test_data.df["label"].values.tolist(),
        )
        X_valid, y_valid = (
            valid_data.df["content"].values.tolist(),
            valid_data.df["label"].values.tolist(),
        )
        train_dataset = torch.utils.data.TensorDataset(
            *self.tokenize(X_train), torch.tensor(y_train).to(device)
        )
        test_dataset = torch.utils.data.TensorDataset(
            *self.tokenize(X_test), torch.tensor(y_test).to(device)
        )
        valid_dataset = torch.utils.data.TensorDataset(
            *self.tokenize(X_valid), torch.tensor(y_valid).to(device)
        )
        self.dataloaders = {
            "train": torch.utils.data.DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True
            ),
            "test": torch.utils.data.DataLoader(
                test_dataset, batch_size=self.batch_size, shuffle=True
            ),
            "valid": torch.utils.data.DataLoader(
                valid_dataset, batch_size=self.batch_size, shuffle=True
            ),
        }
        return self.dataloaders


class KaggleFakeNewsDataProcessor:
    def __init__(
        self, name=NAME, max_seq_length=MAX_SEQUENCE_LENGTH, batch_size=BATCH_SIZE
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def tokenize(self, sentences, padding="max_length"):
        encoded = self.tokenizer.batch_encode_plus(
            sentences, max_length=self.max_seq_length, truncation=True, padding=padding
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        return torch.tensor(input_ids).to(device), torch.tensor(attention_mask).to(
            device
        )

    def process(
        self, data_path=f"{DATA_PATH}/kaggle_fake_news", num_samples=NUM_SAMPLES
    ):
        train_data = KaggleFakeNewsDataset(
            path=data_path, num_samples=num_samples, split="train"
        )
        test_data = KaggleFakeNewsDataset(
            path=data_path, num_samples=num_samples, split="test"
        )
        valid_data = KaggleFakeNewsDataset(
            path=data_path, num_samples=num_samples, split="valid"
        )
        self.dataset = {
            "train": train_data,
            "test": test_data,
            "valid": valid_data,
        }
        X_train, y_train = (
            train_data.df["content"].values.tolist(),
            train_data.df["label"].values.tolist(),
        )
        X_test, y_test = (
            test_data.df["content"].values.tolist(),
            test_data.df["label"].values.tolist(),
        )
        X_valid, y_valid = (
            valid_data.df["content"].values.tolist(),
            valid_data.df["label"].values.tolist(),
        )
        train_dataset = torch.utils.data.TensorDataset(
            *self.tokenize(X_train), torch.tensor(y_train).to(device)
        )
        test_dataset = torch.utils.data.TensorDataset(
            *self.tokenize(X_test), torch.tensor(y_test).to(device)
        )
        valid_dataset = torch.utils.data.TensorDataset(
            *self.tokenize(X_valid), torch.tensor(y_valid).to(device)
        )
        self.dataloaders = {
            "train": torch.utils.data.DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True
            ),
            "test": torch.utils.data.DataLoader(
                test_dataset, batch_size=self.batch_size, shuffle=True
            ),
            "valid": torch.utils.data.DataLoader(
                valid_dataset, batch_size=self.batch_size, shuffle=True
            ),
        }
        return self.dataloaders


class TFGDataProcessor:
    def __init__(
        self, name=NAME, max_seq_length=MAX_SEQUENCE_LENGTH, batch_size=BATCH_SIZE
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def tokenize(self, sentences, padding="max_length"):
        encoded = self.tokenizer.batch_encode_plus(
            sentences, max_length=self.max_seq_length, truncation=True, padding=padding
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        return torch.tensor(input_ids).to(device), torch.tensor(attention_mask).to(
            device
        )

    def process(self, data_path=f"{DATA_PATH}/tfg", num_samples=NUM_SAMPLES):
        train_data = TFGDataset(path=data_path, num_samples=num_samples, split="train")
        test_data = TFGDataset(path=data_path, num_samples=num_samples, split="test")
        valid_data = TFGDataset(path=data_path, num_samples=num_samples, split="valid")
        self.dataset = {
            "train": train_data,
            "test": test_data,
            "valid": valid_data,
        }
        X_train, y_train = (
            train_data.df["content"].values.tolist(),
            train_data.df["label"].values.tolist(),
        )
        X_test, y_test = (
            test_data.df["content"].values.tolist(),
            test_data.df["label"].values.tolist(),
        )
        X_valid, y_valid = (
            valid_data.df["content"].values.tolist(),
            valid_data.df["label"].values.tolist(),
        )
        train_dataset = torch.utils.data.TensorDataset(
            *self.tokenize(X_train), torch.tensor(y_train).to(device)
        )
        test_dataset = torch.utils.data.TensorDataset(
            *self.tokenize(X_test), torch.tensor(y_test).to(device)
        )
        valid_dataset = torch.utils.data.TensorDataset(
            *self.tokenize(X_valid), torch.tensor(y_valid).to(device)
        )
        self.dataloaders = {
            "train": torch.utils.data.DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True
            ),
            "test": torch.utils.data.DataLoader(
                test_dataset, batch_size=self.batch_size, shuffle=True
            ),
            "valid": torch.utils.data.DataLoader(
                valid_dataset, batch_size=self.batch_size, shuffle=True
            ),
        }
        return self.dataloaders


class ISOTDataProcessor:
    def __init__(
        self, name=NAME, max_seq_length=MAX_SEQUENCE_LENGTH, batch_size=BATCH_SIZE
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def tokenize(self, sentences, padding="max_length"):
        encoded = self.tokenizer.batch_encode_plus(
            sentences, max_length=self.max_seq_length, truncation=True, padding=padding
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        return torch.tensor(input_ids).to(device), torch.tensor(attention_mask).to(
            device
        )

    def process(self, data_path=f"{DATA_PATH}/isot", num_samples=NUM_SAMPLES):
        train_data = ISOTDataset(path=data_path, num_samples=num_samples, split="train")
        test_data = ISOTDataset(path=data_path, num_samples=num_samples, split="test")
        valid_data = ISOTDataset(path=data_path, num_samples=num_samples, split="valid")
        self.dataset = {
            "train": train_data,
            "test": test_data,
            "valid": valid_data,
        }
        X_train, y_train = (
            train_data.df["content"].values.tolist(),
            train_data.df["label"].values.tolist(),
        )
        X_test, y_test = (
            test_data.df["content"].values.tolist(),
            test_data.df["label"].values.tolist(),
        )
        X_valid, y_valid = (
            valid_data.df["content"].values.tolist(),
            valid_data.df["label"].values.tolist(),
        )
        train_dataset = torch.utils.data.TensorDataset(
            *self.tokenize(X_train), torch.tensor(y_train).to(device)
        )
        test_dataset = torch.utils.data.TensorDataset(
            *self.tokenize(X_test), torch.tensor(y_test).to(device)
        )
        valid_dataset = torch.utils.data.TensorDataset(
            *self.tokenize(X_valid), torch.tensor(y_valid).to(device)
        )
        self.dataloaders = {
            "train": torch.utils.data.DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True
            ),
            "test": torch.utils.data.DataLoader(
                test_dataset, batch_size=self.batch_size, shuffle=True
            ),
            "valid": torch.utils.data.DataLoader(
                valid_dataset, batch_size=self.batch_size, shuffle=True
            ),
        }
        return self.dataloaders


class TICNNDataProcessor:
    def __init__(
        self, name=NAME, max_seq_length=MAX_SEQUENCE_LENGTH, batch_size=BATCH_SIZE
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def tokenize(self, sentences, padding="max_length"):
        encoded = self.tokenizer.batch_encode_plus(
            sentences, max_length=self.max_seq_length, truncation=True, padding=padding
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        return torch.tensor(input_ids).to(device), torch.tensor(attention_mask).to(
            device
        )

    def process(self, data_path=f"{DATA_PATH}/ti_cnn", num_samples=NUM_SAMPLES):
        train_data = TICNNDataset(
            path=data_path, num_samples=num_samples, split="train"
        )
        test_data = TICNNDataset(path=data_path, num_samples=num_samples, split="test")
        valid_data = TICNNDataset(
            path=data_path, num_samples=num_samples, split="valid"
        )
        self.dataset = {
            "train": train_data,
            "test": test_data,
            "valid": valid_data,
        }
        X_train, y_train = (
            train_data.df["content"].values.tolist(),
            train_data.df["label"].values.tolist(),
        )
        X_test, y_test = (
            test_data.df["content"].values.tolist(),
            test_data.df["label"].values.tolist(),
        )
        X_valid, y_valid = (
            valid_data.df["content"].values.tolist(),
            valid_data.df["label"].values.tolist(),
        )
        train_dataset = torch.utils.data.TensorDataset(
            *self.tokenize(X_train), torch.tensor(y_train).to(device)
        )
        test_dataset = torch.utils.data.TensorDataset(
            *self.tokenize(X_test), torch.tensor(y_test).to(device)
        )
        valid_dataset = torch.utils.data.TensorDataset(
            *self.tokenize(X_valid), torch.tensor(y_valid).to(device)
        )
        self.dataloaders = {
            "train": torch.utils.data.DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True
            ),
            "test": torch.utils.data.DataLoader(
                test_dataset, batch_size=self.batch_size, shuffle=True
            ),
            "valid": torch.utils.data.DataLoader(
                valid_dataset, batch_size=self.batch_size, shuffle=True
            ),
        }
        return self.dataloaders


class NELADataProcessor:
    def __init__(
        self, name=NAME, max_seq_length=MAX_SEQUENCE_LENGTH, batch_size=BATCH_SIZE
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def tokenize(self, sentences, padding="max_length"):
        encoded = self.tokenizer.batch_encode_plus(
            sentences, max_length=self.max_seq_length, truncation=True, padding=padding
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        return torch.tensor(input_ids).to(device), torch.tensor(attention_mask).to(
            device
        )

    def process(self, data_path=f"{DATA_PATH}/nela", num_samples=NUM_SAMPLES):
        train_data = NELADataset(path=data_path, num_samples=num_samples, split="train")
        test_data = NELADataset(path=data_path, num_samples=num_samples, split="test")
        valid_data = NELADataset(path=data_path, num_samples=num_samples, split="valid")
        self.dataset = {
            "train": train_data,
            "test": test_data,
            "valid": valid_data,
        }
        X_train, y_train = (
            train_data.df["content"].values.tolist(),
            train_data.df["label"].values.tolist(),
        )
        X_test, y_test = (
            test_data.df["content"].values.tolist(),
            test_data.df["label"].values.tolist(),
        )
        X_valid, y_valid = (
            valid_data.df["content"].values.tolist(),
            valid_data.df["label"].values.tolist(),
        )
        train_dataset = torch.utils.data.TensorDataset(
            *self.tokenize(X_train), torch.tensor(y_train).to(device)
        )
        test_dataset = torch.utils.data.TensorDataset(
            *self.tokenize(X_test), torch.tensor(y_test).to(device)
        )
        valid_dataset = torch.utils.data.TensorDataset(
            *self.tokenize(X_valid), torch.tensor(y_valid).to(device)
        )
        self.dataloaders = {
            "train": torch.utils.data.DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True
            ),
            "test": torch.utils.data.DataLoader(
                test_dataset, batch_size=self.batch_size, shuffle=True
            ),
            "valid": torch.utils.data.DataLoader(
                valid_dataset, batch_size=self.batch_size, shuffle=True
            ),
        }
        return self.dataloaders


if __name__ == "__main__":
    data_processor = NELADataProcessor()
    dataloaders = data_processor.process()
    input_ids, attention_mask, labels = next(iter(dataloaders["train"]))

    # ticnn_data_processor = TICNNDataProcessor()
    # dataloaders = ticnn_data_processor.process()
    # input_ids, attention_mask, labels = next(iter(dataloaders["train"]))

    # isot_data_processor = ISOTDataProcessor()
    # dataloaders = isot_data_processor.process()
    # input_ids, attention_mask, labels = next(iter(dataloaders["train"]))

    # tfg_data_processor = TFGDataProcessor()
    # dataloaders = tfg_data_processor.process()
    # input_ids, attention_mask, labels = next(iter(dataloaders["train"]))

    # kaggle_fake_news_data_processor = KaggleFakeNewsDataProcessor()
    # dataloaders = kaggle_fake_news_data_processor.process()
    # input_ids, attention_mask, labels = next(iter(dataloaders["train"]))

    # liar_data_processor = LIARDataProcessor()
    # dataloaders = liar_data_processor.process()
    # input_ids, attention_mask, labels = next(iter(dataloaders["train"]))
