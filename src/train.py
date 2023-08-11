import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from bert_model import BertClassifier
from data_load import (
    ISOTDataProcessor,
    KaggleFakeNewsDataProcessor,
    LIARDataProcessor,
    NELADataProcessor,
    TFGDataProcessor,
    TICNNDataProcessor,
)

# random seed settings
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# paths for IO
MODELS_PATH = "../models"
OUTPUTS_PATH = "../outputs"
LOGS_PATH = "./logs"

# dataset name
DATASET_NAME = "nela"  # "liar", "kaggle_fake_news", "tfg", "isot", "ti_cnn"

# hyperparameters
NUM_EPOCHS = 10 if DATASET_NAME == "liar" else 4
LEARNING_RATE = 5e-6 if DATASET_NAME == "liar" else 5e-5
EPS = 1e-8

# tokenizer and model settings
NAME = "distilbert-base-uncased"
MAX_SEQUENCE_LENGTH = 128 if DATASET_NAME == "liar" else 512
BATCH_SIZE = 64 if DATASET_NAME == "liar" else 32
NUM_SAMPLES = -1
MODE = "fine-tune"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(
    model,
    train_dataloader,
    valid_dataloader,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    models_path,
    save_intermediate=False,
):
    model_name = f"{NAME.replace('/', '-')}_model"
    writer = SummaryWriter(log_dir=LOGS_PATH)
    for epoch in tqdm(range(num_epochs), desc="Epochs", unit="epoch", total=num_epochs):
        train_loss, valid_loss = 0, 0
        train_acc, valid_acc = 0, 0

        model.train()

        for i, data in tqdm(
            enumerate(train_dataloader),
            desc="Batches",
            unit="batch",
            total=len(train_dataloader),
        ):
            input_ids, attention_mask, labels = data

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_acc += (predicted == labels).sum().item()

        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader.dataset)

        writer.add_scalar("Train Loss", train_loss, epoch)
        writer.add_scalar("Train Accuracy", train_acc, epoch)

        with torch.no_grad():
            model.eval()
            for i, data in enumerate(valid_dataloader):
                input_ids, attention_mask, labels = data
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                valid_acc += (predicted == labels).sum().item()

            valid_loss /= len(valid_dataloader)
            valid_acc /= len(valid_dataloader.dataset)

            writer.add_scalar("Validation Loss", valid_loss, epoch)
            writer.add_scalar("Validation Accuracy", valid_acc, epoch)

        print(
            f"Epoch: {epoch+1} | "
            f"Train Loss: {train_loss:.3f} | "
            f"Train Accuracy: {train_acc*100:.2f}% | "
            f"Validation Loss: {valid_loss:.3f} | "
            f"Validation Accuracy: {valid_acc*100:.2f}%"
        )

        if save_intermediate:
            # save intermediate models after each epoch if needed
            filename = DATASET_NAME
            filename += datetime.now().strftime(
                f"_%d-%m-%y-%H_%M_{MODE}_{model_name}_epoch{epoch}.pt"
            )
            torch.save(model.state_dict(), f"{models_path}/{filename}")

    filename = DATASET_NAME
    filename += datetime.now().strftime(f"_%d-%m-%y-%H_%M_{MODE}_{model_name}_final.pt")
    torch.save(model.state_dict(), f"{models_path}/{filename}")
    writer.close()


def evaluate_model(model, dataloader, split):
    model.eval()
    test_acc = 0
    batch_count = 0
    all_texts, all_labels, all_preds = [], [], []
    tokenizer = AutoTokenizer.from_pretrained(NAME)
    for i, data in enumerate(dataloader):
        input_ids, attention_mask, labels = data
        all_labels.append(labels.cpu().numpy())
        all_texts.append(tokenizer.batch_decode(input_ids, skip_special_tokens=True))

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu().numpy())
            test_acc += (preds == labels).sum().item()
            batch_count += 1

    test_acc /= batch_count * dataloader.batch_size
    if split == "test":
        print(f"Test Accuracy: {test_acc*100:.2f}% \n")
    elif split == "valid":
        print(f"Validation Accuracy: {test_acc*100:.2f}% \n")
    return all_texts, all_labels, all_preds


def save_test_as_dataframe(all_texts, all_labels, all_preds, split):
    labels_df = pd.DataFrame(
        {
            "content": [text for batch in all_texts for text in batch],
            "true_labels": [label for batch in all_labels for label in batch],
            "predicted_labels": [pred for batch in all_preds for pred in batch],
        }
    )
    print(labels_df.head())
    if NUM_SAMPLES == -1:
        sample_size = "all"
    else:
        sample_size = NUM_SAMPLES

    filename = DATASET_NAME
    if split == "test":
        filename += datetime.now().strftime(
            f"_%d-%m-%y-%H_%M_{MODE}_{NAME.replace('/', '-')}_test_results_{sample_size}.csv"
        )
    elif split == "valid":
        filename += datetime.now().strftime(
            f"_%d-%m-%y-%H_%M_{MODE}_{NAME.replace('/', '-')}_valid_results_{sample_size}.csv"
        )
    labels_df.to_csv(f"{OUTPUTS_PATH}/{filename}", index=False)


if __name__ == "__main__":
    if DATASET_NAME == "liar":
        data_processor = LIARDataProcessor(
            name=NAME, max_seq_length=MAX_SEQUENCE_LENGTH, batch_size=BATCH_SIZE
        )
    elif DATASET_NAME == "kaggle_fake_news":
        data_processor = KaggleFakeNewsDataProcessor(
            name=NAME, max_seq_length=MAX_SEQUENCE_LENGTH, batch_size=BATCH_SIZE
        )
    elif DATASET_NAME == "tfg":
        data_processor = TFGDataProcessor(
            name=NAME, max_seq_length=MAX_SEQUENCE_LENGTH, batch_size=BATCH_SIZE
        )
    elif DATASET_NAME == "isot":
        data_processor = ISOTDataProcessor(
            name=NAME, max_seq_length=MAX_SEQUENCE_LENGTH, batch_size=BATCH_SIZE
        )
    elif DATASET_NAME == "ti_cnn":
        data_processor = TICNNDataProcessor(
            name=NAME, max_seq_length=MAX_SEQUENCE_LENGTH, batch_size=BATCH_SIZE
        )
    elif DATASET_NAME == "nela":
        data_processor = NELADataProcessor(
            name=NAME, max_seq_length=MAX_SEQUENCE_LENGTH, batch_size=BATCH_SIZE
        )
    start_time = time.time()
    dataloaders = data_processor.process(num_samples=NUM_SAMPLES)
    end_time = time.time()
    time_elapsed = end_time - start_time
    print(f"Data Tokenized and Loaded : {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")

    model = BertClassifier(name=NAME, mode=MODE, pretrained_checkpoint=None)
    model = model.to(device)
    print(f"Model Loaded : {NAME} {MODE}")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, eps=EPS)
    total_steps = len(dataloaders["train"]) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # Training/Validation
    start_time = time.time()
    print("Starting train/val loop now : \n")
    train_model(
        model,
        dataloaders["train"],
        dataloaders["valid"],
        criterion,
        optimizer,
        scheduler,
        NUM_EPOCHS,
        MODELS_PATH,
        save_intermediate=False,
    )
    end_time = time.time()
    time_elapsed = end_time - start_time
    print(f"Training Complete : {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")

    # Evaluation
    print("Model Evaluation : \n")
    all_texts_test, all_labels_test, all_preds_test = evaluate_model(
        model, dataloaders["test"], split="test"
    )
    save_test_as_dataframe(
        all_texts_test, all_labels_test, all_preds_test, split="test"
    )
    all_texts_valid, all_labels_valid, all_preds_valid = evaluate_model(
        model, dataloaders["valid"], split="valid"
    )
    save_test_as_dataframe(
        all_texts_valid, all_labels_valid, all_preds_valid, split="valid"
    )
