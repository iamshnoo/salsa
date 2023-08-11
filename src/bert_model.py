import os

import torch
import torch.nn as nn
from transformers import AutoModel, logging

logging.set_verbosity_error()

# model settings
NUM_LABELS = 2
BERT_ENCODER_OUTPUT_SIZE = 768
CLF_LAYER_1_DIM = 64
CLF_DROPOUT_PROB = 0.4
MODE = "fine-tune"  # pre-train
NAME = "distilbert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BertClassifier(nn.Module):
    def __init__(self, name=NAME, mode=MODE, pretrained_checkpoint=None):
        super(BertClassifier, self).__init__()
        self.mode = mode
        D_in, H, D_out = BERT_ENCODER_OUTPUT_SIZE, CLF_LAYER_1_DIM, NUM_LABELS
        if pretrained_checkpoint is None:
            self.bert = AutoModel.from_pretrained(NAME)
        else:
            state_dict = torch.load(pretrained_checkpoint, map_location=device)
            self.bert = AutoModel.from_pretrained(
                NAME, state_dict={k: v for k, v in state_dict.items() if "bert" in k}
            )

        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(CLF_DROPOUT_PROB),
            nn.Linear(H, D_out),
        )

        if self.mode == "pre-train":
            freeze_bert = True
        else:
            freeze_bert = False

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:, 0, :]
        logits = self.classifier(last_hidden_state_cls)
        return logits


if __name__ == "__main__":
    model = BertClassifier()
    model = model.to(device)
