import json
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from transformers import BertModel, BertConfig, HfArgumentParser
from transformers.models.bert import modeling_bert
from typing import Optional
from dataclasses import dataclass, field


class LM_Dataset(Dataset):

    def __init__(self, file_path):
        train_file = pd.read_csv(file_path)
        self.input_ids = train_file['input_ids']
        self.token_type_ids = train_file['segment_ids']
        self.attention_mask = train_file['attention_mask']
        self.labels = train_file['labels']
        self.next_sentence_label = train_file['next_sentence_label']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        input_id = [int(id) for id in self.input_ids[index].strip().split(' ')]
        token_type_id = [int(id) for id in self.token_type_ids[index].strip().split(' ')]
        attention_mask = [int(id) for id in self.attention_mask[index].strip().split(' ')]
        label = [int(id) for id in self.labels[index].strip().split(' ')]
        next_sentence_label = [self.next_sentence_label[index]]

        return {'input_ids': torch.tensor(input_id),
                'token_type_ids': torch.tensor(token_type_id),
                'attention_mask': torch.tensor(attention_mask),
                'labels': torch.tensor(label),
                'next_sentence_label': torch.tensor(next_sentence_label)}


def lm_dataloader(file_path, batch_size, shuffle):
    dataset = LM_Dataset(file_path)
    dataloader = DataLoader(dataset, batch_size, shuffle=shuffle, pin_memory=True)
    return dataloader


class BertForPreTraining(nn.Module):

    def __init__(self, model_path, config, output_attentions=False, output_hidden_states=False):
        super().__init__()

        self.config = config
        self.bert = BertModel.from_pretrained(model_path)
        self.cls = modeling_bert.BertPreTrainingHeads(config)  # this layer is only for pretraining
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def save_model(self, output_dir):
        self.bert.save_pretrained(output_dir)

    def forward(self, input_ids, attention_mask, token_type_ids, labels, next_sentence_label):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            output_attentions=self.output_attentions, output_hidden_states=self.output_hidden_states)

        # sequence_output: output vector of tokens, shape [batch_size, sequence_length, hidden_size]
        # pooled_output: output vector of first token, shape [batch_size, hidden_size]
        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        total_loss = None

        if labels is not None and next_sentence_label is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
        #
        #     mlm_total_num, mlm_num, nsp_total_num, nsp_num = 0, 0, 0, 0
        #     with torch.no_grad():
        #         for index, label in enumerate(labels.view(-1)):
        #             if label != -100:
        #                 if torch.argmax(prediction_scores.view(-1, self.config.vocab_size)[index]) == label:
        #                     mlm_num += 1
        #                 mlm_total_num += 1
        #         for index, label in enumerate(next_sentence_label.view(-1)):
        #             if torch.argmax(prediction_scores.view(-1, 2)[index]) == label:
        #                 nsp_num += 1
        #             nsp_total_num += 1
        #         print('mlm_acc:{:6f}, nsp_acc:{:6f}'.format(mlm_num / mlm_total_num, nsp_num / nsp_total_num))

        return prediction_scores, seq_relationship_score, total_loss
