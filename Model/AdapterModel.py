import os
import torch
import torch.nn as nn

from transformers.models.bert import modeling_bert
from Model import Adapter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AdapterModel(nn.Module):
    def __init__(self, args, pretrained_model_config):
        super(AdapterModel, self).__init__()
        self.config = pretrained_model_config
        self.args = args
        self.adapter_size = self.args.adapter_size

        # some parameters equal to the configuration of bert
        class AdapterConfig:
            project_hidden_size: int = self.config.hidden_size
            hidden_act: str = "gelu"
            chunk_size_feed_forward: int = 0
            is_decoder: bool = False
            add_cross_attention: bool = False
            adapter_size: int = self.adapter_size  # 64
            adapter_initializer_range: float = 0.0002
            attention_probs_dropout_prob: float = 0.1
            hidden_dropout_prob: float = 0.1
            hidden_size: int = 768
            initializer_range: float = 0.02
            intermediate_size: int = 3072
            layer_norm_eps: float = 1e-12
            max_position_embeddings: int = 512
            num_attention_heads: int = 12
            num_hidden_layers: int = self.args.adapter_transformer_layers
            output_attentions: bool = False
            output_hidden_states: bool = False
            type_vocab_size: int = 2
            vocab_size: int = 30522

        self.adapter_list = args.adapter_list
        self.adapter_num = len(self.adapter_list)
        self.adapter = nn.ModuleList([Adapter.Adapter(AdapterConfig) for _ in range(self.adapter_num)])

        self.com_dense = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)

    def forward(self, pretrained_model_outputs, attention_mask):
        sequence_output = pretrained_model_outputs[0]
        hidden_states = pretrained_model_outputs[2]
        hidden_states_last = torch.zeros(sequence_output.size()).to(device)

        adapter_hidden_states = []
        adapter_hidden_states_count = 0
        for i, adapter_module in enumerate(self.adapter):
            fusion_state = hidden_states[self.adapter_list[i]] + hidden_states_last
            hidden_states_last = adapter_module(fusion_state, attention_mask)
            adapter_hidden_states.append(hidden_states_last)
            adapter_hidden_states_count += 1

        com_features = self.com_dense(torch.cat([sequence_output, hidden_states_last], dim=2))

        return com_features

    def save_pretrained(self, save_dir):
        model_to_save = self.module if hasattr(self, 'module') else self
        model_to_save.config.save_pretrained(save_dir)
        output_model_file = os.path.join(save_dir, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)


class AdapterForPretraining(nn.Module):
    def __init__(self, args, pretrained_model_config):
        super().__init__()
        self.config = pretrained_model_config
        self.adapters = AdapterModel(args, pretrained_model_config)
        self.cls = modeling_bert.BertPreTrainingHeads(pretrained_model_config)

    def forward(self, pretrained_model_outputs, attention_mask, labels, next_sentence_label):
        com_features = self.adapters(pretrained_model_outputs, attention_mask)
        # print(com_features.shape)
        prediction_scores, seq_relationship_score = self.cls(com_features, com_features[:, 0, :])
        total_loss = None

        if labels is not None and next_sentence_label is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss

        return prediction_scores, seq_relationship_score, total_loss
