import numpy as np
import torch
import torch.nn as nn

from transformers import BertConfig, BertModel
from transformers.models.bert import modeling_bert
from transformers.modeling_utils import PreTrainedModel


class Adapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.down_project = nn.Linear(
            self.config.project_hidden_size,
            self.config.adapter_size,
        )
        self.encoder = modeling_bert.BertEncoder(config)
        self.up_project = nn.Linear(self.config.adapter_size, self.config.project_hidden_size)
        self.init_weights()

    def get_extended_attention_mask(self, attention_mask, input_shape):
        """
        code in transformers, extend attention mask for encoder
        """
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, hidden_states, attention_mask):
        down_projected = self.down_project(hidden_states)
        input_shape = hidden_states.size()[:-1]
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)
        head_mask = [None] * self.config.num_hidden_layers
        encoder_outputs = self.encoder(down_projected,
                                       attention_mask=extended_attention_mask,
                                       head_mask=head_mask)
        up_projected = self.up_project(encoder_outputs[0])

        return hidden_states + up_projected

    def init_weights(self):
        self.down_project.weight.data.normal_(mean=0.0, std=self.config.adapter_initializer_range)
        self.down_project.bias.data.zero_()
        self.up_project.weight.data.normal_(mean=0.0, std=self.config.adapter_initializer_range)
        self.up_project.bias.data.zero_()


class PretrainedModel(nn.Module):
    def __init__(self):
        super(PretrainedModel, self).__init__()
        pretrained_model = "bert-base-uncased"
        self.bert = BertModel.from_pretrained(pretrained_model)

        # fix parameters while training
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            output_attentions=True, output_hidden_states=True)

        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)
