import json
import os
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset
from Model import Adapter, AdapterModel

from transformers import (
    BertConfig,
    BertModel,
    BertTokenizer,
    HfArgumentParser,
)

from typing import Optional
from dataclasses import dataclass, field

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class DataProcessingArguments:
    input_file: Optional[str] = field(
        metadata={
            "help": "The text need to be vectorized",
        }
    )

    model_path: Optional[str] = field(
        metadata={
            "help": "the pretrained model you will use, contains a config file and parameters."
        }
    )

    output_dir: Optional[str] = field(
        metadata={
            "help": "output vectors",
        }
    )


class Log_Patterns(Dataset):

    def __init__(self, tokens):
        self.input_ids = tokens['input_ids']
        self.token_type_ids = tokens['token_type_ids']
        self.attention_mask = tokens['attention_mask']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return {'input_ids': torch.tensor(self.input_ids[index]),
                'token_type_ids': torch.tensor(self.token_type_ids[index]),
                'attention_mask': torch.tensor(self.attention_mask[index])}


class AdapterArgs:
    adapter_size: int = 768
    adapter_transformer_layers: int = 2
    adapter_list: list = [0, 6, 11]


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_file(file_path):
    """
    load all log templates here
    :return: list of sentences should be vectorized
    """
    ##### TODO: Change parsing method if you need #####
    with open(file_path, 'r') as data_file:
        data_file = json.load(data_file)

    return list(data_file.values())


# bert
def vectorize(log_templates, model_path):
    model = BertModel.from_pretrained(model_path).to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    same_seeds(0)

    tokens = tokenizer(log_templates, padding=True)
    # print(tokens)

    batch_size = 8
    data_loader = DataLoader(Log_Patterns(tokens), batch_size=batch_size, shuffle=False)

    output_vectors = []
    with torch.no_grad():
        for data in data_loader:
            outputs = model(input_ids=data['input_ids'].to(device),
                            attention_mask=data['attention_mask'].to(device),
                            token_type_ids=data['token_type_ids'].to(device),
                            output_hidden_states=True)

            # choose vector of [CLS] to represent whole sentence
            pooled_output = outputs[1]
            output_vectors.extend(pooled_output.tolist())

    return output_vectors


# adapters
def vectorize_by_adapters(log_templates, model_path):
    config = BertConfig.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    pretrained_model = Adapter.PretrainedModel().to(device)
    adapters = AdapterModel.AdapterModel(AdapterArgs, config).to(device)
    adapters.load_state_dict(torch.load(model_path + '/pytorch_model.bin'))

    same_seeds(0)

    tokens = tokenizer(log_templates, padding=True)
    batch_size = 8
    data_loader = DataLoader(Log_Patterns(tokens), batch_size=batch_size, shuffle=False)

    output_vectors = []
    for data in data_loader:
        with torch.no_grad():
            outputs = pretrained_model(input_ids=data['input_ids'].to(device),
                                       attention_mask=data['attention_mask'].to(device),
                                       token_type_ids=data['token_type_ids'].to(device), )
            com_features = adapters(outputs, attention_mask=data["attention_mask"].to(device))
            output_vectors.extend(com_features[:, 0, :].tolist())

    return output_vectors


def save_vectors(output_vectors, output_dir):
    pattern_vec = {}
    for index, vector in enumerate(output_vectors):
        pattern_vec[str(index)] = [float(v) for v in vector]
    pattern_vec = json.dumps(pattern_vec)

    output_file = 'pattern2vec_SOP'
    with open(output_dir + '/' + output_file, 'w') as output_file_obj:
        output_file_obj.write(pattern_vec)


if __name__ == '__main__':
    parser = HfArgumentParser(DataProcessingArguments)
    data_args = parser.parse_args_into_dataclasses()[0]

    text = load_file(data_args.input_file)

    # output_vectors = vectorize(text, data_args.model_path)
    output_vectors = vectorize_by_adapters(text, data_args.model_path)

    if not os.path.exists(data_args.output_dir):
        os.makedirs(data_args.output_dir)

    save_vectors(output_vectors, data_args.output_dir)
