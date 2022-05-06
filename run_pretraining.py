import os
import numpy as np
import random
import torch

from transformers import (
    BertConfig,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    AdamW,
    get_scheduler,
)

from typing import Optional
from dataclasses import dataclass, field
from Model import Bert

from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class DataProcessingArguments:
    train_file: Optional[str] = field(
        metadata={
            "help": "Input raw text file (or comma-separated list of files).",
        }
    )

    valid_file: Optional[str] = field(
        metadata={
            "help": "Input raw text file (or comma-separated list of files).",
        }
    )

    output_dir: Optional[str] = field(
        metadata={
            "help": "Output training file in this position",
        }
    )

    random_seed: Optional[int] = field(
        default=0,
        metadata={
            "help": "Random seed for data generation.",
        }
    )

    do_train: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to run training."
        }
    )

    do_eval: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to run eval on the dev set."
        }
    )


# Fix random seed for reproducibility
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_pretraining_by_trainer():
    parser = HfArgumentParser((DataProcessingArguments, TrainingArguments))
    data_args, train_args = parser.parse_args_into_dataclasses()
    print(train_args)

    set_seed(train_args.seed)
    train_dataset = Bert.LM_Dataset(data_args.train_file)
    valid_dataset = Bert.LM_Dataset(data_args.valid_file)

    config = BertConfig.from_pretrained('bert-base-uncased')
    model = Bert.BertForPreTraining(config)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset
    )

    if not os.path.exists(data_args.out_dir):
        os.makedirs(data_args.out_dir)

    if train_args.do_train:
        train_result = trainer.train()
        print(train_result)
        model.save_model(data_args.out_dir)

    if train_args.do_eval:
        print("*** Evaluate ***")
        eval_result = trainer.evaluate()
        print(eval_result)


def cal_language_model_accuracy(prediction_scores, seq_relationship_score, labels, next_sentence_label):
    mlm_total_num, mlm_num, nsp_total_num, nsp_num = 0, 0, 0, 0
    with torch.no_grad():
        for index, label in enumerate(labels.view(-1)):
            if label != -100:
                if torch.argmax(prediction_scores.view(-1, 30522)[index]) == label:
                    mlm_num += 1
                mlm_total_num += 1
        for index, label in enumerate(next_sentence_label.view(-1)):
            if torch.argmax(seq_relationship_score.view(-1, 2)[index]) == label:
                nsp_num += 1
            nsp_total_num += 1
    mlm_acc = mlm_num / mlm_total_num
    nsp_acc = nsp_num / nsp_total_num

    return mlm_acc, nsp_acc


def run_pretraining():
    parser = HfArgumentParser(DataProcessingArguments)
    data_args = parser.parse_args_into_dataclasses()[0]

    if not os.path.exists(data_args.output_dir):
        os.makedirs(data_args.output_dir)

    same_seeds(data_args.random_seed)

    train_batch_size = 2
    eval_batch_size = 2
    num_train_steps = 1
    num_warmup_steps = 10
    logging_step = 10
    learning_rate = 5e-5
    pretrained_model = "bert-base-uncased"

    train_loader = Bert.lm_dataloader(data_args.train_file, batch_size=train_batch_size, shuffle=True)
    eval_loader = Bert.lm_dataloader(data_args.valid_file, batch_size=eval_batch_size, shuffle=False)

    config = BertConfig.from_pretrained(pretrained_model)
    model = Bert.BertForPreTraining(pretrained_model, config).to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = get_scheduler(
        'linear',
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_steps * len(train_loader),
    )

    if data_args.do_train:
        model.train()
        print("Start Training ...")

        for epoch in range(num_train_steps):
            step = 1
            train_loss = mlm_acc = nsp_acc = 0
            for data in tqdm(train_loader):
                prediction_scores, seq_relationship_score, total_loss = \
                    model(input_ids=data["input_ids"].to(device), attention_mask=data["attention_mask"].to(device),
                          token_type_ids=data["token_type_ids"].to(device), labels=data["labels"].to(device),
                          next_sentence_label=data["next_sentence_label"].to(device))

                train_loss += total_loss
                train_acc = cal_language_model_accuracy(prediction_scores, seq_relationship_score,
                                                        data["labels"], data["next_sentence_label"])
                mlm_acc += train_acc[0]
                nsp_acc += train_acc[1]

                total_loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                step += 1

                if step % logging_step == 0:
                    print(
                        f"Epoch {epoch + 1} | Step {step} | loss = {train_loss / logging_step:.6f}, "
                        f"mlm_acc = {mlm_acc / logging_step:.6f}, nsp_acc = {nsp_acc / logging_step:.6f}")
                    train_loss = mlm_acc = nsp_acc = 0

        print("Saving Model ...")
        model.save_model(data_args.output_dir)

    if data_args.do_eval:
        print("Evaluating Dev Set ...")
        model.eval()
        with torch.no_grad():
            eval_loss = mlm_acc = nsp_acc = 0
            for data in tqdm(eval_loader):
                prediction_scores, seq_relationship_score, total_loss = \
                    model(input_ids=data["input_ids"].to(device), attention_mask=data["attention_mask"].to(device),
                          token_type_ids=data["token_type_ids"].to(device), labels=data["labels"].to(device),
                          next_sentence_label=data["next_sentence_label"].to(device))

                eval_loss += total_loss
                eval_acc = cal_language_model_accuracy(prediction_scores, seq_relationship_score,
                                                       data["labels"], data["next_sentence_label"])
                mlm_acc += eval_acc[0]
                nsp_acc += eval_acc[1]

            print(f"Validation | loss = {eval_loss / len(eval_loader):.6f}, "
                  f"mlm_acc = {mlm_acc / len(eval_loader):.6f}, nsp_acc = {nsp_acc / len(eval_loader):.6f}")

    print("Completed! Language model saved in {}".format(data_args.output_dir))


if __name__ == '__main__':
    run_pretraining()
