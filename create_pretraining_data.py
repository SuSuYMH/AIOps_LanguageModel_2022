import os
import random
import collections

from transformers import AutoTokenizer, AutoConfig, HfArgumentParser
from typing import Optional
from dataclasses import dataclass, field

from tqdm import tqdm


@dataclass
class DataProcessingArguments:
    """
    Arguments for sentence tokenization
    """

    input_file: Optional[str] = field(
        metadata={
            "help": "Input raw text file (or comma-separated list of files).",
        }
    )

    output_dir: Optional[str] = field(
        metadata={
            "help": "Output training file in this position",
        }
    )

    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
          "help": "The maximum total input sequence length after tokenization. Sequences longer "
          "than this will be truncated.",
        },
    )

    masked_lm_prob: float = field(
        default=0.15,
        metadata={
            "help": "Ratio of tokens to mask for masked language modeling loss",
        }
    )

    max_predictions_per_seq: Optional[int] = field(
        default=20,
        metadata={
            "help": "Maximum number of masked LM predictions per sequence.",
        }
    )

    random_seed: Optional[int] = field(
        default=12345,
        metadata={
            "help": "Random seed for data generation.",
        }
    )

    dupe_factor: Optional[int] = field(
        default=10,
        metadata={
            "help": "Number of times to duplicate the input data (with different masks).",
        }
    )


class TrainingInstance(object):
    """
    data structure for pretraining
    input_ids: wordpiece index of each sentence pair
    segment_ids: 0 for sentence 1, 1 for sentence 2
    attention_mask: 1 means this wordpiece needs to be calculated in self-attention, o means not
    masked_lm_labels: index of wordpiece after masked, -100 means not masked
    is_random_next: True means sentence pair is not continuous
    """

    def __init__(self, input_ids, segment_ids, attention_masks, masked_lm_labels,
                 is_random_next):
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.attention_masks = attention_masks
        self.masked_lm_labels = masked_lm_labels
        self.is_random_next = is_random_next

    def __str__(self):
        s = ""
        s += "input_ids: %s\n" % (" ".join([str(x) for x in self.input_ids]))
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
        s += "attention_mask: %s\n" % (" ".join([str(x) for x in self.attention_masks]))
        s += "is_random_next: %s\n" % self.is_random_next
        s += "masked_lm_labels: %s\n" % (" ".join([str(x) for x in self.masked_lm_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def create_training_instances(input_files, tokenizer, max_seq_length,
                              dupe_factor, masked_lm_prob,
                              max_predictions_per_seq, vocab_size, rng):
    all_documents = [[]]
    # Input file format:
    # (1) One sentence per line. These should ideally be actual sentences, not
    # entire paragraphs or arbitrary spans of text. (Because we use the
    # sentence boundaries for the "next sentence prediction" task).
    # (2) Blank lines between documents. Document boundaries are needed so
    # that the "next sentence prediction" task doesn't span between documents.
    for input_file in input_files:
        with open(input_file, 'r') as reader:
            while True:
                line = reader.readline()
                if not line:
                    break
                line = line.strip()

                if not line:
                    all_documents.append([])
                else:
                    tokens = tokenizer(line)
                    if tokens:
                        all_documents[-1].append(tokens["input_ids"][1:-1])  # remove [CLS] and [SEP]
                        # print(all_documents[-1])

    print("Start Processing")
    all_documents = [x for x in all_documents if x]
    rng.shuffle(all_documents)

    instances = []
    for i in range(dupe_factor):
        print("第 {} 次构建训练集， 共 {} 次".format(i + 1, dupe_factor))
        for document_index in tqdm(range(len(all_documents))):
            instances.extend(
                create_instances_from_document(
                    all_documents, document_index, max_seq_length,
                    masked_lm_prob, max_predictions_per_seq, vocab_size, rng)
            )
    rng.shuffle(instances)

    return instances


def create_instances_from_document(all_documents, document_index, max_seq_length,
                                   masked_lm_prob, max_predictions_per_seq, vocab_size, rng):
    """
    cut out a fixed sequence length from a paragraph
    :return: list of instances
    """
    document = all_documents[document_index]

    instances = []
    current_chunk = []
    current_length = 0
    max_num_tokens = max_seq_length - 3  # 3 means [CLS], [SEP], [SEP]
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= max_num_tokens:
            if current_chunk:
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = rng.randint(1, len(current_chunk) - 1)

                # tokens_a is sentence A, tokens_b is sentence B
                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                is_random_next = False
                # nsp task
                if len(current_chunk) == 1 or rng.random() < 0.5:
                    is_random_next = True

                    target_b_length = max_num_tokens - len(tokens_a)

                    #  choose tokens_b from other document
                    for _ in range(10):
                        random_document_index = rng.randint(0, len(all_documents) - 1)
                        if random_document_index != document_index:
                            break

                    random_document = all_documents[random_document_index]
                    random_start = rng.randint(0, len(random_document) - 1)

                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break

                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments

                else:
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])

                # trim sequence to fixed length
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                input_ids = []
                segment_ids = []
                attention_mask = []

                # 101 = [CLS], 102 = [SEP]
                input_ids.append(101)
                segment_ids.append(0)
                attention_mask.append(1)
                for token in tokens_a:
                    input_ids.append(token)
                    segment_ids.append(0)
                    attention_mask.append(1)

                input_ids.append(102)
                segment_ids.append(0)
                attention_mask.append(1)
                for token in tokens_b:
                    input_ids.append(token)
                    segment_ids.append(1)
                    attention_mask.append(1)

                input_ids.append(102)
                segment_ids.append(1)
                attention_mask.append(1)

                assert len(input_ids) <= max_seq_length

                (input_ids, masked_lm_labels) = create_masked_lm_predictions(
                    input_ids, masked_lm_prob, max_predictions_per_seq, vocab_size, rng)

                # add 0 to pad a sequence if length < fixed length
                if len(input_ids) < max_seq_length:
                    filling_length = max_seq_length - len(input_ids)
                    input_ids.extend([0]*filling_length)
                    segment_ids.extend([0]*filling_length)
                    attention_mask.extend([0]*filling_length)
                    masked_lm_labels.extend([-100]*filling_length)

                instance = TrainingInstance(
                    input_ids=input_ids,
                    segment_ids=segment_ids,
                    attention_masks=attention_mask,
                    is_random_next=is_random_next,
                    masked_lm_labels=masked_lm_labels)
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1

    return instances


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions(input_ids, masked_lm_prob,
                                 max_predictions_per_seq, vocab_size, rng):
    """
    mask wordpiece to predict
    :return: input_ids, masked_lm_labels
    """
    cand_indexes = []
    for i, input_id in enumerate(input_ids):
        # skip [cls] and [sep]
        if input_id == 101 or input_id == 102:
            continue
        else:
            cand_indexes.append([i])

    rng.shuffle(cand_indexes)

    output_ids = list(input_ids)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(input_ids) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break

        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_input_id = None
            # 80% of the time, replace with [MASK]
            if rng.random() < 0.8:
                masked_input_id = 103  # 103 is the index of [MASK]
            else:
                # 10% of the time, replace nothing
                if rng.random() < 0.5:
                    masked_input_id = index
                # 10% of the time, replace with random word
                else:
                    masked_input_id = rng.randint(0, vocab_size - 1)

            output_ids[index] = masked_input_id
            masked_lms.append(MaskedLmInstance(index=index, label=input_ids[index]))
    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_labels = [-100]*len(output_ids)
    for p in masked_lms:
        masked_lm_labels[p.index] = p.label

    return output_ids, masked_lm_labels


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # sometimes truncate from the front and sometimes from the back
        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


def write_instances_to_files(instances, out_dir, train_file_name, valid_file_name):
    """
    Divide the training set and verification set in 9:1 proportion
    """
    with open(out_dir + train_file_name, 'w+') as train_file_obj, \
            open(out_dir + valid_file_name, 'w+') as valid_file_obj:
        train_file_obj.write('input_ids,segment_ids,attention_mask,labels,next_sentence_label\n')
        valid_file_obj.write('input_ids,segment_ids,attention_mask,labels,next_sentence_label\n')
        for i in range(len(instances)):
            if i < len(instances) * 0.9:
                train_file_obj.write(' '.join([str(input_id) for input_id in instances[i].input_ids]) + ', ')
                train_file_obj.write(' '.join([str(segment_id) for segment_id in instances[i].segment_ids]) + ', ')
                train_file_obj.write(' '.join([str(attention_mask) for attention_mask
                                               in instances[i].attention_masks]) + ', ')
                train_file_obj.write(' '.join([str(label) for label in instances[i].masked_lm_labels]) + ', ')
                train_file_obj.write('1\n' if instances[i].is_random_next else '0\n')
            else:
                valid_file_obj.write(' '.join([str(input_id) for input_id in instances[i].input_ids]) + ', ')
                valid_file_obj.write(' '.join([str(segment_id) for segment_id in instances[i].segment_ids]) + ', ')
                valid_file_obj.write(' '.join([str(attention_mask) for attention_mask
                                               in instances[i].attention_masks]) + ', ')
                valid_file_obj.write(' '.join([str(label) for label in instances[i].masked_lm_labels]) + ', ')
                valid_file_obj.write('1\n' if instances[i].is_random_next else '0\n')

    print('预训练集生成完成！')


def main():
    parser = HfArgumentParser(DataProcessingArguments)
    data_args = parser.parse_args_into_dataclasses()[0]

    # you can change pretrained_model here
    pretrained_model = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    config = AutoConfig.from_pretrained(pretrained_model)

    input_files = []
    train_file_name = '/log_train_file'
    valid_file_name = '/log_valid_file'
    for input_pattern in data_args.input_file.split(","):
        input_files.append(input_pattern)

    print("*** Reading from input files ***")
    for input_file in input_files:
        print(input_file)

    rng = random.Random(data_args.random_seed)
    instances = create_training_instances(
        input_files, tokenizer, data_args.max_seq_length, data_args.dupe_factor,
        data_args.masked_lm_prob, data_args.max_predictions_per_seq, config.vocab_size, rng)

    # print(instances)

    if not os.path.exists(data_args.output_dir):
        os.makedirs(data_args.output_dir)

    write_instances_to_files(instances, data_args.output_dir, train_file_name, valid_file_name)


if __name__ == '__main__':
    main()
