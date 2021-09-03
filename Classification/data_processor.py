import os
import sys
import logging
import pandas as pd
import pickle
from tqdm import tqdm, trange
from multiprocessing import Pool, cpu_count
import pdb

from preprocess import normalizeTweet

logger = logging.getLogger()


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, binary_label=None, level_label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text: string. The untokenized text of the tweet sentences. 
            label: (Optional) int. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.binary_label = binary_label
        self.level_label = level_label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_pkl(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, 'rb') as handle:
            return pickle.load(handle)
    
    @classmethod
    def _read_csv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return pd.read_csv(input_file)


class BinaryClassificationProcessor(DataProcessor):
    """Processor for binary classification dataset."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "dev.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "test.csv")), "test")
    
    def get_unlabeled_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "unlabeled_phase3.csv")), "unlabeled")
    
    def get_unlabeled_pickles(self, data_dir):
        return self._create_examples_from_pkl(
            self._read_pkl(os.path.join(data_dir, "tweet_txt.pkl")), "unlabeled")

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _preprocess_label(self, label):
        binary_label, level_label = 0, 0
        if label > 1:
            binary_label = 1
            level_label = (label - 4) / 3
        return binary_label, level_label
    
    def _create_examples(self, tweets_list, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        # for _, tweet in tqdm(tweets_list.iterrows(), desc="Creating examples"):
        for _, tweet in tweets_list.iterrows():
            guid = tweet['ID']
            text = tweet['Text']
            if 'Information_discrete' in tweet:
                label = tweet['Information_discrete']
                binary_label, level_label = self._preprocess_label(label)
                example = InputExample(guid, text, binary_label, level_label)
            else:
                example = InputExample(guid, text)
            examples.append(example)
        
        return examples
    
    def _create_examples_from_pkl(self, tweets_dict, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        
        for guid, text in tweets_dict.items():
            example = InputExample(guid, text)
            examples.append(example)
        
        return examples


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, binary_label, level_label):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.binary_label = binary_label
        self.level_label = level_label


def convert_example_to_feature(example_row):
    # return example_row
    example, label_map, max_seq_length, tokenizer, output_mode = example_row
    
    # text = normalizeTweet(example.text)
    text = example.text
    tokens = tokenizer.tokenize(text)

    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[:(max_seq_length - 2)]

    tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token] 
    segment_ids = [0] * len(tokens) 

#     ####
#     text = "I disclosed my personal information."
#     tokens_q = tokenizer.tokenize(text)

#     # Account for [SEP] with "- 1"
#     if len(tokens) > 50 - 1:
#         tokens_q = tokens_q[:(50 - 1)]

#     tokens = tokens + tokens_q + [tokenizer.sep_token] 
#     segment_ids += [1] * (len(tokens_q) + 1)
#     ####
    
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
#     padding = [0] * (max_seq_length + 50 - len(input_ids))
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    if len(input_ids) != (max_seq_length):
        pdb.set_trace()
    
    assert len(input_ids) == (max_seq_length)
    assert len(input_mask) == (max_seq_length)
    assert len(segment_ids) == (max_seq_length)

    binary_label = example.binary_label
    level_label = example.level_label
    
    feature = InputFeatures(input_ids=input_ids,
                 input_mask=input_mask,
                 segment_ids=segment_ids,
                 binary_label=binary_label,
                 level_label=level_label)
    
    return feature


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_mode):
    label_map = {label: i for i, label in enumerate(label_list)}
    
    examples = [(example, label_map, max_seq_length, tokenizer, output_mode) for example in examples]
    
#     process_count = cpu_count() - 2
    
#     with Pool(process_count) as p:
#         features = list(tqdm(p.imap(convert_example_to_feature, examples, chunksize=100), total=len(examples)))
#         # features = list(p.imap(convert_example_to_feature, examples, chunksize=100))
    
    features = [convert_example_to_feature(ex) for ex in examples]
    
    return features
    


