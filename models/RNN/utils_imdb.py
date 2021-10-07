import random
import warnings

import torch
from torchtext import data, datasets

import config

warnings.filterwarnings("ignore")


def create_fields():
    TEXT = data.Field(tokenize='spacy',
                      tokenizer_language='en_core_web_sm',
                      include_lengths=True)

    LABEL = data.LabelField(dtype=torch.float)
    return TEXT, LABEL


def load_data(TEXT=None, LABEL=None):
    if TEXT is None or LABEL is None:
        TEXT, LABEL = create_fields()
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL, root='{}/models/RNN/.data'.format(config.project_root))
    train_data, valid_data = train_data.split(random_state=random.seed(1234))
    return train_data, valid_data, test_data


def build_vocab(train_data, TEXT, LABEL):
    MAX_VOCAB_SIZE = 25_000

    TEXT.build_vocab(train_data,
                     max_size=MAX_VOCAB_SIZE,
                     vectors="glove.6B.100d",
                     unk_init=torch.Tensor.normal_)

    LABEL.build_vocab(train_data)
