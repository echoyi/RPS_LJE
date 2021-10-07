# RNN model (Bi-LSTM)

## Train

```python train.py --lr 1e-3 --epochs 10 --saved_path saved_models/base --use_pretrained True```

## Data

IMDB movie review data is is a dataset for binary sentiment classification. The data is available as a torchtext dataset
in Pytorch. A sample usage:

```
TEXT = data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm', include_lengths=True)
LABEL = data.LabelField(dtype=torch.float)
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL,root='{}/models/RNN/.data'.format(config.project_root))
```

[utils_imdb.py](utils_imdb.py) contains functions to load the data