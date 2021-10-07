# XGBoost model

## Train

Run the following script to train the model:

```python train.py```

## Data

We use German Credit Data under the [data](data) folder. German credit dataset classifies people described by a set of
attributes as good or bad credit risks. We use the XGBoost model to build a classifier for the data.

### Raw data

- [Data source link](http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data)
- [Attribute information](data/attribute_names.txt)
- [Raw data csv](data/german_data.csv)

### Processing

- [data-preprocessing.py](data/data-preprocessing.py): A pre-processing Python script that downloads, encodes and scales
  the data.
- [translate_categorical_attributes.py](data/translate_categorical_attributes.py): A Python script that translate the
  raw categorical data into human-readable features.