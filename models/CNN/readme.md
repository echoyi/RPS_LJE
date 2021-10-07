# CNN model (ResNet-20)

## Train

```python train.py --lr 0.01 --epochs 10 --saved_path saved_models/base```

## Data

We use CIFAR-10 Data under the [data](data) folder. In our experiments, we perform a binary classification task of Horse
vs. Cars.

- [pick_binary_data.ipynb](data/pick_binary_data.ipynb): a Notebook that picks the horse and car samples and divide them
  into train and test datasets.
- [cifar_binary_training.npz](data/cifar_binary_training.npz): training data
- [cifar_binary_testing.npz](data/cifar_binary_testing.npz): testing data