# Representer Point Selection via Local Jacobian Expansion for Classifier Explanation of Deep Neural Networks and Ensemble Models

This repository is the official implementation of
[Representer Point Selection via Local Jacobian Expansion for Classifier Explanation of Deep Neural Networks and Ensemble Models]()
at NeurIPS 2021. (will update the link)

## Introduction

We propose a novel sample-based explanation method for classifiers with a novel derivation of representer point with
Taylor Expansion on the Jacobian matrix.

If you would like to cite this work, a sample bibtex citation is as following:

```
@inproceedings{yi2021representer,
 author = {Yi Sui, Ga Wu, Scott Sanner},
 booktitle = {Advances in Neural Information Processing Systems},
 title = {Representer Point Selection via Local Jacobian Expansion for Classifier Explanation of Deep Neural Networks and Ensemble Models},
 year = {2021}
}

```

## Set up

To install requirements:

```
pip install -r requirements.txt
```

Change the root path in [config.py](config.py) to the path to the project

```
project_root = #your path here
```

Download the pre-trained models and calculated weights
[here](https://drive.google.com/drive/folders/1JeENy29HNrxay_HAC9m-IDJAeuFVnWaT?usp=sharing)

- Dowload and unzip the saved_models_MODEL_NAME
- Put the content into the corresponding folders (*"models/ MODEL_NAME /saved_models"*)
    - [ResNet (CNN)](models/CNN/saved_models)
    - [Bi-LSTM (RNN)](models/RNN/saved_models)
    - [XGBoost](models/Xgboost/saved_models)

## Training

In our paper, we run experiment with three tasks

- CIFAR image classification with ResNet-20 ([CNN](CNN))
- IMDB sentiment classification with Bi-LSTM ([RNN](RNN))
- German credit analysis with XGBoost ([Xgboost](Xgboost))

The models are implemented in the [models](models) directory with pre-trained weights under *"models/ MODEL_NAME
/saved_models/base"*
: [ResNet (CNN)](models/CNN/saved_models/base), [Bi-LSTM (RNN)](models/RNN/saved_models/base),
and [XGBoost](models/Xgboost/saved_models/base).

To train theses model(s) in the paper, run the following commands:

```
python models/CNN/train.py --lr 0.01 --epochs 10 --saved_path saved_models/base
python models/RNN/train.py --lr 1e-3 --epochs 10 --saved_path saved_models/base --use_pretrained True
python models/Xgboost/train.py
```

## Caculate weights

We implemented three different explainers: RPS-LJE, RPS-l2
(modified from official repository of [RPS-l2](https://github.com/chihkuanyeh/Representer_Point_Selection)), and
Influence Function. To calculate the importance weights, run the following commands:

```
python explainer/calculate_ours_weights.py --model CNN --lr 0.01
python explainer/calculate_representer_weights.py --model RNN --lmbd 0.003 --epoch 3000
python explainer/calculate_influence.py --model Xgboost
```

## Experiments

### Dataset debugging experiment

To run the dataset debugging experiments, run the following commands:

```
python dataset_debugging/experiment_dataset_debugging_cnn.py --num_of_run 10 --flip_portion 0.2 --path ../models/CNN/saved_models/experiment_dataset_debugging --lr 1e-5
python dataset_debugging/experiment_dataset_debugging_cnn.py --num_of_run 10 --flip_portion 0.2 --path ../models/CNN/saved_models/experiment_dataset_debugging_fix_random_split --lr 1e-5 --seed 11

python dataset_debugging/experiment_dataset_debugging_rnn.py --num_of_run 10 --flip_portion 0.2 --path ../models/RNN/saved_models/experiment_dataset_debugging --lr 1e-5

python dataset_debugging/experiment_dataset_debugging_Xgboost.py --num_of_run 10 --flip_portion 0.3 --path ../models/Xgboost/saved_models/experiment_dataset_debugging --lr 1e-5
```

The trained models, intermediate outputs, explainer weights, and accuracies at each checkpoint are stored under the
specified paths
*"models/MODEL_NAME/saved_models/experiment_dataset_debugging"*. To visualize the results, run the notebooks
[plot_res_cnn.ipynb](dataset_debugging/plot_res_cnn.ipynb),
[plot_res_cnn_fixed_random_split.ipynb](dataset_debugging/plot_res_cnn_fixed_random_split.ipynb),
[plot_res_rnn.ipynb](dataset_debugging/plot_res_rnn.ipynb),
[plot_res_xgboost.ipynb](dataset_debugging/plot_res_xgboost.ipynb). The results are saved under
folder [dataset_debugging/figs](dataset_debugging/figs).

### Other experiments

All remaining experiments are in Jupyter-notebooks organized under *"models/ MODEL_NAME /experiments"*
: [ResNet (CNN)](models/CNN/experiments), [Bi-LSTM (RNN)](models/RNN/experiments),
and [XGBoost](models/Xgboost/experiments).

A comparison of explanation provided by Influence Function, RPS-l2, and RPS-LJE.
![Explanation for Image Classification](models/CNN/experiments/figs/img_comparison_3_methods.jpg
)






