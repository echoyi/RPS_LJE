import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from calculate_orders import calculate_orders, calculate_orders_ours, plot_all_orders
from explainer.calculate_influence_weights import calculate_influence_weights
from explainer.calculate_ours_weights import calculate_ours_weights
from explainer.calculate_representer_weights import calculate_representer_weights
from models.RNN.train_with_flipped_data import train_sentiment_model_with_flipped_data, \
    train_and_record_accuracies


def plt_accuracies(path, methods):
    d = {}
    for m, description in methods.items():
        d[description] = np.load('{}/accuracies/accuracies_{}.npy'.format(path, m))
    idx = pd.Index(data=np.linspace(0.05, 0.4, 8), name='Fraction of training data checked')
    df = pd.DataFrame(data=d, index=idx)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = sns.lineplot(data=df, markers=True)
    for l in ax.lines:
        xy = l.get_xydata()
        if len(xy) > 0:
            for data in xy:
                x = data[0]
                y = data[1]
                ax.annotate(f'{y:.2f}', xy=(x, y), xycoords=('data', 'data'),
                            ha='left', va='center', color=l.get_color())
    plt.ylabel('test accuracy')
    plt.tight_layout()
    plt.savefig('{}/accuracies/accuracies_plot.png'.format(path))


def run(path, num_of_run, num_of_samples=17500, portion=0.2, lr=None):
    path = '{}/{}'.format(path, num_of_run)
    # create order
    if not os.path.exists('{}/orders/random_flipping_order.npy'.format(path)):
        print('---------------generating random flipping order---------------')
        flipping_order = np.random.permutation(num_of_samples)
        np.save('{}/orders/random_flipping_order'.format(path), flipping_order)

    if not os.path.exists('{}/model/saved_outputs.npz'.format(path)):
        print('---------------train with flipped data---------------')
        # train with flipped data
        train_sentiment_model_with_flipped_data(path, epochs=20, lr=5e-3, save=True,
                                                portion=portion, num_of_samples=num_of_samples,
                                                use_pretrained=True)
    #
    print('---------------calculating weights---------------')
    # calculate weights with 3  methods: ours, representer, influence
    print('---------------calculating representer weights---------------')
    calculate_representer_weights(path)
    print('---------------calculating IF weights---------------')
    calculate_influence_weights(path)
    print('---------------calculating ours weights---------------')
    calculate_ours_weights(path, lr)

    calculate_orders(path, num_of_samples=num_of_samples, portion=portion)
    calculate_orders_ours(path, lr, num_of_samples=num_of_samples, portion=portion)
    plot_all_orders(path, include_alternative=False, full_range=False)
    plot_all_orders(path, include_alternative=False, full_range=True)

    methods = {'ours': 'Ours', 'rep': 'Representer point',
               'random': 'Random', 'IF': 'Influence function'}

    if not os.path.exists('{}/accuracies/accuracies_plot.png'.format(path)):
        for method in methods.keys():
            if not os.path.exists('{}/accuracies/accuracies_{}.npy'.format(path, method)):
                print('----------calculating accuracy for method {}--------------'.format(methods[method]))
                print("Time:", datetime.now().strftime("%H:%M:%S"))
                train_and_record_accuracies(path, method, lr=5e-3,
                                            epochs=20, save=False,
                                            portion=0.2, num_of_samples=10000,
                                            average_over_runs=3, use_pretrained=True)
            torch.cuda.empty_cache()

        plt_accuracies(path, methods)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_of_run', type=int, default=10)
    parser.add_argument('--flip_portion', type=float, default=0.2)
    parser.add_argument('--path', type=str, default='../models/RNN/saved_models/experiment_dataset_debugging')
    parser.add_argument('--lr', type=float, default=1e-5)
    args = parser.parse_args()

    for i in range(args.num_of_run):
        run(args.path, i, num_of_samples=17500, portion=args.flip_portion, lr=args.lr)
        torch.cuda.empty_cache()
