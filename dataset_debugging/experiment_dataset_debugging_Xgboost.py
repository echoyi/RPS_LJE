import argparse
import time

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from calculate_orders import calculate_orders, calculate_orders_ours
from explainer.calculate_influence_weights import calculate_influence_weights
from explainer.calculate_ours_weights import calculate_ours_weights
from models.Xgboost.train import train_with_flipped, train_and_record_accuracies


def plt_accuracies(path, methods, experiment_range=8):
    d = {}
    for m, description in methods.items():
        d[description] = np.load('{}/accuracies/accuracies_{}.npy'.format(path, m))

    idx = pd.Index(data=np.linspace(0.05, 0.05 * experiment_range, experiment_range),
                   name='Fraction of training data checked')
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


def run(path, num_of_run, num_of_samples=1118, portion=0.2, lr=None):
    path = '{}/{}'.format(path, num_of_run)
    # create order
    print('---------------generating random flipping order---------------')
    flipping_order = np.random.permutation(num_of_samples)
    np.save('{}/orders/random_flipping_order'.format(path), flipping_order)

    print('---------------train with flipped data---------------')
    # train with flipped data
    train_with_flipped(flipped_idx=None, path=path, save=True, portion=portion, num_of_samples=num_of_samples)

    print('---------------calculating weights---------------')
    print('---------------calculating IF weights---------------')
    calculate_influence_weights(path)
    print('---------------calculating ours weights---------------')
    calculate_ours_weights(path, lr)

    calculate_orders(path, num_of_samples=num_of_samples, portion=portion, include_rep=False)
    calculate_orders_ours(path, lr, num_of_samples=num_of_samples, portion=portion)

    methods = {'IF': 'Influence function', 'random': 'Random', 'ours': 'Ours'}

    for method in methods.keys():
        time_1 = time.time()
        print('----------calculating accuracy for method {}--------------'.format(methods[method]))
        train_and_record_accuracies(path, method, save=False, portion=portion,
                                    num_of_samples=1118, average_over_runs=5, experiment_range=20)
        print('-------Time elapsed:{}----'.format(time.time() - time_1))

    plt_accuracies(path, methods, experiment_range=20)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_of_run', type=int, default=10)
    parser.add_argument('--flip_portion', type=float, default=0.3)
    parser.add_argument('--path', type=str, default='../models/Xgboost/saved_models/experiment_dataset_debugging')
    parser.add_argument('--lr', type=float, default=1e-5)
    args = parser.parse_args()
    for i in range(args.num_of_run):
        run(args.path, i, num_of_samples=1118, portion=args.flip_portion, lr=args.lr)
