import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt


# %%
def to_np(x):
    return x.data.cpu().numpy()


def to_tensor(x):
    return torch.tensor(x).cuda().float()


# flipped_idx
def plt_and_save(df, file_path, ylabel='fraction fixed', full_range=False):
    fig, ax = plt.subplots(figsize=(10, 10))
    if full_range:
        ax = sns.lineplot(data=df, markers=True)
    else:
        ax = sns.lineplot(data=df.iloc[0:8, :], markers=True)
    for l in ax.lines:
        xy = l.get_xydata()
        if len(xy) > 0:
            for data in xy:
                x = data[0]
                y = data[1]
                ax.annotate(f'{y:.3f}', xy=(x, y), xycoords=('data', 'data'),
                            ha='left', va='center', color=l.get_color())
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(file_path)
    # plt.show()


def check_and_save(order, save_path, flip_idx, num_of_samples):
    count = 0
    corrected = []
    fraction_fixed = []
    remaining_flipped = list(flip_idx.copy())
    flipped_at_portion = []
    for i in order:
        if i in remaining_flipped:
            corrected.append(i)
            remaining_flipped.remove(i)
        count += 1
        if count % int(num_of_samples * 0.05) == 0:
            fraction_fixed.append(len(corrected) / len(flip_idx))
            flipped_at_portion.append(np.array(remaining_flipped))
    frac_fixed = np.array(fraction_fixed)
    inputs = np.array(flipped_at_portion, dtype=object)
    np.savez(save_path, order=order, inputs=inputs, fraction_fixed=frac_fixed)
    return frac_fixed


def calculate_orders(path, num_of_samples=17500, portion=0.2, include_rep=True):
    flip_idx = np.load('{}/orders/random_flipping_order.npy'.format(path))[:int(num_of_samples * portion)]
    print(len(flip_idx))
    d = {}
    weights_path = '{}/calculated_weights'.format(path)
    print("------------- calculating for IF -------------")
    self_influence_IF = np.load('{}/influence_weight_matrix.npz'.format(weights_path))['self_influence'].squeeze()
    # find bigger value
    order_IF = np.flip(np.argsort(self_influence_IF))
    d['Influence function'] = check_and_save(order_IF, '{}/orders/order_IF'.format(path), flip_idx, num_of_samples)
    if include_rep:
        print("------------- calculating for Representer Pt -------------")
        weight_REP = np.load('{}/representer_weight_matrix.npz'.format(weights_path))['weight_matrix'].squeeze()
        order_REP = np.flip(np.argsort(np.abs(weight_REP)))
        d['Representer point'] = check_and_save(order_REP, '{}/orders/order_rep'.format(path), flip_idx, num_of_samples)
    print("------------- calculating for Random order-------------")
    order_random = np.random.permutation(num_of_samples)
    d['Random'] = check_and_save(order_random, '{}/orders/order_random'.format(path), flip_idx, num_of_samples)
    idx = pd.Index(data=np.linspace(0.05, 1, 20), name='Fraction of training data checked')
    df = pd.DataFrame(data=d, index=idx)
    plt_and_save(df, '{}/orders/baselines.png'.format(path))


def calculate_orders_ours(path, lr, num_of_samples=10000, portion=0.2):
    weights_path = '{}/calculated_weights'.format(path)
    flip_idx = np.load('{}/orders/random_flipping_order.npy'.format(path))[:int(num_of_samples * portion)]
    self_influence_ours = np.load('{}/ours_weight_matrix_with_lr_{}.npz'
                                  .format(weights_path, lr))['self_influence']
    order_ours = np.flip(np.argsort(np.abs(self_influence_ours)))
    check_and_save(order_ours, '{}/orders/order_ours'.format(path), flip_idx,
                   num_of_samples)


def plot_all_orders(path, include_alternative=False, full_range=False):
    order_path = '{}/orders'.format(path)
    frac_IF = np.load('{}/order_IF.npz'.format(order_path))['fraction_fixed']
    frac_REP = np.load('{}/order_rep.npz'.format(order_path))['fraction_fixed']
    frac_random = np.load('{}/order_random.npz'.format(order_path))['fraction_fixed']
    frac_ours = np.load('{}/order_ours.npz'.format(order_path))['fraction_fixed']
    data = {'Representer point': frac_REP,
            'Influence function': frac_IF,
            'Ours': frac_ours,
            'Random': frac_random}
    if include_alternative:
        data['Ours(alternative)'] = np.load('{}/order_ours_alternative.npz'.format(order_path))['fraction_fixed']
    idx = pd.Index(data=np.linspace(0.05, 1, 20), name='Fraction of training data checked')
    df = pd.DataFrame(data=data, index=idx)
    save_to_path = '{}/orders/orders-comparison.png'.format(path)
    if full_range: save_to_path = '{}/orders/orders-comparison-full-range.png'.format(path)
    plt_and_save(df, save_to_path, full_range=full_range)
