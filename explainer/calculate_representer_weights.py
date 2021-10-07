# modified from https://github.com/chihkuanyeh/Representer_Point_Selection/blob/master/compute_representer_vals.py
import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import config

dtype = torch.cuda.FloatTensor
import time
import torch.nn.functional as F

device = torch.device('cuda')


class sigmoid(nn.Module):
    def __init__(self, W):
        super(sigmoid, self).__init__()
        self.W = Variable(W, requires_grad=True)

    def forward(self, x):
        # calculate output and L2 regularizer
        H = torch.matmul(x, self.W.transpose(0, 1))
        Phi = F.sigmoid(H)
        W1 = torch.squeeze(self.W)
        L2 = torch.sum(torch.mul(W1, W1))
        return Phi, L2


def load_data(path):
    file = np.load('{}/saved_outputs.npz'.format(path))
    intermediate_x_train = torch.from_numpy(file['intermediate_train']).to(device).squeeze()
    y_pred = torch.from_numpy(file['pred_train']).to(device).squeeze().unsqueeze(1)
    y_pred.requires_grad = False
    W = torch.from_numpy(file['weight']).to(device)
    model = sigmoid(W).to(device)

    intermediate_x_test = torch.from_numpy(file['intermediate_test']).to(device).data.float()
    y_pred_test = torch.from_numpy(file['pred_test']).squeeze().to(device).data.float()

    return intermediate_x_train, y_pred, model, y_pred_test, intermediate_x_test


def to_np(x):
    return x.data.cpu().numpy()


# implmentation for backtracking line search
def backtracking_line_search(model, grad_w, x, y, val, lambda_l2=0.001):
    t = 10.0
    beta = 0.5
    W_O = to_np(model.W)
    grad_np_w = to_np(grad_w)
    while (True):
        model.W = Variable(torch.from_numpy(W_O - t * grad_np_w).type(dtype), requires_grad=True)
        val_n = 0.0
        (Phi, L2) = model(x)
        val_n = F.binary_cross_entropy(Phi.float(), y.float()) + L2 * lambda_l2
        if t < 0.0000000001:
            # print("t too small")
            break
        if to_np(val_n - val + t * (torch.norm(grad_w) ** 2) / 2) >= 0:
            t = beta * t
        else:
            break


def train(x, y, model, lmbd, epoch, x_test, y_test):
    # Fine tune the last layer
    min_loss = 10000.0
    optimizer = optim.SGD([model.W], lr=1.0)
    N = len(y)
    for epoch in range(epoch):
        phi_loss = 0
        optimizer.zero_grad()
        (Phi, L2) = model(x)
        loss = L2 * lmbd + F.binary_cross_entropy(Phi.float(), y.float())
        phi_loss += to_np(F.binary_cross_entropy(Phi.float(), y.float()))
        loss.backward()
        temp_W = model.W.data
        grad_loss_W = to_np(torch.mean(torch.abs(model.W.grad)))
        # save the W with lowest loss
        if grad_loss_W < min_loss:
            if epoch == 0:
                init_grad = grad_loss_W
            min_loss = grad_loss_W
            best_W = temp_W
            if min_loss < init_grad / 200:
                print('stopping criteria reached in epoch :{}'.format(epoch))
                break
        backtracking_line_search(model, model.W.grad, x, y, loss, lambda_l2=lmbd)
        if epoch % 100 == 0:
            print('Epoch:{:4d}\tloss:{}\tphi_loss:{}\tgrad:{}'.format(epoch, to_np(loss), phi_loss, grad_loss_W))
    # caluculate w based on the representer theorem's decomposition
    temp = torch.matmul(x, Variable(best_W).transpose(0, 1))
    sigmoid_value = F.sigmoid(temp)
    # derivative of sigmoid+BCE
    weight_matrix = sigmoid_value - y
    weight_matrix = torch.div(weight_matrix, (-2.0 * lmbd * N))
    w = torch.matmul(torch.t(x), weight_matrix)
    print(w.shape)
    # calculate y_p, which is the prediction based on decomposition of w by representer theorem
    temp = torch.matmul(x, w.cuda())
    sigmoid_value = F.sigmoid(temp)
    y_p = to_np(sigmoid_value)
    print('L1 difference between ground truth prediction and prediction by representer theorem decomposition')
    print(np.mean(np.abs(to_np(y) - y_p)))
    from scipy.stats.stats import pearsonr
    print('pearson correlation between ground truth  prediction and prediciton by representer theorem')
    y = to_np(y)
    corr, _ = (pearsonr(y.flatten(), (y_p).flatten()))
    print(corr)

    temp_test = torch.matmul(x_test, w.cuda())
    sigmoid_value_test = F.sigmoid(temp_test)
    y_p_test = to_np(sigmoid_value_test)

    print(
        'L1 difference between ground truth prediction and prediction by representer theorem decomposition for test data')
    print(np.mean(np.abs(to_np(y_test) - y_p_test)))

    from scipy.stats.stats import pearsonr
    print('pearson correlation between ground truth  prediction and prediciton by representer theorem for test data')
    y_test = to_np(y_test)
    corr, _ = (pearsonr(y_test.flatten(), (y_p_test).flatten()))
    print(corr)

    sys.stdout.flush()
    return to_np(weight_matrix)


def calculate_representer_weights(path, lmbd=0.003, epoch=3000):
    x, y, model, y_test, x_test = load_data("{}/model".format(path))
    start = time.time()
    weight_matrix = train(x, y, model, lmbd, epoch, x_test, y_test)
    end = time.time()
    print('computational time')
    print(end - start)
    # weight_matrix: $\alpha_i$ in the paper
    np.savez("{}/calculated_weights/representer_weight_matrix".format(path), weight_matrix=weight_matrix)
    return weight_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate RPS-$l_2$ weights')
    parser.add_argument('--model', default='RNN', type=str, help='Interested model: CNN or RNN')
    parser.add_argument('--lmbd', default=0.003, type=float, help='$l_2$ weight')
    parser.add_argument('--epoch', default=3000, type=int)
    args = parser.parse_args()
    path = '{}/models/{}/saved_models/base'.format(config.project_root, args.model)
    calculate_representer_weights(path, lmbd=args.lmbd, epoch=args.epoch)
