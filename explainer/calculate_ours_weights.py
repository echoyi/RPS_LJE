import argparse
import time
import warnings

import numpy as np
import torch
from torch import optim
from torch.autograd import Variable

import config

warnings.filterwarnings("ignore")

device = torch.device('cuda')


class sigmoid(torch.nn.Module):
    def __init__(self, W):
        super(sigmoid, self).__init__()
        self.W = Variable(W, requires_grad=True)

    def forward(self, x):
        H = torch.matmul(x, self.W.transpose(0, 1))
        Phi = torch.sigmoid(H)
        return Phi


def to_np(x):
    return x.data.cpu().numpy()


def to_tensor(x):
    return torch.tensor(x).data.float().to(device)


def load_data(path):
    file = np.load('{}/saved_outputs.npz'.format(path))
    intermediate_x_train = to_tensor(file['intermediate_train'])
    y_pred = to_tensor(file['pred_train']).squeeze()
    y = to_tensor(file['labels_train'])
    W = to_tensor(file['weight'])
    model = sigmoid(W).to(device)

    intermediate_x_test = torch.from_numpy(file['intermediate_test']).to(device).data.float()
    y_pred_test = torch.from_numpy(file['pred_test']).squeeze().to(device).data.float()
    return intermediate_x_train, y_pred, y, model, y_pred_test, intermediate_x_test


def calculate_grads(path, lr=0.1):
    intermediate_x, y_pred, y, model, y_pred_test, intermediate_x_test = load_data('{}/model'.format(path))
    intermediate_x = intermediate_x.squeeze().unsqueeze(1)
    intermediate_x_test = intermediate_x_test.squeeze().unsqueeze(1)
    # train one step
    optimizer = optim.SGD([model.W], lr=lr)
    optimizer.zero_grad()
    Phi = model(intermediate_x).squeeze()
    loss = -torch.nn.BCELoss()(Phi.float(), y.float())
    loss.backward()
    optimizer.step()

    Phi = model(intermediate_x).squeeze()

    grad_first = torch.bmm((Phi - y).view(-1, 1, 1), intermediate_x)
    hessian_inverse = torch.inverse(torch.mean(torch.bmm(torch.bmm(intermediate_x.transpose(1, 2),
                                                                   (Phi * (1 - Phi)).view(-1, 1, 1)), intermediate_x),
                                               dim=0))
    # calculate the y_pred from representer
    y_pred_representer = torch.sigmoid(torch.matmul(model.W.reshape((1, -1)) -
                                                    torch.mean(grad_first, dim=0) @
                                                    hessian_inverse,
                                                    intermediate_x.transpose(1, 2)).squeeze())
    print('L1 difference between ground truth prediction and prediction by representer theorem decomposition')
    print(np.mean(np.abs(to_np(y_pred) - to_np(y_pred_representer))))

    from scipy.stats.stats import pearsonr
    print('pearson correlation between ground truth  prediction and prediciton by representer theorem')
    corr, _ = (pearsonr(to_np(y_pred).flatten(), to_np(y_pred_representer).flatten()))
    print(corr)

    y_pred_representer_test = torch.sigmoid(torch.matmul(model.W.reshape((1, -1)) -
                                                         torch.mean(grad_first, dim=0) @
                                                         hessian_inverse,
                                                         intermediate_x_test.transpose(1, 2)).squeeze())
    print(
        'L1 difference between ground truth prediction and prediction by representer theorem decomposition for test data')
    print(np.mean(np.abs(to_np(y_pred_test) - to_np(y_pred_representer_test))))
    from scipy.stats.stats import pearsonr
    print('pearson correlation between ground truth prediction and prediction by representer theorem for test data')
    corr, _ = (pearsonr(to_np(y_pred_test).flatten(), to_np(y_pred_representer_test).flatten()))
    print(corr)
    return grad_first, hessian_inverse, model.W.data


def calculate_ours_weights(path, lr=0.01):
    time1 = time.time()
    grads_first, hessian_inverse, theta_hat = calculate_grads(path, lr=lr)
    samples = grads_first.shape[0]
    intermediate_x = to_tensor(
        np.load('{}/model/saved_outputs.npz'.format(path))['intermediate_train']).squeeze().unsqueeze(1)
    weight_matrix = []
    self_influence = []
    for i in range(samples):
        weight = (theta_hat - torch.matmul(grads_first[i], hessian_inverse)) / samples
        weight_matrix.append(weight.data)
        self_influence.append(torch.matmul(weight, intermediate_x[i].transpose(0, 1)).squeeze())
    weight_matrix = torch.stack(weight_matrix, dim=0)
    self_influence = torch.stack(self_influence, dim=0)
    print("time took to compute weights {}".format(time.time() - time1))
    np.savez('{}/calculated_weights/ours_weight_matrix_with_lr_{}'.format(path, lr),
             weight_matrix=to_np(weight_matrix), self_influence=to_np(self_influence))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate RPS-LJE weights')
    parser.add_argument('--model', default='CNN', type=str, help='Interested model: CNN, RNN, or Xgboost')
    parser.add_argument('--lr', default=0.01, type=float, help='Learning rate of one-step gradient ascent')
    args = parser.parse_args()
    calculate_ours_weights('{}/models/{}/saved_models/base'.format(config.project_root, args.model), lr=args.lr)
