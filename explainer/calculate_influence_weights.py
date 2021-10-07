import argparse

import numpy as np
import torch

import config

device = torch.device('cuda')


def to_np(x):
    return x.data.cpu().numpy()


def to_tensor(x):
    return torch.tensor(x).float().to(device)


def load_data(path):
    file = np.load('{}/saved_outputs.npz'.format(path))
    intermediate_x_train = to_tensor(file['intermediate_train'])
    intermediate_x_test = to_tensor(file['intermediate_test'])
    y_pred = to_tensor(file['pred_train']).squeeze()
    y = to_tensor(file['labels_train']).float()
    y_test = to_tensor(file['labels_test']).float()
    W = to_tensor(file['weight']).to(device)
    return intermediate_x_train, intermediate_x_test, y_pred, y, y_test, W


def calculate_grads(path):
    intermediate_x_train, intermediate_x_test, y_pred, y, y_test, W = load_data(path)
    intermediate_x_train = intermediate_x_train.squeeze().unsqueeze(1)
    intermediate_x_test = intermediate_x_test.squeeze().unsqueeze(1)
    # use trained model to calculate influences
    Phi = torch.sigmoid(torch.matmul(intermediate_x_train, W.transpose(0, 1)).squeeze())
    grad_first = torch.bmm((Phi - y).view(-1, 1, 1), intermediate_x_train)

    Phi_test = torch.sigmoid(torch.matmul(intermediate_x_test, W.transpose(0, 1)).squeeze())
    jaccobian_test = torch.bmm((Phi_test - y_test).view(-1, 1, 1), intermediate_x_test)

    grad_second = torch.bmm(torch.bmm(intermediate_x_train.transpose(1, 2), (Phi * (1 - Phi)).view(-1, 1, 1)),
                            intermediate_x_train)
    hessian_inverse = torch.inverse(torch.mean(grad_second, dim=0))
    return grad_first, hessian_inverse, jaccobian_test


def calculate_influence_weights(path):
    grads_first, hessian_inverse, jaccobian_test = calculate_grads('{}/model'.format(path))
    samples = len(grads_first)
    weight_matrix = []
    self_influence = []
    for i in range(samples):
        weight = (-1 / samples) * torch.matmul(hessian_inverse, grads_first[i].transpose(0, 1))
        weight_matrix.append(weight)
        self_influence.append(-1 * torch.matmul(grads_first[i], weight))
    weight_matrix = torch.stack(weight_matrix)
    self_influence = torch.stack(self_influence)
    np.savez('{}/calculated_weights/influence_weight_matrix'.format(path),
             weight_matrix=to_np(weight_matrix), self_influence=to_np(self_influence),
             jaccobian_test=to_np(jaccobian_test))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate Influence Function weights')
    parser.add_argument('--model', default='RNN', type=str, help='Interested model: CNN, RNN, or Xgboost')
    args = parser.parse_args()
    calculate_influence_weights(path='{}/models/{}/saved_models/base'.format(config.project_root, args.model))
