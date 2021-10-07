import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import config
from models.CNN.CNN import ResNet20, train_ResNet, test_ResNet, calculate_intermediate_output
from utils_cifar import load_cifar_2


def flip_data(flip_idx, data):
    data.targets[flip_idx] = 1.0 - data.targets[flip_idx]
    return data


def train_with_flipped(path, flip_idx=None, lr=0.01, epochs=30, save=False, portion=0.2, num_of_samples=10000,
                       log=True):
    rand_seed = 1234
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if flip_idx is None:
        flip_idx = np.load('{}/orders/random_flipping_order.npy'.format(path))[:int(portion * num_of_samples)]
    train_set, test_set = load_cifar_2()
    train_set = flip_data(flip_idx, train_set)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=100, shuffle=False, num_workers=1)

    time1 = time.time()
    project_root = config.project_root
    model = ResNet20(num_classes=1).to(device)
    model.load_state_dict(torch.load('{}/models/CNN/saved_models/base/model/ResNet-pretrained.pt'.format(project_root)))

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=0.9, weight_decay=5e-4)
    for epoch in range(epochs):
        train_loss, train_accuracy = train_ResNet(model, train_loader, optimizer, criterion, device)
        if (epoch + 1) % 10 == 0 and log:
            print('Epoch: {} | Train Loss: {:.3f} | Train Acc: {:.3f}% '.
                  format(epoch + 1, train_loss, 100 * train_accuracy))

    test_loss, test_accuracy = test_ResNet(model, test_loader, criterion, device)
    train_loss, train_accuracy = test_ResNet(model, train_loader, criterion, device)
    print('Train Loss: {:.3f} | Train Acc: {:.3f}% '.format(train_loss, 100.0 * train_accuracy))
    print('Test Loss: {:.3f} | Test Acc: {:.3f}% '.format(test_loss, 100.0 * test_accuracy))
    print('Training elapsed: {}'.format(time.time() - time1))
    if save:
        save_path = '{}/model'.format(path)
        torch.save(model.state_dict(), '{}/CNN.pt'.format(save_path))
        # no shuffle
        train_loader_no_shuffle = DataLoader(train_set, batch_size=100, shuffle=False, num_workers=1)
        intermediate_train, outputs_train, pred_train, labels_train = calculate_intermediate_output(model,
                                                                                                    train_loader_no_shuffle,
                                                                                                    device)
        intermediate_test, outputs_test, pred_test, labels_test = calculate_intermediate_output(model, test_loader,
                                                                                                device)

        np.savez('{}/saved_outputs'.format(save_path),
                 intermediate_train=intermediate_train, pred_train=pred_train, labels_train=labels_train,
                 intermediate_test=intermediate_test, pred_test=pred_test, labels_test=labels_test,
                 weight=model.fc.weight.data.cpu().numpy(), test_accuracy=test_accuracy,
                 train_accuracy=train_accuracy)
    return test_accuracy, model


def train_and_record_accuracies(path, method, lr=0.01,
                                epochs=50, save=False,
                                portion=0.2, num_of_samples=10000,
                                average_over_runs=2):
    test_accuracies = []
    inputs = np.load('{}/orders/order_{}.npz'.format(path, method), allow_pickle=True)['inputs']
    start = 0
    if os.path.exists('{}/accuracies/accuracies_temp_{}.npy'.format(path, method)):
        test_accuracies = list(np.load('{}/accuracies/accuracies_temp_{}.npy'.format(path, method)))
        start = len(test_accuracies)
    for count in range(start, 8):
        idx = inputs[count]
        test_accuracy = 0
        for i in range(average_over_runs):
            acc, _ = train_with_flipped(path, flip_idx=idx,
                                        lr=lr, epochs=epochs,
                                        save=save, portion=portion,
                                        num_of_samples=num_of_samples)
            test_accuracy += acc
        test_accuracies.append(test_accuracy / average_over_runs)
        count += 1
        np.save('{}/accuracies/accuracies_temp_{}'.format(path, method), test_accuracies)
    test_accuracies = np.stack(test_accuracies)
    np.save('{}/accuracies/accuracies_{}'.format(path, method), test_accuracies)
