import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.CNN.CNN import ResNet20, train_ResNet, test_ResNet, calculate_intermediate_output
from utils_cifar import load_cifar_2


def main(args):
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_set, test_set = load_cifar_2()
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=100, shuffle=False, num_workers=1)

    time1 = time.time()
    # Model
    print('==> Building model..')

    # load pretrained weights
    if os.path.exists('model/ResNet-pretrained.pt'.format(args.saved_path)):
        model = ResNet20(num_classes=1)
        model.load_state_dict(torch.load('model/ResNet-pretrained.pt'.format(args.saved_path)))
    else:
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
        model.fc = torch.nn.Linear(64, 1)
        torch.save(model.state_dict(), 'model/ResNet-pretrained.pt'.format(args.saved_path))
    model.train()
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(args.epochs):
        train_loss, train_accuracy = train_ResNet(model, train_loader, optimizer, criterion, device)
        if (epoch + 1) % 5 == 0:
            print('Epoch: {} | Train Loss: {:.3f} | Train Acc: {:.3f}% '.
                  format(epoch + 1, train_loss, 100 * train_accuracy))

    test_loss, test_accuracy = test_ResNet(model, test_loader, criterion, device)
    print('Test Loss: {:.3f} | Test Acc: {:.3f}% '.format(test_loss, 100.0 * test_accuracy))
    print('Training elapsed: {}'.format(time.time() - time1))
    torch.save(model.state_dict(), '{}/model/CNN.pt'.format(args.saved_path))

    model.load_state_dict(torch.load('{}/model/CNN.pt'.format(args.saved_path)))
    train_set, test_set = load_cifar_2()
    train_loader_no_shuffle = DataLoader(train_set, batch_size=100, shuffle=False, num_workers=1, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=100, shuffle=False, num_workers=1, drop_last=True)
    intermediate_train, outputs_train, pred_train, labels_train = calculate_intermediate_output(model,
                                                                                                train_loader_no_shuffle,
                                                                                                device)
    intermediate_test, outputs_test, pred_test, labels_test = calculate_intermediate_output(model, test_loader, device)

    np.savez('{}/model/saved_outputs'.format(args.saved_path),
             intermediate_train=intermediate_train, pred_train=pred_train,
             labels_train=labels_train, outputs_train=outputs_train,
             intermediate_test=intermediate_test, pred_test=pred_test, labels_test=labels_test,
             outputs_test=outputs_test, weight=model.fc.weight.data.cpu().numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ResNet Training')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--epochs', default=0, type=int, help='training epochs')
    parser.add_argument('--saved_path', default='saved_models/base')
    args = parser.parse_args()
    main(args)
