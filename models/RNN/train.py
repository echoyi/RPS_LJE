# modified from https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/2%20-%20Upgraded%20Sentiment%20Analysis.ipynb
import argparse
import random
import time
import warnings

import numpy as np
import torch
import torch.optim as optim
from torchtext import data
from torchtext import datasets

import config
from models.RNN.RNN import train_RNN, evaluate_RNN, initialize_RNN, calculate_intermediate_output
from models.RNN.utils_imdb import create_fields, build_vocab

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train_with_data(epochs, model, train_iterator, valid_iterator, test_iterator, lr, path):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    criterion = criterion.to(device)
    best_valid_loss = float('inf')
    print('------------------Training----------------------')

    for epoch in range(epochs):

        start_time = time.time()

        train_loss, train_acc = train_RNN(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate_RNN(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), '{}/model/sentiment-model.pt'.format(path))
        if (epochs + 1) % 10 == 0:
            print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

    model.load_state_dict(torch.load('{}/model/sentiment-model.pt'.format(path)))

    print("----------finished training----------")
    train_loss, train_acc = evaluate_RNN(model, train_iterator, criterion)
    valid_loss, valid_acc = evaluate_RNN(model, valid_iterator, criterion)
    test_loss, test_acc = evaluate_RNN(model, test_iterator, criterion)

    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')
    return train_acc, test_acc


def train_sentiment_model(path, epochs, use_pretrained, lr=5e-4):
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    torch.backends.cudnn.deterministic = True
    print('------------------preparing data----------------------')
    # prepare data
    TEXT, LABEL = create_fields()
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL, root='D:{}/.data'.format(config.project_root))
    train_data, valid_data = train_data.split(random_state=random.seed(1234))
    print('------------------building vocab---------------------')
    # build vocab
    build_vocab(train_data, TEXT, LABEL)

    # train, valid, test data
    BATCH_SIZE = 64
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        sort_within_batch=True,
        device=device)

    # initialize model
    model = initialize_RNN(TEXT, use_pretrained=use_pretrained)

    model = model.to(device)
    train_acc, test_acc = train_with_data(epochs, model, train_iterator, valid_iterator, test_iterator, lr, path)

    intermediate_train, pred_train, labels_train = calculate_intermediate_output(model, train_data, TEXT, LABEL, device)
    intermediate_test, pred_test, labels_test = calculate_intermediate_output(model, test_data, TEXT, LABEL, device)
    np.savez('{}/model/saved_outputs'.format(path),
             intermediate_train=intermediate_train, pred_train=pred_train, labels_train=labels_train,
             intermediate_test=intermediate_test, pred_test=pred_test, labels_test=labels_test,
             weight=model.fc.weight.data.cpu().numpy(), train_acc=train_acc, test_acc=test_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LSTM Training')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epochs', default=10, type=int, help='training epochs')
    parser.add_argument('--saved_path', default='saved_models/base')
    parser.add_argument('--use_pretrained', default=True)
    args = parser.parse_args()
    train_sentiment_model(args.saved_path, epochs=args.epochs, use_pretrained=args.use_pretrained, lr=args.lr)
