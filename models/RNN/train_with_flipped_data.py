import os
import random
import warnings

import numpy as np
import torch
import torch.optim as optim
from models.RNN.utils import create_fields, build_vocab
from torchtext import data
from torchtext import datasets

from models.RNN.RNN import train_RNN, evaluate_RNN, initialize_RNN, calculate_intermediate_output

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def flip_data(train_data, flipped_idx):
    train_data_copy = train_data
    for i in flipped_idx:
        if train_data_copy.examples[i].label == 'pos':
            train_data_copy.examples[i].label = 'neg'
        else:
            train_data_copy.examples[i].label = 'pos'
    return train_data_copy


def train_with_data(epochs, model, train_iterator, test_iterator, lr, path):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    save_path = '{}/model'.format(path)
    for epoch in range(epochs):
        train_loss, train_acc = train_RNN(model, train_iterator, optimizer, criterion)
        torch.save(model.state_dict(), '{}/sentiment-model-temp.pt'.format(save_path))
        if (epoch + 1) % 5 == 0:
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')

    model.load_state_dict(torch.load('{}/sentiment-model-temp.pt'.format(save_path)))
    print("----------finished training----------")
    train_loss, train_acc = evaluate_RNN(model, train_iterator, criterion)
    test_loss, test_acc = evaluate_RNN(model, test_iterator, criterion)

    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')

    return train_acc, test_acc, model


def train_sentiment_model_with_flipped_data(path, flip_idx=None, epochs=20, lr=1e-3, save=False,
                                            portion=0.2, num_of_samples=17500, use_pretrained=True):
    if save:
        random_seed = 1234
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
    print('------------------preparing data----------------------')
    # prepare data
    TEXT, LABEL = create_fields()
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    train_data, valid_data = train_data.split(random_state=random.seed(1234))

    if flip_idx is None:
        flip_idx = np.load('{}/orders/random_flipping_order.npy'.format(path))[:int(portion * num_of_samples)]
        save = True
    flip_data(train_data=train_data, flipped_idx=flip_idx)

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
    train_acc, test_acc, model = train_with_data(epochs=epochs, model=model,
                                                 train_iterator=train_iterator,
                                                 test_iterator=test_iterator, lr=lr,
                                                 path=path)

    if save:
        save_path = '{}/model'.format(path)
        torch.save(model.state_dict(), '{}/sentiment-model.pt'.format(save_path))
        intermediate_train, pred_train, labels_train = calculate_intermediate_output(model, train_data, TEXT, LABEL,
                                                                                     device)
        intermediate_test, pred_test, labels_test = calculate_intermediate_output(model, train_data, TEXT, LABEL,
                                                                                  device)
        np.savez('{}/model/saved_outputs'.format(path),
                 intermediate_train=intermediate_train, pred_train=pred_train, labels_train=labels_train,
                 intermediate_test=intermediate_test, pred_test=pred_test, labels_test=labels_test,
                 weight=model.fc.weight.data.cpu().numpy(), train_acc=train_acc, test_acc=test_acc)
    return test_acc


def train_and_record_accuracies(path, method, lr=1e-3,
                                epochs=20, save=False,
                                portion=0.2, num_of_samples=10000,
                                average_over_runs=2, use_pretrained=True):
    test_accuracies = []
    inputs = np.load('{}/orders/order_{}.npz'.format(path, method), allow_pickle=True)['inputs']
    start = 0
    if os.path.exists('{}/accuracies/accuracies_temp_{}.npy'.format(path, method)):
        test_accuracies = list(np.load('{}/accuracies/accuracies_temp_{}.npy'.format(path, method)))
        start = len(test_accuracies)
    for count in range(start, 8):
        idx = inputs[count]
        torch.cuda.empty_cache()
        print('training for {}% checked'.format(5 * count))
        test_accuracy = 0
        for i in range(average_over_runs):
            acc = train_sentiment_model_with_flipped_data(path, flip_idx=idx,
                                                          lr=lr, epochs=epochs,
                                                          save=save, portion=portion,
                                                          num_of_samples=num_of_samples,
                                                          use_pretrained=use_pretrained)
            test_accuracy += acc
        test_accuracies.append(test_accuracy / average_over_runs)
        count += 1
        np.save('{}/accuracies/accuracies_temp_{}'.format(path, method), test_accuracies)
    test_accuracies = np.stack(test_accuracies)
    np.save('{}/accuracies/accuracies_{}'.format(path, method), test_accuracies)
