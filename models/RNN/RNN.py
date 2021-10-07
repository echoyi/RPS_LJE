# modified from https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/2%20-%20Upgraded%20Sentiment%20Analysis.ipynb
import spacy
import torch
import torch.nn as nn


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 2, output_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths, return_hidden=False):
        # text = [sent len, batch size]

        embedded = self.dropout(self.embedding(text))

        # embedded = [sent len, batch size, emb dim]

        # pack sequence
        # lengths need to be on CPU!
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'))

        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        # output = [sent len, batch size, hid dim * num directions]
        # output over padding tokens are zero tensors

        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout

        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        # hidden = [batch size, hid dim * num directions]
        if return_hidden:
            return hidden, self.fc(hidden)
        return self.fc(hidden)

    def predict_sentiment(self, sentence, TEXT, device):
        nlp = spacy.load('en_core_web_sm')
        self.eval()
        tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
        indexed = [TEXT.vocab.stoi[t] for t in tokenized]
        length = [len(indexed)]
        tensor = torch.LongTensor(indexed).to(device)
        tensor = tensor.unsqueeze(1)
        length_tensor = torch.LongTensor(length)
        prediction = torch.sigmoid(self.forward(tensor, length_tensor))
        return prediction.item()


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def train_RNN(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        text, text_lengths = batch.text

        predictions = model(text, text_lengths).squeeze(1)

        loss = criterion(predictions, batch.label)

        acc = binary_accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate_RNN(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text

            predictions = model(text, text_lengths).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def initialize_RNN(TEXT, use_pretrained=True):
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    # initialize with pretrained
    if use_pretrained:
        pretrained_embeddings = TEXT.vocab.vectors
        # print(pretrained_embeddings.shape)
        model.embedding.weight.data.copy_(pretrained_embeddings)
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
    return model


def calculate_intermediate(model, text, text_lengths):
    embedded = model.dropout(model.embedding(text))
    packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'))
    packed_output, (hidden, cell) = model.rnn(packed_embedded)
    hidden = model.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
    return hidden.data


def calculate_intermediate_output(model, data_set, TEXT, LABEL, device):
    text_pipeline = lambda x: [TEXT.vocab.stoi[t] for t in x]
    label_pipeline = lambda x: LABEL.vocab.stoi[x]
    model.eval()
    intermediate = []
    labels = []
    pred = []
    count = 0
    for example in data_set:
        count += 1
        if count % 3000 == 0:
            print("predicting for {}th training data:".format(count))
        label = torch.tensor(label_pipeline(example.label)).to(device)
        text = text_pipeline(example.text)
        text_tensor = torch.LongTensor(text).to(device).unsqueeze(1)
        length_tensor = torch.LongTensor([len(text)])
        labels.append(label)
        intermediate.append(calculate_intermediate(model, text_tensor, length_tensor))
        pred.append(torch.sigmoid(model(text_tensor, length_tensor)))
    intermediate = torch.stack(intermediate).data
    labels = torch.stack(labels).data
    pred = torch.stack(pred).data
    return to_np(intermediate), to_np(pred), to_np(labels)


def to_np(x):
    return x.cpu().data.numpy()
