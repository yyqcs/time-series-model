# Author: yongqiang yu
# URL: https://github.com/yyqcs/time-series-model
# Copyright (c) 2019, the Author
# All rights reserved.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from lstm_series_model import lstm_model
import pandas as pd
from visdom import Visdom
import matplotlib.pyplot as plt

# __all__ = ['VDE']

LR = 0.001
EPOCHS = 4
BATCH = 32
WINDOWS_LENGTH = 24
HIDDEN_SIZE = 64
PREDICTION_LENGTH = 24
TS_FEATURE = 1  # for univariate time series
RNN_INPUT_FEATURES = 8
TRAIN_PROP = 0.8


def get_samples(dataset, in_seq_len, pred_len):
    samples = []
    smin = dataset.min(axis=0)
    dataset = (dataset - smin) / (dataset.max(axis=0) - smin + 1e-6)
    step = 1
    for i in range(0, dataset.shape[0] - 1 - in_seq_len - pred_len, step):
        train_data = dataset[i:i + in_seq_len]
        target_data = dataset[i + in_seq_len:i + in_seq_len + pred_len]
        samples.append([train_data, target_data])

    return np.array(samples)


class get_dataset(Dataset):
    def __init__(self, values):
        self.raw_data = values
        self.length = self.raw_data.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        idx = index % self.length
        data = self.raw_data[idx]
        train_data = data[0].reshape(-1).astype(np.float32)
        target_data = data[1].reshape(-1).astype(np.float32)
        return train_data, target_data


class ts_model(object):
    """using lstm ,gru,rnn to model univariate time series.

    Parameters
    ----------
    wnd_len : int ,default=24
        the length of sliding window.The sequence in sliding window is the single input sequence of LSTM.
    pred_len : int ,default=24
        prediction sequence length
    net : str ,default=LSTM
        Net to model time series.Choices include 'LSTM' or 'RNN' or "GRU"
    rnn_input_size : int ,default=8
        embedding dimension,used to map input feature(for univariate,feature is one) to a high dimension
    rnn_hid_size : int ,default=64.
        the dimension of hidden layer in RNN,LSTM,GRU
    batch_size : int, default=32
        Batch size to use during SGD optimization.
    lr : float, default=1E-3
        Learning rate used for optimization.
    n_epochs : int, default=299
        Number of epochs to use during optimization.
    optimizer : str, default='Adam'
        Optimizer to use during SGD optimization. Choices include 'Adam' or 'SGD'.
    criterion : str, default='MSELoss'
        Prediction loss function to use.
    train_proportion : float,default=0.8
        the proportion of train data used to train model.The rest part is used for validation
    not_use_visdom : bool,default=True
        not use visdom to visualize the loss in training process,when False ensure run"python -m visdom.server" firstly
    cuda : bool, default=False
        Whether or not to use CUDA.
    dropout_rate : float, default=0.5.
        Dropout rate for hidden layers.
    verbose : bool, default=True
        Print out loss information.
    """

    # , train_data, test_data=None
    def __init__(self, wnd_len=WINDOWS_LENGTH, pred_len=PREDICTION_LENGTH,
                 net="LSTM", rnn_input_size=RNN_INPUT_FEATURES, rnn_hid_size=HIDDEN_SIZE,
                 batch_size=BATCH, lr=LR, n_epochs=EPOCHS, optimizer="Adam", criterion="MSELoss",
                 train_proportion=TRAIN_PROP,
                 not_use_visdom=True, cuda=True, dropout_rate=0.5, verbose=True):
        self.wnd_len = wnd_len
        self.pred_len = pred_len
        self.net = net
        self.rnn_input_size = rnn_input_size
        self.rnn_hid_size = rnn_hid_size
        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = n_epochs
        self.train_proportion = train_proportion
        self.not_use_visdom = not_use_visdom
        self.device = torch.device("cuda" if cuda else "cpu")
        self.dropout_rate = dropout_rate
        self.verbose = verbose
        self.model = lstm_model(1, self.rnn_input_size, self.rnn_hid_size,
                                self.pred_len, self.net, self.dropout_rate).to(self.device)
        if optimizer == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif optimizer == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            raise ValueError("Not a recognized optimizer")
        if criterion == "SmoothL1Loss":
            self.criterion = nn.SmoothL1Loss()
        elif criterion == "MSELoss":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError("Not a recognized loss function")
        self.is_fitted = False

        self.best_model = None
        self.min_loss = 999.99

    def __train(self, train_loader, val_loader):
        self.model.train()
        train_loss = []
        for train_batch_idx, (b_train, b_target) in enumerate(train_loader):
            b_train = b_train.to(self.device)
            b_target = b_target.to(self.device)
            # print("b_train shape={}".format(b_train.shape))
            out_seq = self.model(b_train)
            loss = self.criterion(b_target, out_seq)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss.append(loss.item())
        self.model.eval()
        with torch.no_grad():
            val_loss = []
            for test_batch_idx, (b_test, t_target) in enumerate(val_loader):
                b_test = b_test.to(self.device)
                t_target = t_target.to(self.device)
                t_output = self.model(b_test)
                t_loss = self.criterion(t_target, t_output)
                val_loss.append(t_loss.item())
            epoch__val_loss = sum(val_loss) / len(val_loss)
            if epoch__val_loss < self.min_loss:
                self.min_loss = epoch__val_loss
                self.best_model = self.model.state_dict()
            return sum(train_loss) / len(train_loss), epoch__val_loss

    def __eval(self, val_loader):
        self.model.eval()
        with torch.no_grad():
            val_loss = []
            for test_batch_idx, (b_test, t_target) in enumerate(val_loader):
                b_test = b_test.to(self.device)
                t_target = t_target.to(self.device)
                t_output = self.model(b_test)
                t_loss = self.criterion(t_target, t_output)
                val_loss.append(t_loss.item())

            epoch_loss = sum(val_loss) / len(val_loss)

            if epoch_loss < self.min_loss:
                self.min_loss = epoch_loss
                self.best_model = self.model.state_dict()
            return epoch_loss

    def fit(self, train_data):
        if not self.not_use_visdom:
            viz = Visdom()
            viz.line([[0., 0.]], [0.], win="train_loss", opts=dict(title="train loss ,val loss",
                                                                   legend=["train loss", "val loss"]))
        train_samples = int(train_data.shape[0] * self.train_proportion)
        train_loader = DataLoader(dataset=get_dataset(get_samples(train_data[:train_samples],
                                                                  self.wnd_len, self.pred_len)),
                                  batch_size=self.batch_size)
        val_loader = DataLoader(dataset=get_dataset(get_samples(train_data[train_samples:],
                                                                self.wnd_len, self.pred_len)),
                                batch_size=self.batch_size)
        for i in range(self.n_epochs):
            train_loss, val_loss = self.__train(train_loader, val_loader)
            if self.verbose:
                print('Epoch: {},train loss={},val loss={}'.format(i, train_loss, val_loss))
        self.is_fitted = True

    def transform(self, test_data):
        if self.is_fitted:
            test_min = test_data.min(axis=0)
            test_data = (test_data - test_min) / (test_data.max(axis=0) - test_min + 1e-6)
            preds = []
            reals = []
            data_len = test_data.shape[0]
            self.model.load_state_dict(self.best_model)
            self.model.eval()
            with torch.no_grad():
                for i in range(0, data_len, self.wnd_len):
                    # print("i={}".format(i))
                    # print("i + self.wnd_len + self.pred_len={}".format(i + self.wnd_len + self.pred_len))
                    # print("data_len={}".format(data_len))
                    if i + self.wnd_len + self.pred_len-1 < data_len:
                        print("我进来了")
                        interval_seq = test_data[i:i + self.wnd_len]
                        interval_seq = torch.from_numpy(interval_seq).float().to(self.device)
                        preds.extend(self.model(interval_seq).numpy().reshape(-1, 1))
                        reals.extend(test_data[i + self.wnd_len:
                                               i + self.wnd_len + self.pred_len - 1])
        else:
            raise RuntimeError("Model needs to be fit.")
        return preds, reals

    def fit_transform(self, train_data, test_data):
        self.fit(train_data)
        return self.transform(test_data)

    def save_best_model(self, save_path):
        torch.save(self.best_model, save_path)

    def plot_predict_result(self, preds, real):
        plt.figure(figsize=(16, 2 * 2))
        plt.plot(preds, color="y", label="test prediction")
        plt.plot(real, color='g', label="ground truth")
        plt.legend(loc="best")
        plt.show()


if __name__ == '__main__':
    train_file_path = r"D:\experiment\the_ip_bytes_hour.csv"
    data = pd.read_csv(train_file_path, header=0, index_col=0)["flows_sum"].values
    training = data[:-24]
    test = data[-48:]
    ts = ts_model()
    preds, reals = ts.fit_transform(training, test)
    ts.plot_predict_result(preds, reals)
