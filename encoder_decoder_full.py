# Author: yongqiang yu
# URL: https://github.com/yyqcs/time-series-model
# Copyright (c) 2019, the Author
# All rights reserved.

import torch.nn as nn
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from visdom import Visdom
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

torch.manual_seed(1234)
np.random.seed(1234)

LR = 0.001
EPOCHS = 5  # 5
BATCH = 256
WINDOWS_LENGTH = 24
HIDDEN_SIZE = 64
PREDICTION_LENGTH = 24
TS_FEATURE = 1
RNN_INPUT_FEATURES = 8
TRAIN_PROP = 0.8
SILD_STEP = 1

device = torch.device("cuda")


def generate_sample(dataset, in_seq_len, out_seq_len, slid_step=1):
    samples = []
    step = slid_step
    vaild_len = dataset.shape[0] - in_seq_len - out_seq_len + 1
    for i in range(0, vaild_len, step):
        time_step = dataset[i:i + in_seq_len, 0]
        enc_seqs = dataset[i:i + in_seq_len, 1]
        dec_output_seqs = dataset[i + in_seq_len:i + in_seq_len + out_seq_len, 1]
        samples.append([time_step, enc_seqs, dec_output_seqs])
    return np.array(samples)


# 计算RMSE
def calc_rmse(real, pred):
    return np.sqrt(mean_squared_error(real, pred))


# 计算MAE
def calc_mae(real, pred):
    return mean_absolute_error(real, pred)


# 计算MAPE
def calc_mape(real, pred, epsion=1E-7):
    real += epsion
    return np.sum(np.abs((real - pred) / real)) / len(real) * 100


# 计算SMAPE
def calc_smape(real, pred):
    delim = (np.abs(real) + np.abs(pred)) / 2.0
    return np.sum(np.abs((real - pred) / delim)) / len(real) * 100


class get_set(Dataset):
    def __init__(self, values):
        self.raw_data = values
        self.length = self.raw_data.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # (batch,seq_len)
        idx = index % self.length
        data = self.raw_data[idx]
        time_step = data[0].astype(np.float32)
        enc_seqs = data[1].astype(np.float32)
        dec_output_seqs = data[2].astype(np.float32)
        return time_step, enc_seqs, dec_output_seqs


class mlp_attention(nn.Module):
    def __init__(self, q_size, k_size, attn_size, bias=True, dropout=0.):
        super(mlp_attention, self).__init__()
        self.Wq = nn.Linear(q_size, attn_size, bias=bias)
        self.Wk = nn.Linear(k_size, attn_size, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        q = self.Wq(q)  # (B,S_q,H_q),-->(B,S_q,H_a)
        k = self.Wk(k)  # (B,S_k,H_k)-->(B,S-k,H_a)
        scores = torch.matmul(q, k.transpose(-2, -1))
        weight = self.dropout(torch.softmax(scores, dim=-1))
        # (B, S_q, S_k) dot (B, S_k, S_v) -> (B, S_q, S_v)
        atten_value = torch.bmm(weight, v)
        return atten_value, weight


class seq2seq_pred(nn.Module):
    def __init__(self, ts_features, rnn_intput_featuers, hidden_size, out_seq_len, use_pe=False, use_atten=False):
        super(seq2seq_pred, self).__init__()
        self.input_size = ts_features
        self.hidden_size = hidden_size
        self.out_seq_len = out_seq_len
        self.rnn_inpt_size = rnn_intput_featuers
        self.use_pe = use_pe
        self.use_attention = use_atten

        self.embed = nn.Linear(self.input_size, self.rnn_inpt_size)
        self.encoder = nn.LSTMCell(self.rnn_inpt_size, self.hidden_size)
        self.out_linear = nn.Linear(self.hidden_size, 1)
        if self.use_attention:
            self.attention = mlp_attention(hidden_size, hidden_size, hidden_size)
            self.decoder = nn.LSTMCell(self.hidden_size + self.hidden_size, self.hidden_size)
        else:
            self.decoder = nn.LSTMCell(self.hidden_size, self.hidden_size)
        if self.use_pe:
            self.time_embed = nn.Linear(self.input_size, self.rnn_inpt_size)

    def forward(self, enc_inputs, timestep=None):
        enc_inputs = self.embed(enc_inputs.unsqueeze(-1))
        if self.use_pe:
            enc_inputs = enc_inputs + self.time_embed(timestep.unsqueeze(-1))

        batch = enc_inputs.shape[0]
        in_hx, in_cx = torch.zeros(batch, self.hidden_size).to(device), torch.zeros(batch, self.hidden_size).to(device)
        out_seqs = []
        in_hxs = []
        for in_step in range(enc_inputs.shape[1]):
            in_hx, in_cx = self.encoder(enc_inputs[:, in_step, :], (in_hx, in_cx))
            if self.use_attention:
                in_hxs.append(in_hx)
        out_hx = in_hx
        out_cx = in_cx
        if self.use_attention:
            in_hxs = torch.stack(in_hxs, dim=1)  # 时刻维度增加。(batch,seq_len,features)
        for _ in range(self.out_seq_len):  # 要从循环的角度理解。
            if self.use_attention:
                context, _ = self.attention(out_hx.unsqueeze(1), in_hxs, in_hxs)
                dec_input_i = torch.cat((out_hx, context.squeeze(1)), dim=-1)
            else:
                dec_input_i = out_hx
            out_hx, out_cx = self.decoder(dec_input_i, (out_hx, out_cx))
            out_seqs.append(out_hx)

        outs = self.out_linear(torch.stack(out_seqs, dim=1)).squeeze()  # 每个时刻都有输出。NOT最后一个时刻的拼接
        return outs

    def batch_train(self, ):
        pass

    def plot_predict_result(self, preds, real):
        plt.figure(figsize=(16, 2 * 2))
        plt.plot(preds, color="r", label="test prediction")
        plt.plot(real, color='b', label="ground truth")
        plt.legend(loc="best")
        plt.show()

    def cal_metrics(self, real, pred):
        real = np.array(real)
        pred = np.array(pred)
        return calc_rmse(real, pred), calc_mae(real, pred), calc_mape(real, pred), calc_smape(real, pred)


if __name__ == '__main__':
    viz = Visdom()
    viz.line([[0., 0.]], [0.], win="train_loss",
             opts=dict(title="train loss ,val loss", legend=["train loss", "val loss"]))
    train_file_path = "pollution.csv"
    training = pd.read_csv(train_file_path).values
    if training.ndim == 1:
        training = training.reshape(-1, 1)
    scaler = MinMaxScaler()
    training = scaler.fit_transform(training)
    train_samples = int(training.shape[0] * 0.8)
    test_data = training[train_samples:]
    train_data = generate_sample(training[:train_samples], WINDOWS_LENGTH, PREDICTION_LENGTH)
    trainer, validation = train_test_split(train_data, train_size=0.8)
    train_loader = DataLoader(dataset=get_set(trainer), batch_size=BATCH)
    val_loader = DataLoader(dataset=get_set(validation), batch_size=BATCH)
    model = seq2seq_pred(TS_FEATURE, RNN_INPUT_FEATURES, HIDDEN_SIZE, PREDICTION_LENGTH,
                         use_pe=False, use_atten=False).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    best_model = None
    min_loss = 999.99
    for epoch in range(0, EPOCHS):
        model.train()
        train_loss = []
        for train_batch_idx, (t_time, b_enc, b_dec_out) in enumerate(train_loader):
            t_time, b_enc, b_dec_out = t_time.to(device), b_enc.to(device), b_dec_out.to(device)
            # no position encoding and no attention
            out_seq = model(b_enc)

            # # position encoding and no attention
            # out_seq = model(b_enc, t_time)

            # # No position encoding and Yes attention
            # out_seq = model(b_enc)

            # Yes position encoding and Yes attention
            # out_seq = model(b_enc, t_time)

            loss = criterion(b_dec_out, out_seq)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        model.eval()
        with torch.no_grad():
            val_loss = []
            for test_batch_idx, (e_time, b_enc, b_dec_out) in enumerate(val_loader):
                e_time, b_enc, b_dec_out = e_time.to(device), b_enc.to(device), b_dec_out.to(device)
                out_seq = model(b_enc)
                t_loss = criterion(b_dec_out, out_seq)
                val_loss.append(t_loss.item())

            epoch_loss = sum(val_loss) / len(val_loss)
            if epoch_loss < min_loss:
                min_loss = epoch_loss
                best_model = model.state_dict()
            if epoch % 20 == 0:
                viz.line([[sum(train_loss) / len(train_loss)], [sum(val_loss) / len(val_loss)]],
                         [epoch], win="train_loss", update="append")
                print("epoch {:0>2},train_loss={:.5f},val_loss={:.5f}".format(epoch, sum(train_loss) / len(train_loss),
                                                                              sum(val_loss) / len(val_loss)))
    print("min val_loss={:.5f}".format(min_loss))
    torch.save(best_model, "./model_stat_val_loss_{}.pth".format(min_loss))

    with torch.no_grad():
        model.load_state_dict(best_model)
        test_len = test_data.shape[0]
        print("test_data[:2]={}".format(test_data[:2]))
        reals = []
        preds = []
        for i in range(0, test_len, WINDOWS_LENGTH):
            if i + WINDOWS_LENGTH + PREDICTION_LENGTH - 1 < test_len:
                interval_time, interval_seq = test_data[i:i + WINDOWS_LENGTH, 0], test_data[i:i + WINDOWS_LENGTH, 1]
                interval_time = torch.from_numpy(interval_time).float().unsqueeze_(0).to(device)  # batch,seq_len
                interval_seq = torch.from_numpy(interval_seq).float().unsqueeze_(0).to(device)
                interval_preds = model(interval_seq).cpu().numpy().reshape(-1, 1)
                # interval_preds = model(interval_time, interval_seq).cpu().numpy().reshape(-1, 1)
                interval_reals = test_data[i + WINDOWS_LENGTH:i + WINDOWS_LENGTH + PREDICTION_LENGTH, 1]
                interval_time = test_data[i + WINDOWS_LENGTH:i + WINDOWS_LENGTH + PREDICTION_LENGTH, 0]
                # 可以优化！！！
                interval_preds = np.hstack((interval_time.reshape(-1, 1), interval_preds))
                interval_reals = np.hstack((interval_time.reshape(-1, 1), interval_reals.reshape(-1, 1)))
                interval_preds = scaler.inverse_transform(interval_preds)
                interval_reals = scaler.inverse_transform(interval_reals)
                preds.extend(interval_preds[:, 1])
                reals.extend(interval_reals[:, 1])
        model.plot_predict_result(preds, reals)
        matrics = model.cal_metrics(reals, preds)
        print("test RMSE={:.5f},MAE={:.5f},MAPE={:.5f},SMAPE={:.5f}".format(matrics[0], matrics[1], matrics[2],
                                                                            matrics[3]))
