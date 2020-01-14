import torch.nn as nn
import torch
import torch.optim as optim
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

LR = 0.01
EPOCHS = 3  # 512
BATCH = 256
WINDOWS_LENGTH = 24
HIDDEN_SIZE = 128
PREDICTION_LENGTH = 24
TS_FEATURE = 1
RNN_INPUT_FEATURES = 64
TRAIN_PROP = 0.8
SILD_STEP = 1



device = torch.device("cuda:0")

# f = open("record.txt", "w")

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
    def __init__(self, q_size, k_size, attn_size, bias=True, dropout=0.3):
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
    def __init__(self, ts_features, rnn_inpt_size, hidden_size, out_seq_len, net="LSTM", use_pe=False,
                 use_atten=False, use_seq2seq=False, drop=0.3):
        super(seq2seq_pred, self).__init__()
        self.input_size = ts_features
        self.hidden_size = hidden_size
        self.out_seq_len = out_seq_len
        self.net=net
        self.rnn_inpt_size = rnn_inpt_size
        self.use_pe = use_pe
        self.use_attention = use_atten
        self.dropout = nn.Dropout(drop)
        self.embed = nn.Linear(self.input_size, self.rnn_inpt_size)
        self.encoder = nn.LSTMCell(self.rnn_inpt_size, self.hidden_size)
        self.out_linear = nn.Linear(self.hidden_size, 1)
        self.output_layer=nn.Linear(self.hidden_size,self.out_seq_len)
        if self.use_attention:
            self.attention = mlp_attention(hidden_size, hidden_size, hidden_size, dropout=drop)
            self.decoder = nn.LSTMCell(self.hidden_size + self.hidden_size, self.hidden_size)
        else:
            self.decoder = nn.LSTMCell(self.hidden_size, self.hidden_size)
        if self.use_pe:
            self.time_embed = nn.Linear(self.input_size, self.rnn_inpt_size)
        self.use_seq2seq = use_seq2seq
        if self.net=="LSTM":
            self.hidden_layer = nn.LSTMCell(self.rnn_inpt_size, self.hidden_size)
        elif self.net=="GRU":
            self.hidden_layer=nn.GRUCell(self.rnn_inpt_size, self.hidden_size)#复制时注意参数的修改！！
        elif self.net=="RNN":
            self.hidden_layer=nn.RNNCell(self.rnn_inpt_size,self.hidden_size)
        else:
            raise ValueError("Not a recognized net")

        # self.time_embed=self.activation()

    def forward(self, enc_inputs, timestep=None):
        enc_inputs = self.embed(enc_inputs.unsqueeze(-1))
        if self.use_pe:
            enc_inputs = enc_inputs + self.time_embed(timestep.unsqueeze(-1))
        enc_inputs = self.dropout(enc_inputs)
        batch = enc_inputs.shape[0]
        in_hx, in_cx = torch.zeros(batch, self.hidden_size).to(device), torch.zeros(batch, self.hidden_size).to(device)
        out_seqs = []
        in_hxs = []
        for in_step in range(enc_inputs.shape[1]):
            if self.use_seq2seq:
                in_hx, in_cx = self.encoder(enc_inputs[:, in_step, :], (in_hx, in_cx))
                if self.use_attention:
                    in_hxs.append(in_hx)
            elif self.net=="LSTM":
                in_hx, in_cx = self.hidden_layer(enc_inputs[:, in_step, :], (in_hx, in_cx))
            else:
                in_hx = self.hidden_layer(enc_inputs[:, in_step, :],in_hx)

        if self.use_seq2seq:
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
        else:
            outs = self.output_layer(in_hx)
        if outs.dim()==1:
          outs=outs.view(-1,1)
        return outs


class ts_seq2seq(object):
    def __init__(self, ts_features=TS_FEATURE, rnn_input_size=RNN_INPUT_FEATURES, hidden_size=HIDDEN_SIZE,
                 wnd_len=WINDOWS_LENGTH, pred_len=PREDICTION_LENGTH,net="LSTM", batch_size=BATCH,
                 lr=LR, n_epochs=EPOCHS,train_prop=0.8, step_size=128, gamma=0.1, dropout=0.3,
                 use_pe=False, use_atten=False, use_seq2seq=False,cuda=True, not_use_visdom=True, verbose=True):
        self.input_size = ts_features
        self.rnn_input_size = rnn_input_size
        self.hidden_size = hidden_size
        self.wnd_len = wnd_len
        self.pred_len = pred_len
        self.net=net
        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = n_epochs
        self.dropout = dropout
        self.use_pe = use_pe
        self.use_atten = use_atten
        self.use_seq2seq=use_seq2seq
        self.device = torch.device("cuda" if cuda else "cpu")
        self.not_use_visdom = not_use_visdom
        self.verbose = verbose
        self.model = seq2seq_pred(self.input_size, self.rnn_input_size, self.hidden_size, self.pred_len,net=self.net,
                                  use_pe=self.use_pe, use_atten=self.use_atten,
                                  use_seq2seq=self.use_seq2seq,drop=dropout).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.dropout = nn.Dropout(0.5)
        self.criterion = nn.MSELoss()
        self.train_prop = train_prop
        self.is_fitted = False
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        self.best_model = None
        self.min_loss = 999.99
        if not self.not_use_visdom:
            self.viz = Visdom()
            self.viz.line([[0., 0.]], [0.], win="train_loss",
                          opts=dict(title="train loss ,val loss", legend=["train loss", "val loss"]))

    def __train(self, train_loader, val_loader):
        self.model.train()
        train_loss = []
        for train_batch_idx, (t_time, b_enc, b_dec_out) in enumerate(train_loader):
            t_time, b_enc, b_dec_out = t_time.to(device), b_enc.to(device), b_dec_out.to(device)
            if self.use_pe:
                out_seq = self.model(b_enc, t_time)
            else:
                out_seq = self.model(b_enc)

            loss = self.criterion(b_dec_out, out_seq)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss.append(loss.item())
        epoch_val_loss = self.__eval(val_loader)
        return sum(train_loss) / len(train_loss), epoch_val_loss

    def __eval(self, test_loader):
        self.model.eval()
        with torch.no_grad():
            val_loss = []
            for test_batch_idx, (v_time, v_enc, v_dec_out) in enumerate(test_loader):
                v_time, v_enc, v_dec_out = v_time.to(device), v_enc.to(device), v_dec_out.to(device)
                if self.use_pe:
                    v_out_seq = self.model(v_enc, v_time)
                else:
                    v_out_seq = self.model(v_enc)
                t_loss = self.criterion(v_dec_out, v_out_seq)
                val_loss.append(t_loss.item())
            epoch_val_loss = sum(val_loss) / len(val_loss)
            if epoch_val_loss < self.min_loss:
                self.min_loss = epoch_val_loss
                self.best_model = self.model.state_dict()
        return epoch_val_loss

    def __predict(self, test_data):
        """
        :param test_data: un-scaler ,the columns of are time,value
        :return: real,pred
        """
        reals = []
        preds = []
        test_len = test_data.shape[0]
        # scaled_test_data = self.__standilze(test_data)
        self.model.eval()
        with torch.no_grad():
            for i in range(0, test_len, self.pred_len):
                if i + self.wnd_len + self.pred_len - 1 < test_len:
                    interval_seq = test_data[i:i + self.wnd_len, 1]
                    interval_seq = torch.from_numpy(interval_seq).float().unsqueeze_(0).to(device)
                    if self.use_pe:
                        interval_time = test_data[i:i + self.wnd_len, 0]
                        interval_time = torch.from_numpy(interval_time).float().unsqueeze_(0).to(device)
                        interval_preds = self.model(interval_seq, interval_time).cpu().numpy().reshape(-1, 1)
                    else:
                        interval_preds = self.model(interval_seq).cpu().numpy().reshape(-1, 1)
                    interval_reals = test_data[i + self.wnd_len:i + self.wnd_len + self.pred_len, 1]
                    preds.extend(interval_preds)
                    reals.extend(interval_reals)
        return np.array(reals).reshape(-1, 1), np.array(preds).reshape(-1, 1)

    def fit(self, train_data):
        """
        :param train_data:  origin data ,cols：time,value
        :return: is_fitted: bool
        """
        train_data = generate_sample(train_data, self.wnd_len, self.pred_len)
        trainer, validation = train_test_split(train_data, train_size=self.train_prop)
        train_loader = DataLoader(dataset=get_set(trainer), batch_size=self.batch_size)
        val_loader = DataLoader(dataset=get_set(validation), batch_size=self.batch_size)
        for i in range(self.n_epochs):
            train_loss, val_loss = self.__train(train_loader, val_loader)
            if self.verbose and i % 20 == 0:
                print('Epoch: {},train loss={},val loss={}'.format(i, train_loss, val_loss))
                # f.write('Epoch: {},train loss={},val loss={}\n'.format(i, train_loss, val_loss))
            if not self.not_use_visdom:
                self.viz.line([[train_loss], [val_loss]],
                              [i], win="train_loss", update="append")
            self.scheduler.step()
        self.is_fitted = True
        return self.is_fitted

    def transform(self, test_data):
        """
        :param test_data: origin data ,cols：time,value
        :return:real,pred
        """
        if self.is_fitted:
            self.model.load_state_dict(self.best_model)
            real, pred = self.__predict(test_data)
        else:
            raise RuntimeError("Model needs to be fit.")
        return real, pred

    def fit_transform(self, train_data, test_data):
        self.fit(train_data)
        return self.transform(test_data)

    def save_best_model(self, save_path):
        torch.save(self.best_model, save_path)

    def get_min_loss(self):
        return self.min_loss

    def calc_rmse(self, real, pred):
        return np.sqrt(mean_squared_error(real, pred))

    def calc_mae(self, real, pred):
        return mean_absolute_error(real, pred)

    def calc_mape(self, real, pred, epsion=1E-7):
        real += epsion
        return np.sum(np.abs((real - pred) / real)) / len(real) * 100

    def calc_smape(self, real, pred):
        delim = (np.abs(real) + np.abs(pred)) / 2.0
        return np.sum(np.abs((real - pred) / delim)) / len(real) * 100

    def cal_metrics(self, real, pred, show=False):
        metricx = {}
        real = np.asarray(real)
        pred = np.asarray(pred)
        metricx["RMSE"] = self.calc_rmse(real, pred)
        metricx["MAE"] = self.calc_mae(real, pred)
        metricx["MAPE"] = self.calc_mape(real, pred)
        metricx["SMAPE"] = self.calc_smape(real, pred)
        if show:
            # f.write("RMSE={:.5f},MSE={:.5f},MAPE={:.5f},SMAPE={:.5f}\n".
            #         format(metricx["RMSE"], metricx["MSE"], metricx["MAPE"], metricx["SMAPE"]))
            print("RMSE={:.5f},MAE={:.5f},MAPE={:.5f},SMAPE={:.5f}".
                  format(metricx["RMSE"], metricx["MAE"], metricx["MAPE"], metricx["SMAPE"]))
        return metricx

    def plot_prediction(self, real, pred, filename=None):
        plt.figure(figsize=(16, 2 * 2))
        plt.plot(real, color="b", label="ground truth")  # model_eval prediction
        plt.plot(pred, color='r', label="model_eval prediction")  # ground truth
        plt.legend(loc="best")
        if filename is not None:
            plt.savefig(filename + ".svg")
        # plt.show()


def model_eval(net,use_pe, use_atten, use_seq2seq,data_path):
    filename = data_path.split(".")[0] + "_"
    filename+=net+"_"
    filename += "PE_" if use_pe else "NO_PE_"
    filename += "ATTENTION" if use_atten else "NO_ATTENTION"
    # f.write(filename + "\n")
    print(filename)
    train_file_path = data_path
    data = pd.read_csv(train_file_path).values
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    train_samples = int(data.shape[0] * 0.8)
    train_data, test_data = data[:train_samples], data[train_samples:]
    ts_model = ts_seq2seq(wnd_len=WINDOWS_LENGTH, pred_len=PREDICTION_LENGTH,
                          net=net,use_pe=use_pe, use_atten=use_atten,use_seq2seq=use_seq2seq)
    real, pred = ts_model.fit_transform(train_data, test_data)
    real = scaler.inverse_transform(np.hstack((real, real)))[:, 1]
    pred = scaler.inverse_transform(np.hstack((pred, pred)))[:, 1]
    print("mimimum val_loss={}".format(ts_model.get_min_loss()))
    # f.write("mimimum val_loss={}\n".format(ts_model.get_min_loss()))
    ts_model.cal_metrics(real, pred, show=True)
    ts_model.save_best_model(filename + ".pth")
    ts_model.plot_prediction(real, pred, filename)
    # f.write("\n")
    print()
    # f.write("-" * 89 + "\n")
    print("-" * 89)


def test_all(filename):
    model_eval(net="LSTM",use_pe=False, use_atten=False, use_seq2seq=False,data_path=filename)
    model_eval(net="LSTM",use_pe=True, use_atten=False, use_seq2seq=False,data_path=filename)
    model_eval(net="GRU",use_pe=False, use_atten=False, use_seq2seq=False,data_path=filename)
    model_eval(net="GRU",use_pe=True, use_atten=False, use_seq2seq=False,data_path=filename)
    model_eval(net="RNN",use_pe=False, use_atten=False, use_seq2seq=False,data_path=filename)
    model_eval(net="RNN",use_pe=True, use_atten=False, use_seq2seq=False,data_path=filename)



if __name__ == '__main__':
    test_all(r"D:\PycharmProjects\TS\DualAttentionSeq2Seq\ts_model\pollution.csv")
    test_all(r"D:\PycharmProjects\TS\DualAttentionSeq2Seq\ts_model\bike_hour.csv")
    test_all(r"D:\PycharmProjects\TS\DualAttentionSeq2Seq\ts_model\tas2016.csv")

    # f.close()
