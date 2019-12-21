import torch.nn as nn
from utils import initialize_weights


class lstm_model(nn.Module):
    def __init__(self, enc_inpt_size, rnn_inpt_size, rnn_hid_size, decd_out_size, net="LSTM", dropout=0.5):
        super(lstm_model, self).__init__()
        self.enc_inpt_size = enc_inpt_size
        self.rnn_inpt_size = rnn_inpt_size
        self.rnn_hid_size = rnn_hid_size
        self.decd_out_size = decd_out_size
        self.net = net

        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Linear(self.enc_inpt_size, self.rnn_inpt_size)
        if net == "LSTM":
            self.hidden_layer = nn.LSTM(input_size=rnn_inpt_size, hidden_size=rnn_hid_size, batch_first=True)
        elif net == "GRU":
            self.hidden_layer = nn.GRU(input_size=rnn_inpt_size, hidden_size=rnn_hid_size, batch_first=True)
        elif net == "RNN":
            self.hidden_layer = nn.RNN(input_size=rnn_inpt_size, hidden_size=rnn_hid_size, batch_first=True)
        self.decoder = nn.Linear(self.rnn_hid_size, self.decd_out_size)
        self.apply(initialize_weights)

    def forward(self, x):
        emb = self.drop(self.encoder(x.reshape(-1, self.enc_inpt_size)))  #batch,seql_len
        emb = emb.reshape(x.shape[0], x.shape[1], self.rnn_inpt_size)
        if self.net == "LSTM":
            _, (h_n, _) = self.hidden_layer(emb)
        else:
            _, h_n = self.hidden_layer(emb)

        out = h_n.reshape(-1, self.rnn_hid_size)
        out = self.decoder(out)
        return out
