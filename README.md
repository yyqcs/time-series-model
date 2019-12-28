### This repository encapsulates the process of modeling and forecasting of time series based deep learning,making it super easy  to use.

### Quick start

```python
train_file_path = r"dataset/pollution.csv"
#data is univariate tite series
data = pd.read_csv(train_file_path, header=0, index_col=0)["pollution"].values
#get train_data and test_data
train_data, test_data = divide_train_test(data)
#initialise model parameters
ts = ts_model()
# fit and predict
preds, reals = ts.fit_transform(train_data, test_data)
# plot prediction result
ts.plot_predict_result(preds, reals)
```

#### 1. model parameters

Some model parameters are diagrammed below：

![model-part-parameters](https://github.com/yyqcs/time-series-model/blob/master/fig/model-part-parameters.jpg)

| The meaning of model parameters                              |
| ------------------------------------------------------------ |
| wnd_len : int ,default=24                                    |
| the length of sliding window.The sequence in sliding window is the  single input sequence of LSTM. |
| pred_len : int ,default=24                                   |
| prediction sequence length                                   |
| net : str ,default=LSTM                                      |
| Net to model time series.Choices include 'LSTM' or 'RNN' or  "GRU" |
| rnn_input_size : int ,default=8                              |
| embedding dimension,used to map input feature(for univariate,feature  is one) to a high dimension |
| rnn_hid_size : int ,default=64.                              |
| the dimension of hidden layer in RNN,LSTM,GRU                |
| batch_size : int, default=32                                 |
| Batch size to use during SGD optimization.                   |
| lr : float, default=1E-3                                     |
| Learning rate used for optimization.                         |
| n_epochs : int, default=299                                  |
| Number of epochs to use during optimization.                 |
| optimizer : str, default='Adam'                              |
| Optimizer to use during SGD optimization. Choices include 'Adam' or  'SGD'. |
| criterion : str, default='MSELoss'                           |
| Prediction loss function to use.                             |
| train_proportion : float,default=0.8                         |
| the proportion of train data used to train model.The rest part is used  for validation |
| not_use_visdom : bool,default=True                           |
| not use visdom to visualize the loss in training process,when False  ensure run"python -m visdom.server" firstly |
| cuda : bool, default=False                                   |
| Whether or not to use CUDA.                                  |
| dropout_rate : float, default=0.5.                           |
| Dropout rate for hidden layers.                              |
| verbose : bool, default=True                                 |
| Print out loss information.                                  |

#### 2. model method

| fit_transform(train_data,test_data) | train model on train_data and predict on test_data |
| ----------------------------------- | -------------------------------------------------- |
| fit(train_data)                     | train model on train_data                          |
| transform(test_data)                | predict on test_data                               |
| plot_predict_result(preds, real)    | plot prediction result                             |
| save_best_model(save_path):         | save file on specified path                        |
| get_min_loss()                      | return minimum validation loss                     |

If you have any questions, please open an issue.

