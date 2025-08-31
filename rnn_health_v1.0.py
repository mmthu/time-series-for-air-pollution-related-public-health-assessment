#  _*_coding: utf-8 _*_
# Python 3.7.13. Author: Raymond Zhang. Date: Aug 9, 2023.
#  An LSTM model for air-pollutants-related health risk assessment

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, math
import dt_prep

class att(nn.Module):
    def __init__(self,hidden):
        super(att,self).__init__()
        self.input = nn.Sequential(nn.Linear(hidden,1))
    def forward(self, x):
        w = self.input(x)
        ws = F.softmax(w.squeeze(-1),dim=1)
        out_att0 = (x * ws.unsqueeze(-1)).sum(dim=1)
        return out_att0, ws
############################################################ RNN class
class LSTM_model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, out_size,num_layers) -> None:
        super(LSTM_model, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = att(hidden_size)
        self.linear_out = nn.Linear(hidden_size, out_size)
        self.hidden_size = hidden_size
    def forward(self, x):
        x, _ = self.rnn(x)
        out_att, ws = self.attention(x)
        out_in = self.linear_out(out_att)
        return out_in
def train_model(feature_size, train_x, train_y, train_size, epochs_set):
    # para setting for attention:  h_size = 13, o_size = 1, n_layers = 5, l_rate = 5e-2, seed = 65, step = 100
    h_size, o_size, n_layers = 13, 1, 3
    l_rate = 5e-2
    epochs = epochs_set
    step = 100
    seed = 65
    torch.manual_seed(seed)
    ############################################################################
    nn_net = LSTM_model(input_size = feature_size, hidden_size = h_size, out_size = o_size, num_layers = n_layers)
    loss_fun = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(nn_net.parameters(), lr=l_rate)
    step_schedule = torch.optim.lr_scheduler.StepLR(step_size=step, gamma=0.95, optimizer=optimizer)
    running_loss = 0.0
    loss_show = []
    for epoch in range(epochs):
        var_x = train_x
        var_y = train_y.reshape(train_size, -1)
        out = nn_net(var_x)
        loss = loss_fun(out, var_y)
        loss_show.append(loss.item())
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step_schedule.step()
        if (epoch + 1) % 100 == 0:
            print('Epoch: {}, Loss: {:.5f}, lr:{:.5f}'.format(epoch + 1, running_loss / 100, step_schedule.get_last_lr()[0]))
            running_loss = 0.0
    return nn_net, loss_show

def test_model(nn_net, dataY, torch_dataX):
    test_net = nn_net.eval()
    train_predict = test_net(torch_dataX)
    data_predict = train_predict.data.numpy()
    data_true = dataY
    return data_predict, data_true
def plot_prediction1(seqs,train_sizes):
    file_prediction = "./predictions.xlsx"
    if os.path.isfile(file_prediction):
        print("Plotting...")
    df_prediction = pd.read_excel(file_prediction, sheet_name=None)
    font_size = 10
    mk_size = 0.8
    l, r, t, b = 0.07, 0.96, 0.96, 0.07  # bord width
    h, w = 0.13, 0.07  # width among sub-figures
    lw = 1.3  # linewidth
    lpx, lpy = 0.05, 1  # label pad

    len_seqs = len(seqs)
    nr, nc = math.ceil(len_seqs/2), 2  # row and column numbers

    fig, ax = plt.subplots(nrows=nr, ncols=nc, sharex=True, sharey=True)
    for i in range(nr):
        for j in range(nc):
            if i!=nr-1:
                ax[i,j].tick_params(axis="x", labelbottom=False)
            else:
                ax[i, j].set_xticks([0,730,1460,2191,2921,3652,4382,5112],[r'1987', r'1989', r'1991', r'1993', r'1995', r'1997', r'1999', r'2001'], size=font_size)
            if j==1:
                ax[i, j].tick_params(axis="y", labelbottom=False)

            if (len_seqs%2!=0)&(i==nr-1)&(j==1):
                ax[i,j].plot()
            else:
                true_value = df_prediction['seq={}'.format(2 * i + j + 1)]['true']
                predict_value = df_prediction['seq={}'.format(2 * i + j + 1)]['predict']
                x = len(true_value)
                seg = train_sizes[2 * i + j]
                ymin = np.floor(min(min(true_value), min(predict_value)) - 1)
                ymax = np.ceil(max(max(true_value),max(predict_value))+1)
                ax[i, j].set_ylim(ymin, ymax)
                ax[i, j].set_yticks(np.arange(ymin, ymax, 2))
                if (i==0) and (j==0):
                    ax[i, j].plot(np.arange(x), true_value, color="tab:blue", linewidth=lw, label='True Mortality')
                    ax[i, j].plot(np.arange(0, seg), predict_value[:seg], color='tab:orange', linewidth=lw, label='Prediction_train')
                    ax[i, j].plot(np.arange(seg, x), predict_value[seg:], color='tab:red', linewidth=lw, label='Prediction_test')
                    ax[i, j].axvline(x=seg, color='blue', linestyle='-', linewidth=lw + 0.6, alpha=0.7)
                    ax[i, j].text(2500, ymin+0.5, "m={}".format(2 * i + j + 1), fontsize=12, color="red")
                else:
                    ax[i, j].plot(np.arange(x),true_value,color="tab:blue",linewidth=lw)
                    ax[i, j].plot(np.arange(0, seg), predict_value[:seg], color='tab:orange', linewidth=lw)
                    ax[i, j].plot(np.arange(seg, x), predict_value[seg:], color='tab:red', linewidth=lw)

                    ax[i, j].axvline(x=seg, color='blue', linestyle='-',linewidth=lw+0.6,alpha=0.7)
                    ax[i, j].text(2500, ymin+0.5, "m={}".format(2*i+j+1), fontsize=12,color="red")
    handles, labels = ax[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3,frameon=True,loc='upper center', labelspacing=0.2, columnspacing=0.5,
                                handlelength=1.0, handletextpad=0.2, borderaxespad=0.1,fontsize=9)

    fig.text(0.025, 0.5, 'Daily Mortality', va='center', rotation='vertical', fontsize=10)
    fig.text(0.5, 0.025, 'Time', va='center', rotation='horizontal', fontsize=10)
    plt.subplots_adjust(left=l, right=r, top=t, bottom=b, hspace=h, wspace = w)
    plt.savefig('./predict_true.png', dpi=600,  bbox_inches='tight', transparent=True)
def plot_loss_var(seqs,epochs_set):
    file = "./losses.xlsx"
    if os.path.isfile(file):
        print("Plotting loss...")
    df_loss = pd.read_excel(file, sheet_name=None)
    font_size, mk_size = 10, 0.8
    l, r, t, b = 0.07, 0.96, 0.96, 0.07
    h, w = 0.13, 0.07  # width among sub-figures
    lw = 1.3  # linewidth
    lpx, lpy = 0.05, 1  # label pad
    len_seqs = len(seqs)
    nr, nc = math.ceil(len_seqs/2), 2  # row and column numbers

    x = epochs_set
    fig, ax = plt.subplots(nrows=nr, ncols=nc, sharex=True, sharey=True)
    for i in range(nr):
        for j in range(nc):
            if i % 2 == 0:
                ax[i, j].set_yticks(np.arange(0, 1.2, 0.2))
            if i!=nr-1:
                ax[i,j].tick_params(axis="x", labelbottom=False)
            else:
                ax[i, j].set_xticks(np.arange(0,x,round(x/8)))
            if j==1:
                ax[i, j].tick_params(axis="y", labelbottom=False)
            if (len_seqs%2!=0)&(i==nr-1)&(j==1):
                ax[i,j].plot()
            else:
                loss = df_loss['seq={}'.format(2 * i + j + 1)]['loss']
                ax[i, j].plot(np.arange(x),loss,color="tab:orange",linewidth=lw)
                ax[i, j].text(round(0.75*x), 0.8, "m={}".format(2*i+j+1), fontsize=12,color="red")

    fig.text(0.02, 0.5, 'Loss', va='center', rotation='vertical', fontsize=10)
    fig.text(0.5, 0.02, 'epoch', va='center', rotation='horizontal', fontsize=10)
    plt.subplots_adjust(left=l, right=r, top=t, bottom=b, hspace=h, wspace = w)
    plt.savefig('./loss.png', dpi=600,  bbox_inches='tight', transparent=True)
#########################################
if __name__=="__main__":
    features = ['death','tempt', 'PM10', 'O3', 'CO','NO2']
    dt = dt_prep.read_data(features)
    # if os.path.isfile("predictions.xlsx"):
    #     os.remove("predictions.xlsx")
    # if os.path.isfile("losses.xlsx"):
    #     os.remove("losses.xlsx")
    # df1 = pd.DataFrame()
    # df2 = pd.DataFrame()
    # df1.to_excel("predictions.xlsx")
    # df2.to_excel("losses.xlsx")
    seqs = [1,2,3,4,5,6,7,8,9,10,11,12]  # multiple seqs
    epochs_set = 5000
    ###################
    train_sizes = []
    mse, rmse, mae, mape, r2 = [], [], [], [], []
    mseT, rmseT, maeT, mapeT, r2T = [], [], [], [], []
    for seq in seqs:
        print("##################### seq = {} ############################".format(seq))
        dataX, dataY, torch_dataX, torch_dataY, feature_size, train_x, train_y, train_size, test_x, test_y, test_size = dt_prep.split_dataset(dt, seq)
        train_sizes.append(train_size)
        nn_net, loss_show = train_model(feature_size, train_x, train_y, train_size, epochs_set)
        data_predict, data_true = test_model(nn_net, dataY, torch_dataX)

        mse_train, rmse_train, mae_train, mape_train, r2_train, mse_test, rmse_test, mae_test, mape_test, r2_test = dt_prep.performance_metrics2(data_predict, data_true, train_size)
        mse.append(mse_train)
        rmse.append(rmse_train)
        mae.append(mae_train)
        mape.append(mape_train)
        r2.append(r2_train)
        mseT.append(mse_test)
        rmseT.append(rmse_test)
        maeT.append(mae_test)
        mapeT.append(mape_test)
        r2T.append(r2_test)
        df = pd.DataFrame({'true': data_true.flatten(),
                           'predict': data_predict.flatten()}, dtype='float32')
        with pd.ExcelWriter('predictions.xlsx',mode="a") as writer:
            df.to_excel(writer, index=False, sheet_name="seq={}".format(seq))

        df_loss = pd.DataFrame({'loss': loss_show}, dtype='float32')
        with pd.ExcelWriter('losses.xlsx',mode="a") as writer_loss:
            df_loss.to_excel(writer_loss, index=False, sheet_name="seq={}".format(seq))