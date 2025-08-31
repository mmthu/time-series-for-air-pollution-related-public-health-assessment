#  _*_coding: utf-8 _*_
# Python 3.7.13. Author: Raymond Zhang. Date: Oct 10, 2024.
#  An TCN model for air-pollutants-related health risk assessment

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import os, math,sys,copy


def data_prep(features,seq):
    filepath = './data/chic_1987_2000.xlsx' # chic_1987_2000
    df = pd.read_excel(filepath, engine='openpyxl')
    df = df[features]
    dt = df.values
    dt = dt.astype("float32")
    r,c = dt.shape # 5114,4
    dt_scaler= dt[:,0].reshape((r,1))
    scaler = []
    dt_out = np.empty((r,0))
    for i in np.arange(c-1):
        dt_scaler =np.concatenate((dt_scaler,dt[:,i+1].reshape((r,1))),axis=1)
    for j in np.arange(c):
        scaler.append(StandardScaler())
        tempt = scaler[j].fit_transform(dt_scaler[:, j].reshape((r, 1)))
        dt_out = np.concatenate((dt_out,tempt),axis=1)
    dt_T = dt_out.T

    dataX, dataY = [], []
    for i in range(r - seq + 1):
        y = dt_T[0,i + seq - 1]
        x = dt_T[1:r,i:(i + seq)]
        dataY.append([y])
        dataX.append(x)
    dataX, dataY = np.array(dataX), np.array(dataY)
    r1,c1 = dataY.shape
    train_size = int(r1 * 0.7)
    torch_dataX= torch.from_numpy(dataX).type(torch.float32)
    torch_dataY= torch.from_numpy(dataY).type(torch.float32)
    return torch_dataX, torch_dataY,dataX,dataY,scaler

class Crop(nn.Module):
    def __init__(self, crop_size):
        super(Crop, self).__init__()
        self.crop_size = crop_size
    def forward(self, x):
        out = x[:, :, :-self.crop_size].contiguous()
        return out


class TemporalCasualLayer(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout):
        super(TemporalCasualLayer, self).__init__()
        padding = (kernel_size - 1) * dilation
        conv_params = {'kernel_size': kernel_size,
                    'stride': stride,
                    'padding': padding,
                    'dilation': dilation}

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, **conv_params))
        # print("Causal layer input and output:",n_inputs, n_outputs)
        self.crop1 = Crop(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, **conv_params))
        self.crop2 = Crop(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.crop1, self.relu1, self.dropout1,
                                 self.conv2, self.crop2, self.relu2, self.dropout2)
        # shortcut connect
        self.bias = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.net(x)
        b = x if self.bias is None else self.bias(x)
        return self.relu(y + b)


class TemporalConvolutionNetwork(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout):
        super(TemporalConvolutionNetwork, self).__init__()
        layers = []
        num_levels = len(num_channels) # 4
        tcl_param = {
            'kernel_size': kernel_size,
            'stride': 1,
            'dropout': dropout
        }
        print(num_levels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            # print(in_ch,out_ch)
            tcl_param['dilation'] = dilation
            tcl = TemporalCasualLayer(in_ch, out_ch, **tcl_param)
            layers.append(tcl)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        # input_size=look_back, num_channels=[10]*4
        super(TCN, self).__init__()
        self.tcn = TemporalConvolutionNetwork(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(10*25, output_size)
    def forward(self, x):
        y = self.tcn(x)
        out = self.linear(torch.flatten(y,1))
        return out

if __name__=="__main__":
    import matplotlib
    matplotlib.use('QtAgg')
    import matplotlib.pyplot as plt
    from torchsummary import summary
    features = ['death','tempt', 'PM10', 'O3']
    seq = 25
    torch_dataX,torch_dataY,dataX,dataY,scaler = data_prep(features,seq)
    # print(torch_dataX.shape,torch_dataY.shape)

    train_size = int(torch_dataY.shape[0] * 0.8)

    x_train, y_train = torch_dataX[:train_size,:,:],torch_dataY[:train_size,:]
    x_test, y_test= torch_dataX[train_size:,:,:],torch_dataY[train_size:,:]
    #################################################
    # training epochs
    epochs = 4000
    channel_sizes = [10] * 4
    kernel_size = 3
    dropout = .0
    model_params = {
        'input_size': 3,  # features
        'output_size': 1,
        'num_channels': channel_sizes,
        'kernel_size': kernel_size,
        'dropout': dropout
    }
    model = TCN(**model_params)

    # print(summary(model, input_size=(3, seq), batch_size=5))
    # torch.onnx.export(model, torch.rand(5,3, seq), "model.onnx")
    # import netron
    # netron.start('model.onnx')

    # import onnx
    # onnx_ori = "model.onnx"
    # onnx_show = "model_show.onnx"
    # onnx_graph = onnx.load(onnx_ori)
    # estimated_graph = onnx.shape_inference.infer_shapes(onnx_graph)
    # onnx.save(estimated_graph, onnx_show)
    # import netron
    # netron.start('model_show.onnx')

    optimizer = torch.optim.Adam(params=model.parameters(), lr=.005)
    mse_loss = torch.nn.MSELoss()
    best_params = None
    min_val_loss = sys.maxsize
    training_loss = []
    validation_loss = []
    x_val = x_test
    y_val = y_test
    for t in range(epochs):
        prediction = model(x_train)
        loss = mse_loss(prediction, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        val_prediction = model(x_val)
        val_loss = mse_loss(val_prediction, y_val)
        training_loss.append(loss.item())
        validation_loss.append(val_loss.item())
        if val_loss.item() < min_val_loss:
            best_params = copy.deepcopy(model.state_dict())
            min_val_loss = val_loss.item()
        if t % 100 == 0:
            diff = (y_train - prediction).view(-1).abs_().tolist()
            print(f'epoch {t}. train: {round(loss.item(), 4)}, '
                  f'validation: {round(val_loss.item(), 4)}')
    fig,ax = plt.subplots()
    ax.set_title('Training Progress')
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    ax.plot(training_loss, label='train')
    ax.plot(validation_loss, label='validation')
    ax.legend()
    plt.savefig('c:/Users/m/Desktop/loss_epoch.png', dpi=300, bbox_inches='tight', transparent=False)

    best_model = TCN(**model_params)
    best_model.eval()
    best_model.load_state_dict(best_params)

    # tcn_prediction = best_model(x_test)
    # data_prediction = tcn_prediction.view(-1).detach().numpy().reshape(-1, 1)
    data_true = scaler[0].inverse_transform(dataY)
    tcn_prediction = best_model(torch_dataX)
    data_prediction_s = tcn_prediction.view(-1).detach().numpy().reshape(-1, 1)
    data_prediction = scaler[0].inverse_transform(data_prediction_s)

    fig1,ax1 = plt.subplots(2,1,sharex=True)
    ax1[0].set_ylabel('Scaled Mortality')
    ax1[0].plot(np.arange(data_prediction_s.size), data_prediction_s, color='red', label='tcn_scale')
    ax1[0].plot(dataY, alpha=0.5, label='real_scale')
    ax1[0].axvline(x=train_size, color='red', linestyle='--', linewidth=2, label='train:test')
    ax1[0].legend()
    # ax1.plot(np.arange(torch_dataY.shape[0]  - len(tcn_prediction), torch_dataY.shape[0] ), data_prediction, color='red', label='tcn')

    ax1[1].set_ylabel('Mortality')
    ax1[1].set_xlabel('Time samples from 1987 to 2000')
    ax1[1].plot(np.arange(data_prediction.size), data_prediction, color='red', label='tcn')
    ax1[1].plot(data_true, alpha=0.5, label='real')
    ax1[1].axvline(x=train_size, color='red', linestyle='--', linewidth=2, label='train:test')
    ax1[1].legend()
    plt.savefig('c:/Users/m/Desktop/prediction_real.png', dpi=300, bbox_inches='tight', transparent=False)
    plt.show()




