import numpy as np
import pandas as pd

import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from random import randint

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.callbacks import EarlyStopping
from keras import initializers

from matplotlib import pyplot as plt

import plotly.offline as py
import plotly.graph_objs as go

#location of the CSV file
file = 'XBTUSD_5_min.csv'

#function for rearranging the data according to model's needs
def rearranging_data(file):
    data = pd.read_csv(file)    
    for i in range(data.shape[0]):
        data.iat[i,0] = data.iat[i,0][0:19]

    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.set_index("timestamp")

    data = data.loc["2018-08-23":]
    return data


#function for extracting the workable set of data from the whole dataset
def working_data(data):
    hourly_data = data.vwap.resample('D').mean()

    train_data = hourly_data["2018-08-23":"2019-06-01"]
    test_data = hourly_data["2019-06-01":]

    workable_data = [train_data, test_data]
    workable_data = pd.concat(workable_data)

    workable_data = workable_data.reset_index()
    workable_data["timestamp"] = pd.to_datetime(workable_data["timestamp"])
    workable_data = workable_data.set_index("timestamp")

    workable_data = workable_data.fillna(method="pad")

    return workable_data

#the workable set of data - a concatenated dataset of train and test datasets
workable_data = working_data(rearranging_data(file))

print("Shape of Workable data = ", workable_data.shape)

#extraction of train and test data from workabole data
train_data = workable_data[:283]
test_data = workable_data[283:]

print("\n\nShape of Train data =", train_data.shape)
print("\n\nShape of Test data =", test_data.shape)

print("\n\n\n")


#function for creating lookbacks for sequential forecasting
def lookback_creator(dataset, look_back): #increase the value of look_back for increasing the number of past values you wanna take to predict the future values
    inputs_X, labels_Y = [], []
    for i in range(len(dataset) - look_back):
        lb_data = dataset[i:(i + look_back), 0]
        inputs_X.append(lb_data)
        labels_Y.append(dataset[i + look_back, 0])
    
    return np.array(inputs_X), np.array(labels_Y)


train_dataset = train_data.values
train_dataset = np.reshape(train_dataset, (len(train_dataset), 1))

test_dataset = test_data.values
test_dataset = np.reshape(test_dataset, (len(test_dataset), 1))
    
#Scaling the datasets
scaler = MinMaxScaler()
train_dataset = scaler.fit_transform(train_dataset)
test_dataset = scaler.transform(test_dataset)

#Making the datasets suitable for time series forecasting
look_back = 10
train_X, train_Y = lookback_creator(train_dataset, look_back)
test_X, test_Y = lookback_creator(test_dataset, look_back)

#reshaping the datasets to meet the requirements of Keras LSTM model
train_X = np.reshape(train_X, (len(train_X), 1, train_X.shape[1]))
test_X = np.reshape(test_X, (len(test_X), 1, test_X.shape[1]))


#Initiliazing the sequential model, add 2 stacked LSTM layers and densely connected output neurons


predict_model = Sequential()
predict_model.add(LSTM(256, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
predict_model.add(LSTM(256))
predict_model.add(Dense(1))

predict_model.compile(loss='mean_squared_error', optimizer='adam')

history = predict_model.fit(train_X, train_Y, epochs=100, batch_size=20, shuffle=False, validation_data=(test_X, test_Y), callbacks=[EarlyStopping(monitor='val_loss', min_delta=5e-5, patience=20, verbose=1)])


first_trace = go.Scatter(
    x = np.arange(0, len(history.history['loss']), 1),
    y = history.history['loss'],
    mode = 'lines',
    name = 'Loss in Train data',
    line = dict(color=('rgb(66, 244, 155)'), width=2, dash='dash')
)


second_trace = go.Scatter(
    x = np.arange(0, len(history.history['val_loss']), 1),
    y = history.history['val_loss'],
    mode = 'lines',
    name = 'Loss in Test data',
    line = dict(color=('rgb(244, 146, 65)'), width=2)
)

plot_data = [first_trace, second_trace]
plot_layout = dict(title = 'Train and Test Loss during Training', xaxis = dict(title ="Epoch Number"), yaxis = dict(title="Loss"))

plot_figure = dict(data=plot_data, layout=plot_layout)
py.plot(plot_figure, filename="training_process")


#This is for plotting the graph for Test Prices compared to predicted prices
test_X = np.append(test_X, scaler.transform(workable_data.iloc[[-1]]))
test_X = np.reshape(test_X, (len(test_X), 1, 1))

prediction = predict_model.predict(test_X)
prediction_inverse = scaler.inverse_transform(prediction.reshape(-1,1))
test_Y_inverse = scaler.inverse_transform(test_Y.reshape(-1,1))
prediction2_inverse = np.array(prediction_inverse[:,0][1:])
test_Y2_inverse = np.array(test_Y_inverse[:,0])


trace_1 = go.Scatter(
    x = np.arange(0, len(prediction2_inverse), 1),
    y = prediction2_inverse,
    mode = 'lines',
    name = 'Predicted labels',
    line = dict(color=('rgb(244, 146, 65)'), width=2)
)

trace_2 = go.Scatter(
    x = np.arange(0, len(test_Y2_inverse), 1),
    y = test_Y2_inverse,
    mode = 'lines',
    name = 'Original Labels',
    line = dict(color=('rgb(66, 244, 155)'), width=2)
)


p_data = [trace_1, trace_2]
p_layout = dict(title ="Comparison of original Prices with Prices Predicted by our model", xaxis = dict(title="Hours"), yaxis = dict(title="Price in USD"))
p_fig = dict(data=p_data, layout=p_layout)
py.plot(p_fig, filename="results_demonstration_0")

#calculating the RMSE
RMSE = sqrt(mean_squared_error(test_Y2_inverse, prediction2_inverse))
print('Test RMSE: %.3f' % RMSE)





