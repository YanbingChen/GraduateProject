import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# fix random seed for reproducibility
np.random.seed(7)

train_data = np.load("../out/train_data.npy")
train_label = np.load("../out/train_label.npy")
test_data = np.load("../out/test_data.npy")
test_label = np.load("../out/test_label.npy")

# -------- Data preprocess --------
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
look_back = 1

# -------- Training model --------
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(train_data, train_label, epochs=100, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(train_data)
testPredict = model.predict(test_data)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([train_label])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([test_label])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

