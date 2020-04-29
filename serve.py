import tensorflow as tf
import numpy as np
import tushare as ts
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

timesteps = seq_length = 7
data_dim = 5
output_dim = 1

stock_data = ts.get_k_data('600000', start='2015-01-01', end='2017-12-01')
xy = stock_data[['open', 'close', 'high', 'low', 'volume']]

# xy_new = pd.DataFrame()
# scaler = MinMaxScaler()

# scaler.fit(xy)
# t = scaler.transform(xy)

# for col in xy.columns:
#	xy_new.ix[:, col] = t[col]

x = xy
y = xy[['close']]
dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
	_x = x[i:i + seq_length]
	_y = y.loc[i + seq_length]
	#print(_x, "->", _y)
	dataX.append(_x)
	dataY.append(_y)

x_real = np.vstack(dataX).reshape(-1, seq_length, data_dim)
y_real = np.vstack(dataY).reshape(-1, output_dim)
print(x_real.shape)
print(y_real.shape)
dataX = x_real
dataY = y_real

train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size
trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:len(dataY)])

X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 1])


def add_layer(inputs, in_size, out_size, activation_function=None):
	inputs = tf.reshape(inputs, [-1, in_size])
	Weights = tf.Variable(tf.random_normal([in_size, out_size]))
	biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
	Wx_plus_b = tf.matmul(inputs, Weights) + biases
	if activation_function is None:
		outputs = Wx_plus_b
	else:
		outputs = activation_function(Wx_plus_b)
	return outputs


outsize_first = 5
l1 = add_layer(X, data_dim, outsize_first, activation_function=tf.nn.relu)
l1_output = tf.reshape(l1, [-1, seq_length, outsize_first])

cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=output_dim, state_is_tuple=True)
outputs, _states = tf.nn.dynamic_rnn(cell, l1_output, dtype=tf.float32)
Y_pred = outputs[:, -1]

loss = tf.reduce_sum(tf.square(Y_pred - Y))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(100):
	_, l = sess.run(
		[train, loss],
		feed_dict={X: trainX, Y: trainY}
	)
	#print(i, l)

testPredict = sess.run(Y_pred, feed_dict={X: testX})

print(testY)
print(testPredict)
