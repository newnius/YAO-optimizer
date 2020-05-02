from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
import numpy


# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag + 1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df = df.drop(0)
	return df


# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)


# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]


# scale train and test data to [-1, 1]
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled


# inverse scaling for a forecasted value
def invert_scale(scaler, X, yhat):
	new_row = [x for x in X] + [yhat]
	array = numpy.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]


# fit an LSTM network to training data
def fit_lstm(train, batch_size2, nb_epoch, neurons):
	print(train)
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	print(X, y)
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size2, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		print("Epoch {}/{}".format(i, nb_epoch))
		model.fit(X, y, epochs=1, batch_size=batch_size2, verbose=0, shuffle=False)
		loss, accuracy = model.evaluate(X, y)
		print(loss, accuracy)
		model.reset_states()
	return model


# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0, 0]


# run a repeated experiment
def experiment(repeats, series, seed):
	# transform data to be stationary
	raw_values = series.values
	diff_values = difference(raw_values, 1)
	# transform data to be supervised learning
	lag2 = 4
	supervised = timeseries_to_supervised(diff_values, lag2)
	supervised_values = supervised.values

	test_data_num = 32
	# split data into train and test-sets
	train, test = supervised_values[0:-test_data_num], supervised_values[-test_data_num:]
	# transform the scale of the data
	print(test)
	scaler, train_scaled, test_scaled = scale(train, test)
	print(test_scaled)
	# run experiment
	error_scores = list()
	for r in range(repeats):
		# fit the model
		batch_size = 32
		t1 = train.shape[0] % batch_size
		t2 = test.shape[0] % batch_size

		train_trimmed = train_scaled[t1:, :]
		lstm_model = fit_lstm(train_trimmed, batch_size, 30, 4)
		# forecast the entire training dataset to build up state for forecasting
		print(train_trimmed)
		print(train_trimmed[:, 0])
		print(train_trimmed[:, :-1])
		# if seed:
		#	train_reshaped = train_trimmed[:, :-1].reshape(len(train_trimmed), 1, lag2)
		#	lstm_model.predict(train_reshaped, batch_size=batch_size)
		# forecast test dataset
		test_reshaped = test_scaled[:, 0:-1]
		test_reshaped = test_reshaped.reshape(len(test_reshaped), 1, lag2)
		output = lstm_model.predict(test_reshaped, batch_size=batch_size)
		predictions = list()
		for i in range(len(output)):
			yhat = output[i, 0]
			X = test_scaled[i, 0:-1]
			# invert scaling
			yhat = invert_scale(scaler, X, yhat)
			# invert differencing
			yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
			# store forecast
			predictions.append(yhat)
		# report performance
		rmse = sqrt(mean_squared_error(raw_values[-test_data_num:], predictions))
		print(predictions, raw_values[-test_data_num:])
		print('%d) Test RMSE: %.3f' % (r + 1, rmse))
		error_scores.append(rmse)
	return error_scores


# load dataset
series = read_csv('data.csv', header=0, index_col=0, squeeze=True)
# experiment
repeats = 1
results = DataFrame()
# with seeding
with_seed = experiment(repeats, series, True)
results['with-seed'] = with_seed
# without seeding
without_seed = experiment(repeats, series, False)
results['without-seed'] = without_seed
# summarize results
print(results.describe())
# save boxplot
results.boxplot()
