import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

frame = "tensorflow"
from model.model_tensorflow import train, predict


class Config:
	feature_columns = list([2,5])
	label_columns = [5]
	feature_and_label_columns = feature_columns + label_columns
	label_in_feature_columns = (lambda x, y: [x.index(i) for i in y])(feature_columns, label_columns)

	predict_day = 1

	input_size = len(feature_columns)
	output_size = len(label_columns)

	hidden_size = 128
	lstm_layers = 2
	dropout_rate = 0.2
	time_step = 20

	do_train = True
	do_predict = True
	add_train = False
	shuffle_train_data = True

	train_data_rate = 0.95
	valid_data_rate = 0.15

	batch_size = 64
	learning_rate = 0.001
	epoch = 20
	patience = 5
	random_seed = 42

	do_continue_train = False
	continue_flag = ""
	if do_continue_train:
		shuffle_train_data = False
		batch_size = 1
		continue_flag = "continue_"

	train_data_path = "./data/stock_data.csv"
	model_save_path = "./checkpoint/"
	figure_save_path = "./figure/"
	do_figure_save = False
	if not os.path.exists(model_save_path):
		os.mkdir(model_save_path)
	if not os.path.exists(figure_save_path):
		os.mkdir(figure_save_path)

	used_frame = frame
	model_postfix = {"pytorch": ".pth", "keras": ".h5", "tensorflow": ".ckpt"}
	model_name = "model_" + continue_flag + used_frame + model_postfix[used_frame]


class Data:
	def __init__(self, config):
		self.config = config
		self.data, self.data_column_name = self.read_data()

		self.data_num = self.data.shape[0]
		self.train_num = int(self.data_num * self.config.train_data_rate)

		self.mean = np.mean(self.data, axis=0)
		self.std = np.std(self.data, axis=0)
		self.norm_data = (self.data - self.mean) / self.std

		self.start_num_in_test = 0

	def read_data(self):
		init_data = pd.read_csv(self.config.train_data_path,
		                        usecols=self.config.feature_and_label_columns)
		return init_data.values, init_data.columns.tolist()

	def get_train_and_valid_data(self):
		feature_data = self.norm_data[:self.train_num]
		label_data = self.norm_data[self.config.predict_day: self.config.predict_day + self.train_num,
		             self.config.label_in_feature_columns]
		if not self.config.do_continue_train:
			train_x = [feature_data[i:i + self.config.time_step] for i in range(self.train_num - self.config.time_step)]
			train_y = [label_data[i:i + self.config.time_step] for i in range(self.train_num - self.config.time_step)]
		else:
			train_x = [
				feature_data[start_index + i * self.config.time_step: start_index + (i + 1) * self.config.time_step]
				for start_index in range(self.config.time_step)
				for i in range((self.train_num - start_index) // self.config.time_step)]
			train_y = [
				label_data[start_index + i * self.config.time_step: start_index + (i + 1) * self.config.time_step]
				for start_index in range(self.config.time_step)
				for i in range((self.train_num - start_index) // self.config.time_step)]

		train_x, train_y = np.array(train_x), np.array(train_y)

		train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=self.config.valid_data_rate,
		                                                      random_state=self.config.random_seed,
		                                                      shuffle=self.config.shuffle_train_data)
		return train_x, valid_x, train_y, valid_y

	def get_test_data(self, return_label_data=False):
		feature_data = self.norm_data[self.train_num:]
		self.start_num_in_test = feature_data.shape[0] % self.config.time_step
		time_step_size = feature_data.shape[0] // self.config.time_step

		test_x = [feature_data[self.start_num_in_test + i * self.config.time_step: self.start_num_in_test + (
				i + 1) * self.config.time_step]
		          for i in range(time_step_size)]
		if return_label_data:
			label_data = self.norm_data[self.train_num + self.start_num_in_test:, self.config.label_in_feature_columns]
			return np.array(test_x), label_data
		return np.array(test_x)


def draw(config, origin_data, predict_norm_data):
	label_norm_data = origin_data.norm_data[origin_data.train_num + origin_data.start_num_in_test:,
	                  config.label_in_feature_columns]
	assert label_norm_data.shape[0] == predict_norm_data.shape[
		0], "The element number in origin and predicted data is different"

	label_name = [origin_data.data_column_name[i] for i in config.label_in_feature_columns]
	label_column_num = len(config.label_columns)

	loss = np.mean((label_norm_data[config.predict_day:] - predict_norm_data[:-config.predict_day]) ** 2, axis=0)
	print("The mean squared error of stock {} is ".format(label_name), loss)

	label_X = range(origin_data.data_num - origin_data.train_num - origin_data.start_num_in_test)
	predict_X = [x + config.predict_day for x in label_X]

	label_data = label_norm_data * origin_data.std[config.label_in_feature_columns] + \
	             origin_data.mean[config.label_in_feature_columns]

	predict_data = predict_norm_data * origin_data.std[config.label_in_feature_columns] + \
	               origin_data.mean[config.label_in_feature_columns]

	print(label_data)
	print(predict_data)

	'''
	for i in range(label_column_num):
		plt.figure(i + 1)
		plt.plot(label_X, label_data[:, i], label='label')
		plt.plot(predict_X, predict_data[:, i], label='predict')
		plt.legend(loc='upper right')
		plt.xlabel("Day")
		plt.ylabel("Price")
		plt.title("Predict stock {} price with {}".format(label_name[i], config.used_frame))
		print("The predicted stock {} for the next {} day(s) is: ".format(label_name[i], config.predict_day),
		      np.squeeze(predict_data[-config.predict_day:, i]))
		if config.do_figure_save:
			plt.savefig(config.figure_save_path + "{}predict_{}_with_{}.png".format(config.continue_flag, label_name[i],
			                                                                        config.used_frame))

	plt.show()
	'''


def main(config):
	np.random.seed(config.random_seed)
	data_gainer = Data(config)

	if config.do_train:
		train_X, valid_X, train_Y, valid_Y = data_gainer.get_train_and_valid_data()
		train(config, train_X, train_Y, valid_X, valid_Y)

	if config.do_predict:
		test_X, test_Y = data_gainer.get_test_data(return_label_data=True)
		pred_result = predict(config, test_X)
		draw(config, data_gainer, pred_result)


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser()
	# parser.add_argument("-t", "--do_train", default=False, type=bool, help="whether to train")
	# parser.add_argument("-p", "--do_predict", default=True, type=bool, help="whether to train")
	# parser.add_argument("-b", "--batch_size", default=64, type=int, help="batch size")
	# parser.add_argument("-e", "--epoch", default=20, type=int, help="epochs num")
	args = parser.parse_args()

	con = Config()
	for key in dir(args):
		if not key.startswith("_"):
			setattr(con, key, getattr(args, key))

	main(con)
