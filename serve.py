#!/usr/bin/python
from threading import Thread
from threading import Lock
from http.server import BaseHTTPRequestHandler, HTTPServer
import cgi
import json
from urllib import parse
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from model_tensorflow import train, predict
import csv


class Config:
	feature_columns = list(range(0, 2))
	label_columns = [1]
	feature_and_label_columns = feature_columns + label_columns
	label_in_feature_columns = (lambda x, y: [x.index(i) for i in y])(feature_columns, label_columns)

	predict_day = 1

	input_size = len(feature_columns) - 1
	output_size = len(label_columns)

	hidden_size = 128
	lstm_layers = 2
	dropout_rate = 0.2
	time_step = 5

	do_train = True
	do_predict = True
	add_train = False
	shuffle_train_data = True

	# train_data_rate = 0.95 #comment yqy
	train_data_rate = 0.95  # add yqy
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

	train_data_path = "./data/data.csv"
	model_save_path = "./checkpoint/"
	figure_save_path = "./figure/"

	do_figure_save = False
	if not os.path.exists(model_save_path):
		os.mkdir(model_save_path)
	if not os.path.exists(figure_save_path):
		os.mkdir(figure_save_path)

	used_frame = "tensorflow"
	model_postfix = {"pytorch": ".pth", "keras": ".h5", "tensorflow": ".ckpt"}
	model_name = "model_" + continue_flag + used_frame + model_postfix[used_frame]


class Data:
	def __init__(self, config):
		self.config = config
		self.data, self.data_column_name = self.read_data()

		self.data_num = self.data.shape[0]
		self.train_num = int(self.data_num * self.config.train_data_rate)

		self.mean = np.mean(self.data, axis=0)
		self.std = np.std(self.data, axis=0) + 0.0001
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
				i + 1) * self.config.time_step] for i in range(time_step_size)]

		if return_label_data:
			label_data = self.norm_data[self.train_num + self.start_num_in_test:, self.config.label_in_feature_columns]
			return np.array(test_x), label_data
		return np.array(test_x)

	# add yqy
	def get_test_data_yqy(self, test_data_yqy=None):
		if test_data_yqy is None:
			test_data_yqy = []
		# test_data_yqy=test_data_yqy[1:21]
		feature_data = (test_data_yqy - self.mean) / self.std
		test_x = [feature_data]
		print(feature_data)
		return np.array(test_x)


# add end


def draw_yqy(config2, origin_data, predict_norm_data, mean_yqy, std_yqy):
	label_norm_data = (origin_data - mean_yqy) / std_yqy
	assert label_norm_data.shape[0] == predict_norm_data.shape[
		0], "The element number in origin and predicted data is different"

	# label_norm_data=label_norm_data[:,1]
	label_name = 'high'
	label_column_num = 3

	loss = \
		np.mean((label_norm_data[config.predict_day:, 1:2] - predict_norm_data[:-config.predict_day]) ** 2, axis=0)
	print("The mean squared error of stock {} is ".format(label_name), loss)

	# label_X = range(origin_data.data_num - origin_data.train_num - origin_data.start_num_in_test)
	# predict_X = [x + config.predict_day for x in label_X]

	# print(label_norm_data[:, 1:2])
	label_data = label_norm_data[:, 1:2] * std_yqy[1:2] + mean_yqy[1:2]
	# print(label_data)

	# print(predict_norm_data)
	predict_data = predict_norm_data * std_yqy[config.label_in_feature_columns] + mean_yqy[
		config.label_in_feature_columns]
	# print(predict_data)

	print(label_data[:, -1])
	print(predict_data[:, -1])


PORT_NUMBER = 8080
lock = Lock()
config = Config()


def train_models():
	lock.acquire()
	np.random.seed(config.random_seed)
	data_gainer = Data(config)

	train_X, valid_X, train_Y, valid_Y = data_gainer.get_train_and_valid_data()

	if train_X.shape[0] < 500:
		config.batch_size = 32
	if train_X.shape[0] < 200:
		config.batch_size = 16

	print(train_X[:, :, :1], valid_X[:, :, :1], train_Y, valid_Y)
	train(config, train_X[:, :, :1], train_Y, valid_X[:, :, :1], valid_Y)

	lock.release()


class MyHandler(BaseHTTPRequestHandler):
	# Handler for the GET requests
	def do_GET(self):
		req = parse.urlparse(self.path)
		query = parse.parse_qs(req.query)

		if req.path == "/ping":
			self.send_response(200)
			self.send_header('Content-type', 'application/json')
			self.end_headers()
			self.wfile.write(bytes("pong", "utf-8"))

		elif req.path == "/predict":
			try:
				data = {
					'job': query.get('job')[0],
					'model': query.get('model')[0],
					'time': query.get('time')[0],
					'utilGPU': query.get('utilGPU')[0],
					'utilCPU': query.get('utilCPU')[0],
					'pre': 0,
					'main': 0,
					'post': 0
				}

				data = {
					'seq': query.get('job')[0],
					'value': query.get('model')[0],
				}

				with open(config.train_data_path, 'r') as f:
					df = pd.read_csv(config.train_data_path, usecols=['seq', 'value'])
					df = df.tail(config.time_step - 1)
					df = df.append(data, ignore_index=True)
					df.to_csv('./data/test_data.csv', index=False)

				np.random.seed(config.random_seed)
				data_gainer = Data(config)
				test_data_yqy = pd.read_csv("./data/test_data.csv", usecols=list(range(0, 2)))
				test_data_values = test_data_yqy.values[:]
				test_X = data_gainer.get_test_data_yqy(test_data_values)
				pred_result = predict(config, test_X)

				mean = Data(config).mean
				std = Data(config).std
				draw_yqy(config, test_data_values, pred_result, mean, std)

				msg = {'code': 1, 'error': "container not exist"}
			except Exception as e:
				msg = {'code': 2, 'error': str(e)}
			self.send_response(200)
			self.send_header('Content-type', 'application/json')
			self.end_headers()
			self.wfile.write(bytes(json.dumps(msg), "utf-8"))

		elif req.path == "/feed":
			try:
				job = query.get('job')[0]
				model = query.get('model')[0]
				time = query.get('time')[0]
				utilGPU = query.get('utilGPU')[0]
				utilCPU = query.get('utilCPU')[0]
				pre = query.get('pre')[0]
				main = query.get('main')[0]
				post = query.get('post')[0]

				seq = query.get('seq')[0]
				value = query.get('value')[0]

				with open(config.train_data_path, 'a+', newline='') as csvfile:
					spamwriter = csv.writer(
						csvfile, delimiter=',',
						quotechar='|', quoting=csv.QUOTE_MINIMAL
					)
					# spamwriter.writerow([job, model, time, utilGPU, utilCPU, pre, main, post])
					spamwriter.writerow([seq, value])
				msg = {'code': 1, 'error': "container not exist"}
			except Exception as e:
				msg = {'code': 2, 'error': str(e)}
			self.send_response(200)
			self.send_header('Content-type', 'application/json')
			self.end_headers()
			self.wfile.write(bytes(json.dumps(msg), "utf-8"))

		elif req.path == "/train":
			try:
				t = Thread(target=train_models, name='train_models', args=())
				t.start()
				msg = {'code': 1, 'error': "container not exist"}
			except Exception as e:
				msg = {'code': 2, 'error': str(e)}
			self.send_response(200)
			self.send_header('Content-type', 'application/json')
			self.end_headers()
			self.wfile.write(bytes(json.dumps(msg), "utf-8"))

		else:
			self.send_error(404, 'File Not Found: %s' % self.path)

	# Handler for the POST requests
	def do_POST(self):
		if self.path == "/train2":
			form = cgi.FieldStorage(
				fp=self.rfile,
				headers=self.headers,
				environ={
					'REQUEST_METHOD': 'POST',
					'CONTENT_TYPE': self.headers['Content-Type'],
				})
			try:
				job = form.getvalue('job')[0]
				data = form.getvalue('records')[0]
				records = json.load(data)
				t = Thread(target=train_models(), name='train_models', args=(job, records,))
				t.start()
				msg = {"code": 0, "error": ""}
			except Exception as e:
				msg = {"code": 1, "error": str(e)}
			self.send_response(200)
			self.send_header('Content-type', 'application/json')
			self.end_headers()
			self.wfile.write(bytes(json.dumps(msg), "utf-8"))

		else:
			self.send_error(404, 'File Not Found: %s' % self.path)


if __name__ == '__main__':
	try:
		# Create a web server and define the handler to manage the
		# incoming request
		server = HTTPServer(('', PORT_NUMBER), MyHandler)
		print('Started http server on port ', PORT_NUMBER)

		with open(config.train_data_path, 'w', newline='') as csvfile:
			spamwriter = csv.writer(
				csvfile, delimiter=',',
				quotechar='|', quoting=csv.QUOTE_MINIMAL
			)
			# spamwriter.writerow(["job", "model", "time", "utilGPU", "utilCPU", "pre", "main", "post"])
			spamwriter.writerow(["seq", "value"])

		# Wait forever for incoming http requests
		server.serve_forever()

	except KeyboardInterrupt:
		print('^C received, shutting down the web server')

	server.socket.close()
