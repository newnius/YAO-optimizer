#!/usr/bin/python
from threading import Thread
from threading import Lock
from http.server import BaseHTTPRequestHandler, HTTPServer
import cgi
import json
from urllib import parse

PORT_NUMBER = 8000

lock = Lock()


def train_models():
	lock.acquire()

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
				job = query.get('job')[0]
				gpu_model = query.get('gpu_model')[0]
				time = query.get('time')[0]
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
		if self.path == "/train":
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

		# Wait forever for incoming http requests
		server.serve_forever()

	except KeyboardInterrupt:
		print('^C received, shutting down the web server')

	server.socket.close()
