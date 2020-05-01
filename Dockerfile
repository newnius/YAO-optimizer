FROM tensorflow/tensorflow:1.14.0-gpu

MAINTAINER Newnius <newnius.cn@gmail.com>

RUN apt update && \
	apt install -y python3 python3-pip && \
	rm -rf /var/lib/apt/lists/*

RUN apt update && \
	apt install -y git vim httpie && \
	rm -rf /var/lib/apt/lists/*

RUN pip3 install pandas sklearn tensorflow-gpu==1.14

ADD bootstrap.sh /etc/bootstrap.sh

RUN mkdir /root/data/
ADD serve.py /root/serve.py
ADD model_tensorflow.py /root/model_tensorflow.py

WORKDIR /root

CMD ["/etc/bootstrap.sh"]