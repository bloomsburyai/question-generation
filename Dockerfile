FROM ubuntu:latest
MAINTAINER Tom Hosking "code@tomho.sk"

RUN apt-get update -y
RUN apt-get install -y python3 python3-pip python3-dev build-essential
ADD ./src /app/src
COPY ./requirements.txt /app

WORKDIR /app
RUN pip3 install -r requirements.txt
WORKDIR /app/src/demo
ENTRYPOINT ["python3"]
CMD ["app.py"]