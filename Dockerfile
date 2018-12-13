FROM ubuntu:latest
MAINTAINER Tom Hosking "code@tomho.sk"

RUN apt-get update -y
RUN apt-get install -y python3 python3-pip python3-dev build-essential

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app
RUN pip3 install -r requirements.txt

ADD ./src /app/src
WORKDIR /app
ENV PYTHONPATH "${PYTHONPATH}:./src"
ENTRYPOINT ["python3"]
CMD ["src/demo/app.py"]
# CMD ["bash", 'demo.sh']