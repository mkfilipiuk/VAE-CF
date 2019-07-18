FROM nvcr.io/nvidia/tensorflow:19.05-py3
WORKDIR /code
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8888
