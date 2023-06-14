FROM tensorflow/tensorflow:latest-gpu

RUN mkdir /airbus

COPY train.py submission.py inference.py README.md requirements.txt Dockerfile /airbus/
COPY data/hue.csv data/labels.csv data/masks.csv data/train_ship_segmentations_v2.csv /airbus/data/
COPY models/ /airbus/models/
COPY settings/ /airbus/settings/
COPY src/ /airbus/src/
COPY notebooks/ /airbus/notebooks
COPY inference_input_test /airbus/inference_input_test
WORKDIR /airbus

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENV CUDA_VISIBLE_DEVICES=all