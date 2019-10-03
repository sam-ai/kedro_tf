#!/bin/bash

# Fetch data from Drive
# cp /content/drive/'My Drive'/invoice_table_classifier/invoice.jsonl /content/kedro_table_classifier/table_classifier/data/01_raw

git clone https://github.com/sam-ai/object_detection_demo.git
mv object_detection_demo/ src/table_classifier/nodes/

git clone https://github.com/tensorflow/models.git
mv models/ src/table_classifier/nodes/

apt-get install protobuf-compiler python-pil python-lxml python-tk
pip3 install Cython contextlib2 pillow lxml matplotlib

pip3 install pycocotools

# shellcheck disable=SC2164
cd src/table_classifier/nodes/models/research/
protoc object_detection/protos/*.proto --python_out=.

export PYTHONPATH=':/content/kedro_table_classifier/table_classifier/src/table_classifier/nodes/models/research/:/content/kedro_table_classifier/table_classifier/src/table_classifier/nodes/models/research/slim/'
python3 object_detection/builders/model_builder_test.py

cd ../../../../../






