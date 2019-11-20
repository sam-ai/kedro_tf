#!/bin/bash

# Fetch data from Drive
# cp /content/drive/'My Drive'/invoice_table_classifier/invoice.jsonl /content/kedro_table_classifier/table_classifier/data/01_raw

mv label_map.pbtxt table_classifier/data/04_features/
mkdir table_classifier/data/02_intermediate/images

git clone https://github.com/sam-ai/object_detection_demo.git
mv object_detection_demo/ table_classifier/src/table_classifier/nodes/

git clone https://github.com/tensorflow/models.git
mv models/ table_classifier/src/table_classifier/nodes/

apt-get install protobuf-compiler python-pil python-lxml python-tk
pip3 install --user Cython contextlib2 pillow lxml matplotlib

pip3 install --user pycocotools

# shellcheck disable=SC2164
cd table_classifier/src/table_classifier/nodes/models/research/
protoc object_detection/protos/*.proto --python_out=.

# shellcheck disable=SC1073
# export PYTHONPATH=':/home/Shiftu-Admin/tfodi_invoice_table_classifier/kedro_tf/table_classifier/src/table_classifier/nodes/models/research/:/home/Shiftu-Admin/tfodi_invoice_table_classifier/kedro_tf/table_classifier/src/table_classifier/nodes/models/research/slim/'
# export PYTHONPATH=':/home/Shiftu-Admin/tfodi_cws_detector/kedro_tf/table_classifier/src/table_classifier/nodes/models/research/:/home/Shiftu-Admin/tfodi_cws_detector/kedro_tf/table_classifier/src/table_classifier/nodes/models/research/slim/'
# export PYTHONPATH=':/home/Shiftu-Admin/tfodi_bol_detector/kedro_tf/table_classifier/src/table_classifier/nodes/models/research/:/home/Shiftu-Admin/tfodi_bol_detector/kedro_tf/table_classifier/src/table_classifier/nodes/models/research/slim/'
# export PYTHONPATH=':$(pwd)/table_classifier/src/table_classifier/nodes/models/research/:$(pwd)/table_classifier/src/table_classifier/nodes/models/research/slim/'
export PYTHONPATH=':/home/Shiftu-Admin/tfodi_bol_address_detector/kedro_tf/table_classifier/src/table_classifier/nodes/models/research/:/home/Shiftu-Admin/tfodi_bol_address_detector/kedro_tf/table_classifier/src/table_classifier/nodes/models/research/slim/'



python3 object_detection/builders/model_builder_test.py

cd ../../../../../

pip3 install --user kedro
kedro info

cd table_classifier/
kedro install
kedro run -n jsonl_to_csv
kedro run -n split_dataframe
kedro run -n generate_tfrecord
kedro run -n download_model
kedro run -n edit_config
kedro run -n train_model
kedro run -n store_frozen
