import logging
from typing import Any, Dict, List, Tuple
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import base64
import os
import cv2
import subprocess

import os
import shutil
import glob
import urllib.request
import tarfile

import re

os.environ['PYTHONPATH'] += ':/content/kedro_tf/table_classifier/src/table_classifier/nodes/models' \
                            '/research/:/content/kedro_tf/table_classifier/src/table_classifier/nodes' \
                            '/models/research/slim/'


# from object_detection.utils import label_map_util


def jsonl_image_csv(data: List,
                    store_image_path: str) -> pd.DataFrame:
    image_info_list = []
    cwd = os.getcwd()
    logging.info("storing images at : {}".format(store_image_path))
    # store_dir = parameters['store_image_path']
    for annotations in data:
        imgdata = base64.b64decode(annotations['image'].split(',')[1])
        image_path = os.path.join(cwd,
                                  store_image_path,
                                  annotations['text'].split('.')[0] + '.jpg')
        with open(image_path, 'wb') as f_image:
            f_image.write(imgdata)
        height, width = annotations['height'], annotations['width']
        if 'spans' in annotations:
            if annotations['answer'] == 'accept':
                for span in annotations['spans']:
                    span['label'] = span['label'].replace('\ufeff', '')
                    points = span['points']
                    value = (image_path, width, height, span['label'],
                             int(points[0][0]),
                             int(points[0][1]),
                             int(points[2][0]),
                             int(points[2][1]))
                    image_info_list.append(value)
            else:
                pass
    logging.info("Number of docs : {}".format(str(len(image_info_list))))
    column_name = ["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"]
    logging.info("Number of docs : {}".format(str(len(image_info_list))))
    column_name = ["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"]
    image_info_df = pd.DataFrame(image_info_list, columns=column_name)
    return image_info_df


def split_data(data: pd.DataFrame,
               test_data_ratio: int) -> List:
    logging.info("split with ratio -> {}".format(str(test_data_ratio)))
    data = data.sample(frac=1).reset_index(drop=True)
    train, test = train_test_split(data, test_size=test_data_ratio)
    return [train, test]


def generate_tfrecord(store_image_path: str,
                      train_csv: str,
                      test_csv: str,
                      label_pbtxt: str
                      ) -> None:
    train_record = 'data/05_model_input/train.record'
    test_record = 'data/05_model_input/test.record'

    base_path = os.path.dirname(__file__)

    subprocess.call(['python3',
                     os.path.join(base_path, "object_detection_demo/generate_tfrecord.py"),
                     '--img_path={}'.format(store_image_path),
                     '--csv_input={}'.format(train_csv),
                     '--output_path={}'.format(train_record),
                     '--label_map={}'.format(label_pbtxt)])

    subprocess.call(['python3',
                     os.path.join(base_path, "object_detection_demo/generate_tfrecord.py"),
                     '--img_path={}'.format(store_image_path),
                     '--csv_input={}'.format(test_csv),
                     '--output_path={}'.format(test_record),
                     '--label_map={}'.format(label_pbtxt)])


def download_model(train_model: str,
                   download_base: str,
                   store_pre_model: str) -> None:
    MODEL_FILE = train_model + '.tar.gz'
    if not (os.path.exists(MODEL_FILE)):
        urllib.request.urlretrieve(download_base + MODEL_FILE, MODEL_FILE)
    tar = tarfile.open(MODEL_FILE)
    tar.extractall()
    tar.close()
    os.remove(MODEL_FILE)
    if os.path.exists(store_pre_model):
        shutil.rmtree(store_pre_model)
    os.rename(train_model, store_pre_model)


# def get_num_classes(pbtxt_fname) -> int:
#     label_map = label_map_util.load_labelmap(pbtxt_fname)
#     categories = label_map_util.convert_label_map_to_categories(
#         label_map, max_num_classes=90, use_display_name=True)
#     category_index = label_map_util.create_category_index(categories)
#     return len(category_index.keys())


def edit_config_pipeline(pipeline_file: str,
                         label_map_pbtxt_fname: str,
                         test_record_fname: str,
                         train_record_fname: str,
                         store_pre_model: str) -> None:
    batch_size = 12
    num_steps = 1212

    fine_tune_checkpoint = os.path.join(store_pre_model, "model.ckpt")
    base_path = os.path.dirname(__file__)
    pipeline_fname = os.path.join(base_path, 'models/research/object_detection/samples/configs/', pipeline_file)

    assert os.path.isfile(pipeline_fname), '`{}` not exist'.format(pipeline_fname)

    num_classes = 1
    with open(pipeline_fname) as f:
        s = f.read()
    with open(pipeline_fname, 'w') as f:
        # fine_tune_checkpoint
        s = re.sub('fine_tune_checkpoint: ".*?"',
                   'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint), s)

        # tfrecord files train and test.
        s = re.sub(
            '(input_path: ".*?)(train.record)(.*?")', 'input_path: "{}"'.format(train_record_fname), s)
        s = re.sub(
            '(input_path: ".*?)(val.record)(.*?")', 'input_path: "{}"'.format(test_record_fname), s)

        # label_map_path
        s = re.sub(
            'label_map_path: ".*?"', 'label_map_path: "{}"'.format(label_map_pbtxt_fname), s)

        # Set training batch_size.
        s = re.sub('batch_size: [0-9]+',
                   'batch_size: {}'.format(batch_size), s)

        # Set training steps, num_steps
        s = re.sub('num_steps: [0-9]+',
                   'num_steps: {}'.format(num_steps), s)

        # Set number of classes num_classes.
        s = re.sub('num_classes: [0-9]+',
                   'num_classes: {}'.format(num_classes), s)

        logging.info("config file ---> {}".format(s))
        f.write(s)


def train_tensorflow_model(pipeline_file: str) -> None:
    base_path = os.path.dirname(__file__)

    pipeline_fname = os.path.join(base_path, 'models/research/object_detection/samples/configs/', pipeline_file)

    store_model_dir = 'data/07_model_output'
    # Number of training steps.
    num_steps = 500  # 200000

    # Number of evaluation steps.
    num_eval_steps = 50

    subprocess.call(['python3',
                     os.path.join(base_path, "models/research/object_detection/model_main.py"),
                     '--pipeline_config_path={}'.format(pipeline_fname),
                     '--model_dir={}'.format(store_model_dir),
                     '--alsologtostderr',
                     '--num_train_steps={}'.format(num_steps),
                     '--num_eval_steps={}'.format(num_eval_steps)
                     ])


def store_frozen_model(pipeline_file: str) -> None:
    store_model_dir = 'data/07_model_output'
    output_directory = 'data/08_frozen_model'

    base_path = os.path.dirname(__file__)

    pipeline_fname = os.path.join(base_path, 'models/research/object_detection/samples/configs/', pipeline_file)

    lst = os.listdir(store_model_dir)
    print(lst)
    lst = [l for l in lst if 'model.ckpt-' in l and '.meta' in l]
    steps = np.array([int(re.findall('\d+', l)[0]) for l in lst])
    last_model = lst[steps.argmax()].replace('.meta', '')

    last_model_path = os.path.join(store_model_dir, last_model)

    subprocess.call(['python3',
                     os.path.join(base_path, "models/research/object_detection/export_inference_graph.py"),
                     '--input_type=image_tensor',
                     '--pipeline_config_path={}'.format(pipeline_fname),
                     '--alsologtostderr',
                     '--output_directory={}'.format(output_directory),
                     '--trained_checkpoint_prefix={}'.format(last_model_path)
                     ])
