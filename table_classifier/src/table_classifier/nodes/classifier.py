import logging
from typing import Any, Dict, List, Tuple
from sklearn.model_selection import train_test_split


from PIL import Image
import io
import numpy as np
import pandas as pd
import base64
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


# def jsonl_image_csv(data: List,
#                     jsonl_image_store_image_path: str
#                     ) -> pd.DataFrame:
#
#     image_info_list = []
#     cwd = os.getcwd()
#     logging.info("storing images at : {}".format(jsonl_image_store_image_path))
#     for annotations in data:
#         imgdata = base64.b64decode(annotations['image'].split(',')[1])
#         image_path = os.path.join(cwd,
#                                   jsonl_image_store_image_path,
#                                   annotations['text'].split('.')[0] + '.jpg')
#         with open(image_path, 'wb') as f_image:
#             f_image.write(imgdata)
#         height, width = annotations['height'], annotations['width']
#         if 'spans' in annotations:
#             if annotations['answer'] == 'accept':
#                 for span in annotations['spans']:
#                     span['label'] = span['label'].replace('\ufeff', '')
#                     points = span['points']
#                     value = (image_path, width, height, span['label'],
#                              int(points[0][0]),
#                              int(points[0][1]),
#                              int(points[2][0]),
#                              int(points[2][1]))
#                     image_info_list.append(value)
#             else:
#                 pass
#
#     logging.info("Number of docs : {}".format(str(len(image_info_list))))
#     column_name = ["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"]
#     image_info_df = pd.DataFrame(image_info_list, columns=column_name)
#     return image_info_df


def _create_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
    else:
        print("Directory {} already  exists".format(path))


def b64_uri_to_bytes(b64_uri):
    if b64_uri.startswith("data"):
        imgstring = b64_uri.split(',')[1]
        imgdata = base64.b64decode(imgstring)
        return imgdata
    else:
        return None


def jsonl_image_csv(data: List,
                    jsonl_image_store_image_path: str
                    ) -> pd.DataFrame:

    image_info_list = []
    cwd = os.getcwd()
    logging.info("storing images at : {}".format(jsonl_image_store_image_path))
    for annotations in data:

        image_byte_stream = b64_uri_to_bytes(annotations["image"])
        encoded_image_io = io.BytesIO(image_byte_stream)
        image = Image.open(encoded_image_io)

        filename = str(annotations["meta"]["file"])
        file_extension = filename.split(".")[-1].lower()

        if file_extension == "png":
            image_format = '.png'
        elif file_extension in ("jpg", "jpeg"):
            image_format = '.jpg'
        else:
            logging.info("Only 'png', 'jpeg' or 'jpg' files are supported by ODAPI. "
                         "Got {}. Thus treating it as `jpg` file. "
                         "Might cause errors".format(file_extension))
            image_format = '.jpg'

        image_path = os.path.join(cwd,
                                  jsonl_image_store_image_path,
                                  filename.split(".")[0] + image_format)

        # height, width = annotations['height'], annotations['width']
        width, height = image.size
        if 'spans' in annotations:
            if annotations['answer'] == 'accept':
                image.save(image_path)
                for span in annotations['spans']:
                    # span['label'] = span['label'].replace('\ufeff', '')
                    points = np.array(span["points"])
                    xmin, ymin = np.amin(points, axis=0)
                    xmax, ymax = np.amax(points, axis=0)
                    # points need to be normalized
                    xmin = xmin/width
                    ymin = ymin/height
                    xmax = xmax/width
                    ymax = ymax/height
                    assert xmin < xmax
                    assert ymin < ymax
                    # Clip bounding boxes that go outside the image
                    if xmin < 0:
                        xmin = 0
                    if xmax > width:
                        xmax = width - 1
                    if ymin < 0:
                        ymin = 0
                    if ymax > height:
                        ymax = height - 1

                    value = (image_path, width, height, span['label'],
                             (xmin * width),
                             (ymin * height),
                             (xmax * width),
                             (ymax * height))
                    image_info_list.append(value)
            else:
                pass

    logging.info("Number of docs : {}".format(str(len(image_info_list))))
    column_name = ["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"]
    image_info_df = pd.DataFrame(image_info_list, columns=column_name)
    return image_info_df


def split_data(data: pd.DataFrame,
               split_data_test_data_ratio: int
               ) -> List:

    logging.info("split with ratio -> {}".format(str(split_data_test_data_ratio)))
    data = data.sample(frac=1).reset_index(drop=True)
    train, test = train_test_split(data, test_size=split_data_test_data_ratio)
    return [train, test]


def generate_tfrecord(record_store_image_path: str,
                      generate_tfrecord_train_csv: str,
                      generate_tfrecord_test_csv: str,
                      generate_tfrecord_label_pbtxt: str,
                      generate_tfrecord_store_record_dir: str
                      ) -> None:

    train_record = os.path.join(generate_tfrecord_store_record_dir, 'train.record')
    test_record = os.path.join(generate_tfrecord_store_record_dir, 'test.record')

    base_path = os.path.dirname(__file__)

    try:

        subprocess.check_call(['python3',
                               os.path.join(base_path, "object_detection_demo/generate_tfrecord.py"),
                               '--img_path={}'.format(record_store_image_path),
                               '--csv_input={}'.format(generate_tfrecord_train_csv),
                               '--output_path={}'.format(train_record),
                               '--label_map={}'.format(generate_tfrecord_label_pbtxt)])

    except subprocess.CalledProcessError as e:
        logging.error("error in generating train tfrecord -> {}".format(e))
    except OSError as e:
        logging.error("error in generating train tfrecord -> {}".format(e))

    try:
        subprocess.check_call(['python3',
                               os.path.join(base_path, "object_detection_demo/generate_tfrecord.py"),
                               '--img_path={}'.format(record_store_image_path),
                               '--csv_input={}'.format(generate_tfrecord_test_csv),
                               '--output_path={}'.format(test_record),
                               '--label_map={}'.format(generate_tfrecord_label_pbtxt)])
    except subprocess.CalledProcessError as e:
        logging.error("error in generating test tfrecord -> {}".format(e))
    except OSError as e:
        logging.error("error in generating test tfrecord -> {}".format(e))


def download_model(download_model_train_model: str,
                   download_model_base_url: str,
                   download_model_store_pre_model: str
                   ) -> None:

    MODEL_FILE = download_model_train_model + '.tar.gz'
    if not (os.path.exists(MODEL_FILE)):
        urllib.request.urlretrieve(download_model_base_url + MODEL_FILE, MODEL_FILE)
    tar = tarfile.open(MODEL_FILE)
    tar.extractall()
    tar.close()
    os.remove(MODEL_FILE)
    if os.path.exists(download_model_store_pre_model):
        shutil.rmtree(download_model_store_pre_model)
    os.rename(download_model_train_model, download_model_store_pre_model)


# def get_num_classes(pbtxt_fname) -> int:
#     label_map = label_map_util.load_labelmap(pbtxt_fname)
#     categories = label_map_util.convert_label_map_to_categories(
#         label_map, max_num_classes=90, use_display_name=True)
#     category_index = label_map_util.create_category_index(categories)
#     return len(category_index.keys())


def edit_config_pipeline(edit_config_pipeline_file: str,
                         edit_config_label_map_pbtx: str,
                         edit_config_test_record: str,
                         edit_config_train_record: str,
                         edit_config_store_pre_model: str,
                         edit_config_num_class: int,
                         edit_config_num_steps: int,
                         edit_config_batch_size: int
                         ) -> None:

    fine_tune_checkpoint = os.path.join(edit_config_store_pre_model, "model.ckpt")
    base_path = os.path.dirname(__file__)
    pipeline_fname = os.path.join(base_path, 'models/research/object_detection/samples/configs/',
                                  edit_config_pipeline_file)

    assert os.path.isfile(pipeline_fname), '`{}` not exist'.format(pipeline_fname)

    with open(pipeline_fname) as f:
        s = f.read()
    with open(pipeline_fname, 'w') as f:
        # fine_tune_checkpoint
        s = re.sub('fine_tune_checkpoint: ".*?"',
                   'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint), s)

        # tfrecord files train and test.
        s = re.sub(
            '(input_path: ".*?)(train.record)(.*?")', 'input_path: "{}"'.format(edit_config_train_record), s)
        s = re.sub(
            '(input_path: ".*?)(val.record)(.*?")', 'input_path: "{}"'.format(edit_config_test_record), s)

        # label_map_path
        s = re.sub(
            'label_map_path: ".*?"', 'label_map_path: "{}"'.format(edit_config_label_map_pbtx), s)

        # Set training batch_size.
        s = re.sub('batch_size: [0-9]+',
                   'batch_size: {}'.format(edit_config_batch_size), s)

        # Set training steps, num_steps
        s = re.sub('num_steps: [0-9]+',
                   'num_steps: {}'.format(edit_config_num_steps), s)

        # Set number of classes num_classes.
        s = re.sub('num_classes: [0-9]+',
                   'num_classes: {}'.format(edit_config_num_class), s)

        logging.info("config file ---> {}".format(s))
        f.write(s)


def train_tensorflow_model(train_model_pipeline_file: str,
                           train_model_store_dir: str,
                           train_model_num_steps: str,
                           train_model_num_eval_steps: str
                           ) -> None:

    base_path = os.path.dirname(__file__)
    pipeline_fname = os.path.join(base_path, 'models/research/object_detection/samples/configs/',
                                  train_model_pipeline_file)

    try:
        subprocess.call(['python3',
                         os.path.join(base_path, "models/research/object_detection/model_main.py"),
                         '--pipeline_config_path={}'.format(pipeline_fname),
                         '--model_dir={}'.format(train_model_store_dir),
                         '--alsologtostderr',
                         '--num_train_steps={}'.format(train_model_num_steps),
                         '--num_eval_steps={}'.format(train_model_num_eval_steps)
                         ])
    except subprocess.CalledProcessError as e:
        logging.error("error in training tensorflow-> {}".format(e))
    except OSError as e:
        logging.error("error in training tensorflow -> {}".format(e))


def store_frozen_model(store_frozen_pipeline_file: str,
                       store_frozen_store_model_dir: str,
                       store_frozen_output_directory: str
                       ) -> None:
    base_path = os.path.dirname(__file__)

    pipeline_fname = os.path.join(base_path, 'models/research/object_detection/samples/configs/',
                                  store_frozen_pipeline_file)

    lst = os.listdir(store_frozen_store_model_dir)
    logging.info("save checkpoints -> ")
    print(lst)
    lst = [l for l in lst if 'model.ckpt-' in l and '.meta' in l]
    steps = np.array([int(re.findall('\d+', l)[0]) for l in lst])
    last_model = lst[steps.argmax()].replace('.meta', '')

    last_model_path = os.path.join(store_frozen_store_model_dir, last_model)

    try:
        subprocess.call(['python3',
                         os.path.join(base_path, "models/research/object_detection/export_inference_graph.py"),
                         '--input_type=image_tensor',
                         '--pipeline_config_path={}'.format(pipeline_fname),
                         '--alsologtostderr',
                         '--output_directory={}'.format(store_frozen_output_directory),
                         '--trained_checkpoint_prefix={}'.format(last_model_path)
                         ])
    except subprocess.CalledProcessError as e:
        logging.error("error in storing frozen model -> {}".format(e))
    except OSError as e:
        logging.error("error in storing frozen model -> {}".format(e))
