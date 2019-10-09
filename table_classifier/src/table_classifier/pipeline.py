from kedro.pipeline import Pipeline, node

from .nodes.classifier import (split_data,
                               jsonl_image_csv,
                               generate_tfrecord,
                               download_model,
                               edit_config_pipeline,
                               train_tensorflow_model,
                               store_frozen_model)


def create_pipeline(**kwargs):
    """Create the project's pipeline.

    Args:
        kwargs: Ignore any additional arguments added in the future.

    Returns:
        Pipeline: The resulting pipeline.

    """

    pipeline = Pipeline(
        [
            node(
                jsonl_image_csv,
                [
                    "invoice_data",
                    "params:jsonl_image_store_image_path"
                ],
                "label_dataframe",
                name="jsonl_to_csv"
            ),
            node(
                split_data,
                [
                    "label_dataframe",
                    "params:split_data_test_data_ratio"
                ],
                [
                    "train_dataframe",
                    "test_dataframe"
                ],
                name="split_dataframe"
            ),
            node(
                generate_tfrecord,
                [
                    "params:generate_tfrecord_image_path",
                    "params:generate_tfrecord_train_csv",
                    "params:generate_tfrecord_test_csv",
                    "params:generate_tfrecord_label_pbtxt",
                    "params:generate_tfrecord_store_record_dir"
                ],
                None,
                name="generate_tfrecord"
            ),
            node(
                download_model,
                [
                    "params:download_model_train_model",
                    "params:download_model_base_url",
                    "params:download_model_store_pre_model"
                ],
                None,
                name="download_model"
            ),
            node(
                edit_config_pipeline,
                [
                    "params:edit_config_pipeline_file",
                    "params:edit_config_label_map_pbtx",
                    "params:edit_config_test_record",
                    "params:edit_config_train_record",
                    "params:edit_config_store_pre_model",
                    "params:edit_config_num_class",
                    "params:edit_config_num_steps",
                    "params:edit_config_batch_size"
                ],
                None,
                name="edit_config"
            ),
            node(
                train_tensorflow_model,
                [
                    "params:train_model_pipeline_file",
                    "params:train_model_store_dir",
                    "params:train_model_num_steps",
                    "params:train_model_num_eval_steps"
                ],
                None,
                name="train_model"
            ),
            node(
                store_frozen_model,
                [
                    "params:store_frozen_pipeline_file",
                    "params:store_frozen_store_model_dir",
                    "params:store_frozen_output_directory"
                 ],
                None,
                name="store_frozen"
            )
        ]
    )
    ###########################################################################

    return pipeline
