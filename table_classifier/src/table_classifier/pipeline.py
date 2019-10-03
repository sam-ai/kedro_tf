from kedro.pipeline import Pipeline, node

from .nodes.classifier import split_data, jsonl_image_csv, generate_tfrecord, download_model, edit_config_pipeline


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
                ["invoice_data", "params:store_image_path"],
                "label_dataframe",
                name="jsonl_to_csv"
            ),
            node(
                split_data,
                ["label_dataframe", "params:test_data_ratio"],
                ["train_dataframe", "test_dataframe"],
                name="split_dataframe"
            ),
            node(
                generate_tfrecord,
                ["params:store_image_path", "params:train_csv", "params:test_csv", "params:label_pbtxt"],
                None,
                name="generate_tfrecord"
            ),
            node(
                download_model,
                ["params:train_model", "params:download_base", "params:store_pre_model"],
                None,
                name="download_model"
            ),
            node(
                edit_config_pipeline,
                ["params:pipeline_file",
                 "params:label_map_pbtxt_fname",
                 "params:test_record_fname",
                 "params:train_record_fname",
                 "params:store_pre_model"],
                None,
                name="edit_config"
            )
        ]
    )
    ###########################################################################

    return pipeline
