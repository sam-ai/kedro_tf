from os.path import isfile
from typing import Any, Union, Dict, List

import pandas as pd
import jsonlines


from kedro.io import AbstractDataSet


class JsonlLocalDataSet(AbstractDataSet):

    def _describe(self) -> Dict[str, Any]:
        return dict(
            filepath=self._filepath,
            load_args=self._load_args,
            save_args=self._save_args,
        )

    def __init__(
        self,
        filepath: str,
        load_args: Dict[str, Any] = None,
        save_args: Dict[str, Any] = None,
    ) -> None:

        self._filepath = filepath
        default_save_args = {}
        default_load_args = {}

        self._load_args = (
            {**default_load_args, **load_args}
            if load_args is not None
            else default_load_args
        )
        self._save_args = (
            {**default_save_args, **save_args}
            if save_args is not None
            else default_save_args
        )

    def _load(self) -> List:
        with jsonlines.open(self._filepath, mode='r') as f:
            image_info = list(f)
        return image_info

    def _save(self, data: pd.DataFrame) -> None:
        NotImplemented()

    def _exists(self) -> bool:
        return isfile(self._filepath)
