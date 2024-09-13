# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING
from latched.model_exporters.base import BaseModelExporter

if TYPE_CHECKING:
    from latched.model_wrappers.base import BaseModelWrapper


class OpenVINOExporter(BaseModelExporter):
    """
    Export the model to OpenVINO.
    """

    @classmethod
    def run(cls, model_wrapper: BaseModelWrapper, output_name: str = "model.xml") -> None:
        pass
