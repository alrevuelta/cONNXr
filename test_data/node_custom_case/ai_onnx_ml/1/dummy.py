from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx

#from ..base import Base
from onnx.backend.test.case.base import Base

from node_custom_case import expect

class Dummy(Base):

    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Abs',
            inputs=['x'],
            outputs=['y'],
        )
        x = np.random.randn(3, 4, 5).astype(np.float32)

        # TODO: Some modifications need to be made so that the operator
        # version is provided for the model.

        # TODO: The expect outout won't be hardcoded but calculated
        # using onnx runtime.
        y = abs(x)

        # opset_import should match the folder number
        expect(node, inputs=[x], outputs=[y],
               name='test_dummy_ml', opset_import=6, domain="ai.onnx.ml")