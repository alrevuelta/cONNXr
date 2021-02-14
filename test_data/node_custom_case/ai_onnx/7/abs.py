import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from node_custom_case import expect

class Abs(Base):

    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Abs',
            inputs=['x'],
            outputs=['y'],
        )
        x = np.array([-0.5, 0, 0.5]).astype(np.float32)
        expect(node,
               inputs=[x],
               name='test_abs_custom',
               opset_import=7,
               domain="ai.onnx")