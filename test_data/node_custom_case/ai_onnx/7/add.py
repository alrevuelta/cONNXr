import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from node_custom_case import expect

class Add(Base):

    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Add',
            inputs=['x', 'y'],
            outputs=['sum'],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(3, 4, 5).astype(np.float32)
        expect(node,
               inputs=[x, y],
               name='test_add_custom',
               opset_import=7,
               domain="ai.onnx")