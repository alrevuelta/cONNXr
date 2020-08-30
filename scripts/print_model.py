import onnx
model_path = "temp/1.1/onnx-1.1.0/onnx/backend/test/data/node/test_abs/node.pb"
onnx_model = onnx.load(model_path)
print("doc_string", onnx_model.doc_string)
print("domain", onnx_model.domain)
print("ir_version", onnx_model.ir_version)
print("metadata_props", onnx_model.metadata_props)
print("model_version", onnx_model.model_version)
print("opset_import", onnx_model.opset_import)
print("producer_name", onnx_model.producer_name)