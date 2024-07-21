import numpy as np
import onnx
import onnxruntime


def dummy_inference(model: onnx.ModelProto):
    onnx.checker.check_model(model, full_check=True)
    inputs = {}
    sess = onnxruntime.InferenceSession(model.SerializeToString())
    for node in sess.get_inputs():
        input_ = np.array([[None]])
        if node.type == "tensor(string)":
            input_ = input_.astype(np.str_)
        else:
            input_ = input_.astype(np.float32)
        inputs[node.name] = input_
    outputs = [node.name for node in sess.get_outputs()]
    dat = sess.run(outputs, inputs)
    dat = dict(zip(outputs, dat))
    return dat
