import onnxruntime as ort

MODEL = 'swahili_onnx_bundle/model.onnx'

sess = ort.InferenceSession(MODEL, providers=['CPUExecutionProvider'])
print('inputs:')
for i in sess.get_inputs():
    print(f" - {i.name} shape={i.shape} type={i.type}")
print('outputs:')
for o in sess.get_outputs():
    print(f" - {o.name} shape={o.shape} type={o.type}")
