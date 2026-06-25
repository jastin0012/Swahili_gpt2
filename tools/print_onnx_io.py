import sys
try:
    import onnxruntime as ort
except Exception as e:
    print('ERROR: onnxruntime not installed:', e)
    sys.exit(2)

if len(sys.argv) < 2:
    print('Usage: python print_onnx_io.py model.onnx')
    sys.exit(1)

path = sys.argv[1]
sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
inputs = sess.get_inputs()
outputs = sess.get_outputs()
print(f'ONNX session inputs: {len(inputs)} outputs: {len(outputs)}')
for i, inp in enumerate(inputs):
    print(f' input[{i}] = name={inp.name} shape={inp.shape} type={inp.type}')
for i, out in enumerate(outputs):
    print(f' output[{i}] = name={out.name} shape={out.shape} type={out.type}')
