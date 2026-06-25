import onnxruntime as ort
import numpy as np

MODEL = 'swahili_onnx_bundle/model.onnx'
s = ort.InferenceSession(MODEL, providers=['CPUExecutionProvider'])
print('inputs:', [i.name for i in s.get_inputs()])

ids = [39,23577,22651,11,387,65,2743,331,25496,30]
input_array = np.array([ids], dtype=np.int64)
feeds1 = {'input_ids': input_array}
try:
    print('Running with only input_ids...')
    s.run(None, feeds1)
    print('OK')
except Exception as e:
    print('Error:', e)

feeds2 = {'input_ids': input_array, 'attention_mask': np.ones_like(input_array, dtype=np.int64), 'position_ids': np.arange(input_array.shape[1], dtype=np.int64).reshape(1, -1)}
try:
    print('Running with attention_mask and position_ids...')
    outs = s.run(None, feeds2)
    print('Success. Outputs count:', len(outs))
    for o in outs:
        if isinstance(o, np.ndarray):
            print('out shape', o.shape)
except Exception as e:
    print('Error2:', e)
