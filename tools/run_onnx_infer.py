#!/usr/bin/env python3
import sys, json, os
import numpy as np
import onnxruntime as ort

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'swahili_onnx_bundle', 'model.onnx')

def run_tokenize(text):
    import subprocess
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), 'tokenize_bridge.py'), text]
    out = subprocess.check_output(cmd, text=True)
    data = json.loads(out)
    return data.get('ids', [])

def main():
    if len(sys.argv) < 2:
        print('Usage: run_onnx_infer.py "text"')
        return 2
    text = sys.argv[1]
    ids = run_tokenize(text)
    if not ids:
        print('No token ids')
        return 1

    if not os.path.exists(MODEL_PATH):
        print('Model not found:', MODEL_PATH)
        return 2
    sess = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    input_names = [i.name for i in sess.get_inputs()]

    if 'input_ids' in input_names:
        in_name = 'input_ids'
    else:
        in_name = input_names[0]

    arr = np.array([ids], dtype=np.int64)
    feeds = {in_name: arr}
    feeds['attention_mask'] = np.ones_like(arr, dtype=np.int64)
    seq_len = arr.shape[1]
    feeds['position_ids'] = np.arange(seq_len, dtype=np.int64).reshape(1, seq_len)

    outputs = sess.run(None, feeds)
    # find logits-like output
    logits = None
    for out in outputs:
        if isinstance(out, np.ndarray) and out.ndim >= 2 and out.shape[-1] > 1:
            logits = out
            break
    if logits is None:
        print('Failed to obtain logits from model outputs')
        return 3

    last = logits[0, -1, :]
    token = int(np.argmax(last))
    print('Predicted next token id:', token)
    return 0

if __name__ == '__main__':
    sys.exit(main())
