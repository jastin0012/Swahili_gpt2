from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess, json, os
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

app = FastAPI()

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'swahili_onnx_bundle', 'model.onnx')

class RequestBody(BaseModel):
    input: str = None
    token_ids: list[int] | None = None
    max_tokens: int = 32


def run_tokenize_bridge(text: str):
    # Calls tools/tokenize_bridge.py to get gold token ids
    cmd = ['python', os.path.join(os.path.dirname(__file__), '..', 'tools', 'tokenize_bridge.py'), text]
    try:
        out = subprocess.check_output(cmd, text=True)
        data = json.loads(out)
        return data.get('ids', [])
    except Exception as e:
        raise RuntimeError(f"tokenize_bridge failed: {e}")


def load_session():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"ONNX model not found at {MODEL_PATH}")
    sess = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    return sess


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=-1, keepdims=True)


@app.post('/v1/chat/completions')
def chat(body: RequestBody):
    ids = body.token_ids
    if ids is None:
        if not body.input:
            raise HTTPException(status_code=400, detail='input or token_ids required')
        ids = run_tokenize_bridge(body.input)
    max_tokens = int(body.max_tokens or 32)

    sess = load_session()
    input_names = [i.name for i in sess.get_inputs()]
    output_names = [o.name for o in sess.get_outputs()]

    # Heuristic: find input name for ids
    if 'input_ids' in input_names:
        in_name = 'input_ids'
    else:
        in_name = input_names[0]

    # Prepare initial input
    cur_ids = list(ids)
    generated = []

    # simple greedy loop
    for _ in range(max_tokens):
        input_array = np.array([cur_ids], dtype=np.int64)
        feeds = {in_name: input_array}
        # Always include attention_mask and position_ids (model expects them)
        feeds['attention_mask'] = np.ones_like(input_array, dtype=np.int64)
        seq_len = input_array.shape[1]
        feeds['position_ids'] = np.arange(seq_len, dtype=np.int64).reshape(1, seq_len)
        # try to include past if available in inputs but we don't know names; skip for first iteration
        outputs = sess.run(None, feeds)
        # assume first output is logits: (batch, seq_len, vocab)
        logits = None
        for out in outputs:
            if isinstance(out, np.ndarray) and out.ndim >= 2:
                # pick candidate as logits shape (1, seq_len, vocab)
                if out.shape[-1] > 1:
                    logits = out
                    break
        if logits is None:
            raise HTTPException(status_code=500, detail='Failed to find logits in ONNX outputs')
        # take last token logits
        last = logits[0, -1, :]
        token = int(np.argmax(last))
        generated.append(token)
        cur_ids = [token]

    # decode generated tokens via tokenize_bridge using decode mode (bridge prints tokens)
    # decode generated tokens to text using tokenizer from bundle
    tok_path = os.path.join(os.path.dirname(__file__), '..', 'swahili_onnx_bundle', 'tokenizer.json')
    tok = Tokenizer.from_file(tok_path)
    try:
        decoded = tok.decode(generated)
    except Exception:
        decoded = json.dumps({'token_ids': generated})

    return {'id':'chatcmpl-1','object':'chat.completion','choices':[{'index':0,'message':{'role':'assistant','content':decoded}}]}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('app:app', host='0.0.0.0', port=8080, log_level='info')
