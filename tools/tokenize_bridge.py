#!/usr/bin/env python3
import sys, json, os
try:
    from tokenizers import Tokenizer
except Exception as e:
    print('Missing dependency: please pip install -r requirements.txt', file=sys.stderr)
    raise

def main():
    if len(sys.argv) < 2:
        print('Usage: tokenize_bridge.py "text to tokenize"')
        return 2
    text = sys.argv[1]
    # load the tokenizer produced during training (path relative to this script)
    tok_path = os.path.join(os.path.dirname(__file__), '..', 'swahili_onnx_bundle', 'tokenizer.json')
    tok_path = os.path.normpath(tok_path)
    tok = Tokenizer.from_file(tok_path)
    enc = tok.encode(text)
    print(json.dumps({'ids': enc.ids, 'tokens': enc.tokens}))
    return 0

if __name__ == '__main__':
    sys.exit(main())
