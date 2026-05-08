import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load(name, relpath):
    path = ROOT / relpath
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

load('preprocessing', 'src/preprocessing.py')
utils = load('utils', 'src/utils.py')

print('utils module:', getattr(utils, '__file__', None))
print('data file used (in utils):', getattr(utils, 'data_path', 'unknown'))
print('intent count:', len(utils.intent_to_responses))
print('texts count:', len(utils.texts))
print('example labels:', list(utils.intent_to_responses.keys())[:20])
print('example text-label pairs (first 10):')
for t, l in zip(utils.texts[:10], utils.labels[:10]):
    print('-', t, '=>', l)
