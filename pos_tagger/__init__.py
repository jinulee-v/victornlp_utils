import warnings

def register_pos_tagger(fn):
  """
  @brief decorator for embedding classes.
  """
  if 'victornlp_pos_tagger' not in globals():
    globals()['victornlp_pos_tagger'] = {}
  victornlp_pos_tagger[fn.__name__] = fn
  return fn

try:
  from .korean import *
except:
  warnings.warn('Komoran PoS tagger cannot be loaded:\n  pip install konlpy; sudo apt-get install default-jre')