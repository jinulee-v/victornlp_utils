import warnings

pos_taggers = {}
def register_pos_tagger(name):
  """
  @brief decorator for embedding classes.
  """
  def decorator(fn):
    pos_taggers[name] = fn
    return fn
  return decorator

try:
  from .korean import *
except:
  warnings.warn('Komoran PoS tagger cannot be loaded:\n  pip install konlpy; sudo apt-get install default-jre')