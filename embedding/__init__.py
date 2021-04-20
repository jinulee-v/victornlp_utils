import warnings

def register_embedding(cls):
  """
  @brief decorator for embedding classes.
  """
  if 'victornlp_embeddings' not in globals():
    globals()['victornlp_embeddings'] = {}
  victornlp_embeddings[cls.__name__] = cls
  return cls

try:
  from .bert_embeddings import *
  from .bert_korbert_embedding import *
  try:
    from .bert_kobert_embedding import *
  except:
    warnings.warn('KoBERT embedding cannot be loaded:\n  pip install kobert_transformers sentencepiece')
except:
  warnings.warn('BERT embeddings cannot be loaded:\n  pip install transformers')

from .dict_embeddings import *
from .dict_kor_wordphrase_embeddings import *