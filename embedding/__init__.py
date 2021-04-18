
def register_embedding(cls):
  """
  @brief decorator for embedding classes.
  """
  if 'embeddings' not in globals():
    globals['victornlp_embeddings'] = []
  globals['victornlp_embeddings'].append(cls)
  return cls

try:
  from .bert_embeddings import *
  from .bert_korbert_embedding import *
  try:
    from .bert_kobert_embedding import *
  except:
    raise ImportWarning('KoBERT embedding cannot be loaded:\n  pip install kobert_transformers sentencepiece')
except:
  raise ImportWarning('BERT embeddings cannot be loaded:\n  pip install transformers')

from .dict_embeddings import *
from .dict_kor_wordphrase_embeddings import *