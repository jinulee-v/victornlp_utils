"""
@module dataset
'VictorNLP Framework' corpus formatting functions.

'@param inputs' shares the following format.
[
  {
    'text': (raw sentence)
    'pos': (pos-tagged format)
    'dependency': [
       {
         dep:
         head:
         label:
       }, ...
     ],
     ...
  },
  ...
]
"""

from torch.utils.data import Dataset

dataset_preprocessors = {}
def register_preprocessors(name):
  def decorator(fn):
    dataset_preprocessors[name] = fn
    return fn
  return decorator

from .accuracy import *

class VictorNLPDataset(Dataset):
  """
  Transparent wrapper for 'List of dictionaries' format. Implements map-style Dataset class.
  """
  
  def __init__(self, inputs, preprocessors=[]):
    """
    Constructor for VictorNLPDataset.
    
      @param inputs List of dictionaries.
      @param preprocessors List of callables. preprocessor_*() or other callable formats that take 'inputs' is accepted.
    """
    for preprocessor in preprocessors:
      inputs = preprocessor(inputs)
    self._data = inputs

  def __getitem__(self, idx):
    return self._data[idx]
  
  def __len__(self):
    return len(self._data)
  
  def collate_fn(batch):
    return batch

@register_preprocessors('word-count')
def preprocessor_WordCount(inputs):
  """
  Adds word_count to inputs.
  """
  for input in inputs:
     assert input['text']
     input['word_count'] = len(input['text'].split())

  return inputs

@register_preprocessors('dependency-parsing')
def preprocessor_DependencyParsing(inputs):
  """
  Checks data integrity for dependency parsing.
  
  @param inputs List of dictionaries.
  
  @return modified 'inputs' for dependency parsing
  """
  for input in inputs:
     assert input['text']
     assert input['pos']
     assert len(input['pos']) == input['word_count']
    
  return inputs
