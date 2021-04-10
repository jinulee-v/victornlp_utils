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

def preprocessor_WordCount(inputs):
  """
  Adds word_count to inputs.
  """
  for input in inputs:
     assert input['text']
     input['word_count'] = len(input['text'].split())

  return inputs

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
