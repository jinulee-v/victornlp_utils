"""
@module dict_embedding
Various lexicon-based embedding modules.

Class name format: 
  Embedding(description)_language
Requirements:
  __init__(self, config)
    @param config Dictionary. Configuration for Embedding*_* (free-format)
  self.embed_size
  forward(self, inputs)
    @param inputs List of dictionaries. Refer to 'corpus.py' for more details.
    @return output Tensor(batch, length+1, self.embed_size). length+1 refers to the virtual root.
"""

import torch
import torch.nn as nn
import json

class EmbeddingDict(nn.Module):
  """
  Abstract Embedding model that creates embedding based on lexicon data.
  """
  
  def __init__(self, config):
    """
    Constructor for EmbeddingDict.
    All embedding files should match the following format:
     [
        {
          "text": "language",
          "pos": "language/NNG", (optional, any tags)
          "embedding": "1.0 2.1 -0.9 ...", (mandatory if config['from_pretrained']')
          ...
        },
        ...
     ]
     UNK tokens are mapped to the last of the sequence(i.e. -1 th index).
    
    @param self The object pointer.
    @param config Dictionary. Configuration for the embedding.
    """
    super(EmbeddingDict, self).__init__()
    
    self.file_dir = config['file_directory']
    with open(self.file_dir) as embedding_file:
      self.full_information = json.load(embedding_file)
     
    if 'key' in config:
      key = config['key']
    else:
      key = 'text'
        
    self.itos = []
    self.stoi = {}
    self.embeddings = []
    for i, info in enumerate(self.full_information):
      self.itos.append(info[key])
      self.stoi[info[key]] = i
      if config['from_pretrained']:
        self.embeddings.append(torch.FloatTensor([float(x) for x in split(info['embedding'])]).unsqueeze(0))
    
    if config['from_pretrained']:
      # Read pretrained embedding
      self.embed_size = self.embeddings[0].size(0)
      self.embeddings.append(torch.zeros(self.embed_size).unsqueeze(0))
      self.embeddings = nn.Embedding.from_pretrained(torch.cat(self.embeddings, 0))
    else:
      # From-Scratch embedding
      self.embed_size = config['embed_size']
      self.embeddings = nn.Embedding(len(self.itos) + 1, config['embed_size'])
    
    self.requires_grad = bool(config['train'])
    self.stoi['UNK'] = len(self.itos)
    self.itos.append('UNK')


class EmbeddingDictSelectPoS_kor(EmbeddingDict):
  """
  Abstract template for Korean lexicon-style embeddings. Implements practical PoS tag selection.
  
  Korean "pos" format:
        [
          ["", ""],
          ["", "", ""],
          ...
        ]
  
  Follows Sejong morpheme tagging format.
  """
  def __init__(self, config):
    super(EmbeddingDictSelectPoS_kor, self).__init__(config)
    # Convert to tensor for simple indexing
    self.embeddings = self.embeddings.weight
    self.embed_size *= 2
  
  def forward(self, inputs):
    lexical_morphemes_tag = [
      'NNG', 'NNP', 'NNB', 'NR', 'NP',
      'VV', 'VA', 'VX', 'VCP', 'VCN',
      'MM' 'MAG', 'MAJ',
      'IC', 'SL'
    ]
    embedded = []
    for input in inputs:
      assert 'pos' in input
      
      word_i = 1
      word_embedding = torch.zeros(input['word_count'] + 1, self.embed_size).to(self.embeddings.device)
      for word in input['pos']:
        
        for morph in word:
          pos_tag = morph['pos_tag']
          if pos_tag in lexical_morphemes_tag:
            word_embedding[word_i, :self.embed_size//2] = self.embeddings[self.stoi[self.target(morph)]]
          else:
            word_embedding[word_i, self.embed_size//2:] = self.embeddings[self.stoi[self.target(morph)]]
        word_i += 1
            
      embedded.append(word_embedding)
    
    embedded = torch.nn.utils.rnn.pad_sequence(embedded, batch_first=True)
      
    return embedded


class EmbeddingGloVe_kor(EmbeddingDictSelectPoS_kor):
  def __init__(self, config):
    super(EmbeddingGloVe_kor, self).__init__(config)
  
  def target(self, word):
    if word['text'] in self.stoi:
      return word['text']
    else:
      return 'UNK'

class EmbeddingPoS_kor(EmbeddingDictSelectPoS_kor):
  def __init__(self, config):
    super(EmbeddingPoS_kor, self).__init__(config)
  
  def target(self, word):
    if word['pos_tag'] in self.stoi:
      return word['pos_tag']
    else:
      return 'UNK'