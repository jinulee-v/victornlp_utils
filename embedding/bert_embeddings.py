"""
@module bert_embedding
Various transformer-based embedding modules.

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
import itertools

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from kobert_transformers import get_tokenizer

from .KorBERT_morph_tokenizer import BertTokenizer as KorBertTokenizer

class EmbeddingBERTWordPhr_kor(nn.Module):
  """
  Word-phrase level Embedding model using KoBERT(SKT-brain, 2019).
  Concatenates the hidden state of initial & final wordpiece tokens.
  """
  def __init__(self, config):
    """
    Constructor for EmbeddingBERTWordPhr_kor.
    
    @param self The object pointer.
    @param config Dictionary. Configuration for EmbeddingBERTWordPhr_kor
    """
    super(EmbeddingBERTWordPhr_kor, self).__init__()
    self.tokenizer = get_tokenizer()
    self.model = BertModel.from_pretrained('monologg/kobert')
    self.embed_size = 1536
    self.special_tokens = config['special_tokens']

  def forward(self, inputs):
    """
    Overridden forward().
    
    @param self The object pointer.
    @param inputs List of dictionary. Refer to 'corpus.py' for more details.
    
    @return Tensor(batch, length, self.embed_size)
    """
    device = next(self.parameters()).device
    
    tokens_list = []
    sentences = []
    attention_mask = []
    for input in inputs:
      tokens = self.tokenizer.tokenize('[CLS] '+input['text'].replace(' _ ', ' ').replace('_', '')+' [SEP]')
      tokens_list.append(tokens)
      tokens = self.tokenizer.convert_tokens_to_ids(tokens)
      sentences.append(torch.tensor(tokens, dtype=torch.long))
      attention_mask.append(torch.ones([len(tokens)], dtype=torch.long))
    sentences = nn.utils.rnn.pad_sequence(sentences, batch_first=True).to(device)
    attention_mask = nn.utils.rnn.pad_sequence(attention_mask, batch_first=True).to(device)

    with torch.no_grad():
      output = self.model(sentences, attention_mask, torch.zeros_like(attention_mask))['last_hidden_state']
    embedded = []
    temp = None
    for i, tokens in enumerate(tokens_list):
      embedded.append([])
      for j in range(len(tokens)):
        if tokens[j] == '[SEP]':
          embedded[i].append(output[i][j].repeat(2).unsqueeze(0))
          break
        if tokens[j] == '[CLS]' or tokens[j].startswith('▁'):
          temp = output[i][j]
        if j+1 == len(tokens) or tokens[j+1] == '[SEP]' or tokens[j+1].startswith('▁'):
          temp = torch.cat([temp, output[i][j]], 0)
          embedded[i].append(temp.unsqueeze(0))
          temp = None
      embedded[i] = torch.cat(embedded[i], 0)
      if 'bos' not in self.special_tokens:
        embedded[i] = embedded[i][1:]
      if 'eos' not in self.special_tokens:
        embedded[i] = embedded[i][:-1]
    embedded = nn.utils.rnn.pad_sequence(embedded, batch_first=True)
    return embedded


lexical_morphemes_tag = [
  'NNG', 'NNP', 'NNB', 'NR', 'NP',
  'VV', 'VA', 'VX', 'VCP', 'VCN',
  'MMA', 'MMD', 'MMN' 'MAG', 'MAJ',
  'IC', 'SL', 'SN', 'SH', 'XR', 'NF', 'NV'
]
class EmbeddingBERTMorph_kor(nn.Module):
  """
  Morpheme level Embedding model using KorBERT(ETRI, 2020).
  Initially outputs hidden states for every morphemes,
  but can configurate to select word-phrase level embeddings.
  """
  def __init__(self, config):
    """
    Constructor for EmbeddingBERTMorph_kor.
    
    @param self The object pointer.
    @param config Dictionary. Configuration for EmbeddingBERTMorph_kor
    """
    super(EmbeddingBERTMorph_kor, self).__init__()

    # Load model from configuration
    model_dir = config['file_directory']
    self.tokenizer = KorBertTokenizer.from_pretrained(model_dir)
    self.model = BertModel.from_pretrained(model_dir)
    self.model.eval()
    self.model.requires_grad = False

    self.is_word_phrase_embedding = config['word_phrase']
    self.embed_size = 1536 if self.is_word_phrase_embedding else 768
    self.special_tokens = config['special_tokens']
  
  def forward(self, inputs):
    """
    Overridden forward().
    
    @param self The object pointer.
    @param inputs List of dictionary. Refer to 'corpus.py' for more details.
    
    @return Tensor(batch, length, self.embed_size)
    """
    device = next(self.parameters()).device
    
    tokens_list = []
    sentences = []
    attention_mask = []
    lengths = torch.zeros(len(inputs), dtype=torch.long)
    # List for word phrase/morpheme recovery index
    if self.is_word_phrase_embedding:
      selects = [([[0, 0]] if 'bos' in self.special_tokens else []) for _ in range(len(inputs))]
    else:
      selects = [([0] if 'bos' in self.special_tokens else []) for _ in range(len(inputs))]

    for i, input in enumerate(inputs):
      # Input format:
      # ETRI/SL 에서/JKB 한국어/NNP BERT/SL 언어/NNG 모델/NNG 을/JKO 배포/NNG 하/XSV 었/EP 다/EF ./SF
      pos_text = []
      wp_last_tokens = []
      j = 1
      for word_phrase in input['pos']:
        for morph in word_phrase:
          pos_text.append(morph['text']+'/'+morph['pos_tag'])
          j += 1
        wp_last_tokens.append(j)
      assert len(wp_last_tokens) == input['word_count']
      pos_text = ' '.join(pos_text)

      tokens = self.tokenizer.tokenize('[CLS] '+pos_text+' [SEP]')
      tokens_list.append(tokens)
      token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
      sentences.append(torch.tensor(token_ids, dtype=torch.long))
      attention_mask.append(torch.ones([len(tokens)], dtype=torch.long))
      lengths[i] = len(tokens) - 2 # index of [SEP]

      if self.is_word_phrase_embedding:
        # Word phrase recovery: select last lexical / functional morpheme from each WP
        pos_tag = lambda token: token.split('/')[-1].replace('_', '')
        new_pair = [None, None] # lexical & functinoal index
        pos_index = 0
        for j, token in enumerate(tokens):
          if pos_index >= len(input['pos']):
            break
          if token.endswith('_') and len(token)>1:
            if pos_tag(token) in lexical_morphemes_tag:
              new_pair[0] = j
            if pos_tag(token) not in lexical_morphemes_tag:
              new_pair[1] = j
          # Last morpheme in word phrase
          if j == wp_last_tokens[pos_index]:
            selects[i].append(new_pair)
            new_pair = [None, None] 
            pos_index += 1
      else:
        # Morpheme recovery: select PoS tag(which contains contextual lexical information)
        for j, token in enumerate(tokens):
          if token.endswith('_') and len(token)>1:
            selects[i].append(j)
      
    sentences = nn.utils.rnn.pad_sequence(sentences, batch_first=True).to(device)
    attention_mask = nn.utils.rnn.pad_sequence(attention_mask, batch_first=True).to(device)
    # eos token treatment for word phrase embedding
    if 'eos' in self.special_tokens:
      for length, select in zip(lengths, selects):
        if self.is_word_phrase_embedding:
          select.append([length, length])
        else:
          select.append(length)
    
    # Run BERT model
    with torch.no_grad():
      output = self.model(sentences, attention_mask, torch.zeros_like(attention_mask))['last_hidden_state']

    # Use `selects` to:
    # Return word-phrase embeddings
    if self.is_word_phrase_embedding:
      embedded = []
      for sent, select in zip(output, selects):
        embedded.append(torch.zeros(len(select), self.embed_size).to(device))
        for i, pair in enumerate(select):
          left = pair[0]; right = pair[1]
          if left:
            embedded[-1][i][:self.embed_size//2] = sent[left]
          if right:
            embedded[-1][i][self.embed_size//2:] = sent[right]
      embedded = nn.utils.rnn.pad_sequence(embedded, batch_first=True)
      return embedded

    # Return embeddings for all morphemes
    else:
      embedded = []
      for sent, select in zip(output, selects):
        embedded.append(sent[select])
      embedded = nn.utils.rnn.pad_sequence(embedded, batch_first=True)
      return embedded

class EmbeddingBERT_eng(nn.Module):
  """
  Word-phrase level Embedding model using bert-base-uncased(Google, 2018).
  Concatenates the hidden state of initial & final wordpiece tokens.
  """
  def __init__(self, config):
    """
    Constructor for EmbeddingBERTWordPhr_kor.
    
    @param self The object pointer.
    @param config Dictionary. Configuration for EmbeddingBERTWordPhr_kor
    """
    super(EmbeddingBERT_eng, self).__init__()
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    self.model = BertModel.from_pretrained('bert-base-uncased')
    self.model.eval()
    self.model.requires_grad = False
    self.embed_size = 1536
    self.special_tokens = config['special_tokens']

  def forward(self, inputs):
    """
    Overridden forward().
    
    @param self The object pointer.
    @param inputs List of dictionary. Refer to 'corpus.py' for more details.
    
    @return Tensor(batch, length, self.embed_size)
    """
    device = next(self.parameters()).device
    
    tokens_list = []
    wp_list = []
    sentences = []
    attention_mask = []
    for input in inputs:
      tokens = self.tokenizer.tokenize('[CLS] '+input['text']+' [SEP]')
      tokens_list.append(tokens)
      wp_list.append(input['text'].lower().split(' ') + ['SEP'])
      tokens = self.tokenizer.convert_tokens_to_ids(tokens)
      sentences.append(torch.tensor(tokens, dtype=torch.long))
      attention_mask.append(torch.ones([len(tokens)], dtype=torch.long))
    sentences = nn.utils.rnn.pad_sequence(sentences, batch_first=True).to(device)
    attention_mask = nn.utils.rnn.pad_sequence(attention_mask, batch_first=True).to(device)

    with torch.no_grad():
      output = self.model(sentences, attention_mask, torch.zeros_like(attention_mask))['last_hidden_state']
    embedded = []
    temp = None
    for i, tokens in enumerate(tokens_list):
      embedded.append([])
      wp_i = -1
      for j in range(len(tokens)):
        if tokens[j] == '[SEP]':
          embedded[i].append(output[i][j].repeat(2).unsqueeze(0))
          break
        if tokens[j] == '[CLS]' or wp_list[i][wp_i].startswith(tokens[j]):
          temp = output[i][j]
        if j+1 == len(tokens) or tokens[j+1] == '[SEP]' or wp_list[i][wp_i+1].startswith(tokens[j+1]):
          temp = torch.cat([temp, output[i][j]], 0)
          embedded[i].append(temp.unsqueeze(0))
          temp = None
          wp_i += 1
      embedded[i] = torch.cat(embedded[i], 0)
      if 'bos' not in self.special_tokens:
        embedded[i] = embedded[i][1:]
      if 'eos' not in self.special_tokens:
        embedded[i] = embedded[i][:-1]
    embedded = nn.utils.rnn.pad_sequence(embedded, batch_first=True)
    return embedded
