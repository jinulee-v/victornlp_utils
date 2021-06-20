"""
@module pos_tagger/korean

Implements VictorNLP_style wrapper for KOMORAN PoS tagger.
"""

from konlpy.tag import Komoran

from . import register_pos_tagger

@register_pos_tagger('korean')
def wrapper_komoran(inputs):
  """
  Korean pos tagger. Uses Komoran by Shineware, wrapped by KoNLPy package.
  
  @param inputs List of dictionaries.
  @return inputs Updated inputs.
  """
  komoran = Komoran()
  for input in inputs:
    assert 'text' in input
    assert 'word_count' in input

    if 'pos' in input:
      continue
    
    words = input['text'].split()
    assert len(words) == input['word_count']
    
    pos_tagged = komoran.pos(input['text'].replace(' ', '\n'), flatten=False)
    word_i = 0
    pos = []
    for word in pos_tagged:
      # Beginning of the eojeol
      pos.append([])
      
      for text, pos_tag in word:
        pos[word_i].append({
          'text': text,
          'pos_tag': pos_tag
        })
      word_i += 1
    assert len(pos) == input['word_count']
    input['pos'] = pos

  return inputs