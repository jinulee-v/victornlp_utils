"""
@module pos_tagger/korean

Implements VictorNLP_style wrapper for Khaiii PoS tagger.
"""

from khaiii import KhaiiiApi

from . import register_pos_tagger

@register_pos_tagger('korean')
def wrapper_khaiii(inputs):
  """
  Korean pos tagger. Uses Khaiii from KakaoBrain.
  
  @param inputs List of dictionaries.
  @return inputs Updated inputs.
  """
  api = KhaiiiApi()
  for input in inputs:
    assert 'text' in input
    assert 'word_count' in input

    if 'pos' in input:
      continue
    
    words = input['text'].split()
    assert len(words) == input['word_count']
    
    pos_tagged = komoran.pos(input['text'].replace(' ', '\n'), flatten=False)
    word_i = 0
    pos = [
        [
          {
            'id': None,
            'text': morph.lex,
            'pos_tag': morph.tag
          } for morph in word.morphs
        ] for word in api.analyze(text)
      ]
    assert len(pos) == input['word_count']
    input['pos'] = pos

  return inputs