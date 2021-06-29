"""
@module reformatting_kaistre
Convert KAIST Relation Extraction corpus to VictorNLP corpus format.
Is designed to reformat the file distributed in:
> https://github.com/machinereading/bert-ko-re/tree/master/ko_re_data
"""

import argparse
from ..pos_tagger import pos_taggers
import json

def main(args):
  with open(args.src_file, 'r', encoding='UTF-8') as file:
    lines = file.readlines()
  
  api = KhaiiiApi()

  a = []
  b = []
  scores = []
  for line in lines:
    line = line.split('\t')
    text = line[3].strip()
    e1 = line[1]; e2 = line[2]
    relation = line[0]

    # Find entity
    e1_index = text.find('<e1>')
    e2_index = text.find('<e2>')
    assert e1_index != -1 and e2_index != -1
    if e1_index < e2_index:
      e2_index -= 9
      named_entity = [
        {
          'text': e1,
          'label': '',
          'begin': e1_index,
          'end': e1_index+len(e1)
        },
        {
          'text': e2,
          'label': '',
          'begin': e2_index,
          'end': e2_index+len(e2)
        }
      ]
    else:
      e1_index -= 9
      named_entity = [
        {
          'text': e2,
          'label': '',
          'begin': e2_index,
          'end': e2_index+len(e2)
        },
        {
          'text': e1,
          'label': '',
          'begin': e1_index,
          'end': e1_index+len(e1)
        }
      ]
    
    text = text.replace('<e1>', '').replace('</e1>', '').replace('<e2>', '').replace('</e2>', '')
    text = text.replace(u'\u00A0', ' ').replace(u'\u2009', ' ')
    
    pos = pos_taggers['korean']([{'text': text, 'word_count': len(text.split(' '))}])['pos']
    
    # Manual correction for PoS tagging
    for i, wp in enumerate(pos):
      if len(wp) == 2 and wp[0]['text'] == '가' and wp[1]['text'] == '아':
        pos[i] = [{
          'id': None,
          'text': '가',
          'pos_tag': 'JKS'
        }]
    # Renumbering IDs after modification
    id = 1
    for wp in pos:
      for morph in wp:
        morph['id'] = id
        id += 1
    if id > 200:
      continue

    a.append({
      'text': text,
      'word_count': len(text.split(' ')),
      'pos': pos,
      'named_entity': named_entity,
      'relation': [
        {
          'predicate': 0,
          'subject': 1,
          'label': relation
        }
      ]
    })
  
  
  with open(args.dst_file, 'w', encoding='UTF-8') as file:
    json.dump(a, file, indent=4, ensure_ascii=False)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--src-file')
  parser.add_argument('--dst-file')
  args = parser.parse_args()

  main(args)