"""
@module reformatting
Convert different corpus to VictorNLP corpus format.
"""

import os, sys
import json
from torch.utils.data import random_split

def modu_to_victornlp(modu_dp_file, modu_pos_file, modu_ner_file, train_file, dev_file, test_file, labels_file):
  victornlp = {}
  
  # process DP
  modu = json.load(modu_dp_file)
  dp_labels = set()
  for doc in modu['document']:
    for sent in doc['sentence']:
      id = sent.pop('id', None)
      sent['text'] = sent['form']
      sent.pop('form', None)
      sent['word_count'] = len(sent['text'].split())
      sent['dependency'] = sent['DP']
      sent.pop('DP', None)
      for arc in sent['dependency']:
        arc['dep'] = arc['word_id']
        arc.pop('word_id', None)
        arc.pop('word_form', None)
        arc.pop('dependent', None)
        if arc['head'] == -1:
          arc['head'] = 0
        arc['head'] = arc.pop('head')
        arc['label'] = arc.pop('label')
        if arc['label'] not in dp_labels:
          dp_labels.add(arc['label'])
      victornlp[id] = (sent)
  del modu
  dp_labels = sorted(list(dp_labels))

  # process PoS
  modu = json.load(modu_pos_file)
  pos_labels = set()
  for doc in modu['document']:
    for sent in doc['sentence']:
      id = sent.pop('id', None)
      target = victornlp[id]
      target['pos'] = []
      for morph in sent['morpheme']:
        while len(target['pos']) < morph['word_id']:
          target['pos'].append([])
        target['pos'][morph['word_id'] - 1].append({
          'id': morph['id'],
          'text': morph['form'],
          'pos_tag': morph['label']
        })
        pos_labels.add(morph['label'])
  del modu
  pos_labels = sorted(list(pos_labels))
 
  # process NER
  modu = json.load(modu_ner_file)
  ner_labels = set()
  for doc in modu['document']:
    for sent in doc['sentence']:
      id = sent.pop('id', None)
      target = victornlp[id]
      target['named_entity'] = []

      # re-write start/end as morpheme IDs, not character.
      for ne in sent['NE']:
        ne.pop('id', None)
        target['named_entity'].append(ne)
      
      # add ner labels
      ner_labels.add(ne['label'])
  del modu
  ner_labels = sorted(list(ner_labels))

  # Sum up...
  victornlp = list(victornlp.values())
  
  labels = {
    'pos_labels': pos_labels,
    'dp_labels': dp_labels,
    'ner_labels': ner_labels
  }

  print('data count: ', len(victornlp))
  
  print(json.dumps(victornlp[0], indent=4, ensure_ascii=False))
 
  train_len = int(0.8 * len(victornlp))
  dev_len = int(0.1 * len(victornlp))
  test_len = len(victornlp) - train_len - dev_len
  split = (train_len, dev_len, test_len)
  train, dev, test = tuple(random_split(victornlp, split))
  json.dump(list(train), train_file, indent=4, ensure_ascii = False)
  json.dump(list(dev), dev_file, indent=4, ensure_ascii = False)
  json.dump(list(test), test_file, indent=4, ensure_ascii = False)
  json.dump(labels, labels_file, indent=4, ensure_ascii = False)


if __name__ == '__main__':
  os.chdir(sys.argv[1])
  with open('Modu_DP_raw.json') as modu_dp_file, \
       open('Modu_PoS_raw.json') as modu_pos_file, \
       open('Modu_NER_raw.json') as modu_ner_file, \
       open('VictorNLP_kor(Modu)_train.json', 'w') as train_file, \
       open('VictorNLP_kor(Modu)_dev.json', 'w') as dev_file, \
       open('VictorNLP_kor(Modu)_test.json', 'w') as test_file, \
       open('VictorNLP_kor(Modu)_labels.json', 'w') as labels_file:
    modu_to_victornlp(modu_dp_file, modu_pos_file, modu_ner_file, train_file, dev_file, test_file, labels_file)