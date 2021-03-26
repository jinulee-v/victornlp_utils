"""
@module reformatting
Convert different corpus to VictorNLP corpus format.
"""

import os
import json
from torch.utils.data import random_split

from ..victornlp_utils.pos_tagger.pos_tagger import *

def modu_dp_to_victornlp(modu_file, train_file, test_file, labels_file):
  
  modu = json.load(modu_file)
  victornlp = []
  
  labels = []
  for doc in modu['document']:
    for sent in doc['sentence']:
      sent.pop('id', None)
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
        if arc['label'] not in labels:
          labels.append(arc['label'])
        if arc['head'] == -1:
          arc['head'] = 0
      victornlp.append(sent)
  pos_tag_kor(victornlp)
  labels.sort()
 
  print('data count: ', len(victornlp))
  
  print(json.dumps(victornlp[0], indent=4, ensure_ascii=False))
 
  split = (int(0.9*len(victornlp)), len(victornlp)-int(0.9*len(victornlp)))
  train, test = tuple(random_split(victornlp, split))
  json.dump(list(train), train_file, indent=4, ensure_ascii = False)
  json.dump(list(test), test_file, indent=4, ensure_ascii = False)
  json.dump(labels, labels_file, indent=4, ensure_ascii = False)


if __name__ == '__main__':
  os.chdir('corpus')
  with open('Modu_DP_raw.json') as modu_file, open('VictorNLP_DP_kor(Modu)_train.json', 'w') as train_file, open('VictorNLP_DP_kor(Modu)_test.json', 'w') as test_file, open('VictorNLP_DP_kor(Modu)_labels.json', 'w') as labels_file:
    modu_dp_to_victornlp(modu_file, train_file, test_file, labels_file)