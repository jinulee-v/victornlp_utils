#-*- coding:utf-8 -*-
"""
Testing code for KorBERT(ETRI).
This file should not merge into the main branch.
"""
import json
import torch
from embedding.bert_embeddings import EmbeddingBERTMorph_kor
from pos_tagger.pos_tagger import pos_tag_Korean

config = {
  "file_directory": "embedding/data/EmbeddingBERTMorph_kor",
  "word_phrase": 0,
  "special_tokens": {
    "unk": "<unk>",
    "bos": "<bos>",
    "pad": "<pad>"
  }
}

if __name__ == '__main__':
  bert = EmbeddingBERTMorph_kor(config)
  raw_sent = "안녕 세상, BERT를 테스트하기 좋은 아침이야."
  pos_tagged = "[CLS] 안녕/IC 세상/NNG ,/SN BERT/SL 를/JKO 테스트/NNG 하/XSV 기/ETN 좋/VA 은/ETM 아침/NNG 이/VCP 야/EF ./SF"
  data = [
    {
      'text': raw_sent,
      'word_count': len(raw_sent.split())
    }
  ]
  data = pos_tag_Korean(data)
  # print(json.dumps(data, indent=4, ensure_ascii=False))
  result = bert(data)
  print(result.size())