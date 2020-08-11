"""
Get FastText representation from pretrained embeddings with subword information

http://christopher5106.github.io/deep/learning/2020/04/02/fasttext_pretrained_embeddings_subword_word_representations.html
"""


import os
import fasttext
import numpy as np


def save_embeddings(model, output_dir):
  os.makedirs(output_dir, exist_ok=True)
  np.save(os.path.join(output_dir, "embeddings"), model.get_input_matrix())
  with open(os.path.join(output_dir, "vocabulary.txt"), "w", encoding='utf-8') as f:
    for word in model.get_words():
      f.write(word+"\n")

def load_embeddings(output_dir):
  input_matrix = np.load(os.path.join(output_dir, "embeddings.npy"))
  words = []
  with open(os.path.join(output_dir, "vocabulary.txt"), "r", encoding='utf-8') as f:
    for line in f.readlines():
      words.append(line.rstrip())
  return words, input_matrix

def get_hash(subword, bucket=2000000, nb_words=2000000):
  h = 2166136261
  for c in subword:
    c = ord(c) % 2**8
    h = (h ^ c) % 2**32
    h = (h * 16777619) % 2**32
  return h % bucket + nb_words

def get_subwords(word, vocabulary, minn=5, maxn=5):
  _word = "<" + word + ">"
  _subwords = []
  _subword_ids = []
  if word in vocabulary:
    _subwords.append(word)
    _subword_ids.append(vocabulary.index(word))
    if word == "</s>":
      return _subwords, np.array(_subword_ids)
  for ngram_start in range(0, len(_word)):
    for ngram_length in range(minn, maxn+1):
      if ngram_start+ngram_length <= len(_word):
        _candidate_subword = _word[ngram_start:ngram_start+ngram_length]
        if _candidate_subword not in _subwords:
          _subwords.append(_candidate_subword)
          _subword_ids.append(get_hash(_candidate_subword))
  return _subwords, np.array(_subword_ids)

def get_word_vector(word, vocabulary, embeddings):
  subwords = get_subwords(word, vocabulary)
  return np.mean([embeddings[s] for s in subwords[1]], axis=0)

def tokenize(sentence):
  tokens = []
  word = ""
  for c in sentence:
    if c in [' ', '\n', '\r', '\t', '\v', '\f', '\0']:
      if word:
        tokens.append(word)
        word = ""
      if c == '\n':
        tokens.append("</s>")
    else:
      word += c
  if word:
    tokens.append(word)
  return tokens

def get_sentence_vector(line):
  tokens = tokenize(line)
  vectors = []
  for t in tokens:
    vec = get_word_vector(t, vocabulary, embeddings)
    norm = np.linalg.norm(vec)
    if norm > 0:
      vec /= norm
    vectors.append(vec)
  return np.mean(vectors, axis=0)


vocabulary, embeddings = load_embeddings('cc.en.300')
