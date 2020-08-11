import os
import fasttext
import numpy as np


def save_embeddings(model, output_dir):
  os.makedirs(output_dir, exist_ok=True)
  np.save(os.path.join(output_dir, "embeddings"), model.get_input_matrix())
  with open(os.path.join(output_dir, "vocabulary.txt"), "w", encoding='utf-8') as f:
    for word in model.get_words():
      f.write(word+"\n")


for lang in ["en"]:
  ft = fasttext.load_model('cc.' + lang + '.300.bin')
  save_embeddings(ft, 'cc.' + lang + '.300')
