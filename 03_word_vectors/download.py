#
# https://fasttext.cc/docs/en/crawl-vectors.html
#

import fasttext.util

fasttext.util.download_model('en', if_exists='ignore')  # English
ft = fasttext.load_model('cc.en.300.bin')
