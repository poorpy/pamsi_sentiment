#!/usr/bin/python

import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import EnglishStemmer

script_dir = os.path.dirname(__file__)
snip_rel_path = "st/original_rt_snippets.txt"
abs_file_path = os.path.join(script_dir, snip_rel_path)

stop_words = stopwords.words('english')
snowball_stemmer = EnglishStemmer()

tokenized_sentences = []
for line in open(abs_file_path, 'r+'):
    tokens = word_tokenize(line)
    filtered_sentence = [w.lower() for w in tokens
                         if w.lower() not in stop_words and w.isalpha()]
    tokenized_sentences.append(list(
        map(snowball_stemmer.stem, filtered_sentence)
    ))
