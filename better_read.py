import os

from keras.preprocessing.text import Tokenizer

srcript_dir = os.path.dirname(__file__)
snip_rel_path = "st/original_rt_snippets.txt"
abs_file_path = os.path.join(srcript_dir, snip_rel_path)

T = Tokenizer()
docs = open(abs_file_path, 'r+')
T.fit_on_texts(docs)

# with open(abs_file_path, 'r+') as f:
#   for line in f:
#      T.fit_on_texts(line)

print('Licznik slow:')
print(T.word_counts)

print('Ilosc linijek:')
print(T.document_count)

print('Slownik:')
print(T.word_docs)

encoded_docs = T.texts_to_sequences(docs)
print('Zakdowane rzeczy:')
print(encoded_docs)
