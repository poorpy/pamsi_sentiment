import os

from keras.preprocessing.text import Tokenizer, text_to_word_sequence

srcript_dir = os.path.dirname(__file__)
snip_rel_path = "zdania.txt"
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

second_docs = open(abs_file_path, 'r+')

docs_list = [line for line in second_docs]

encoded_docs = [text_to_word_sequence(line, lower=False, split=" ") for line in docs_list]
print('Zakdowane rzeczy:')
# print(encoded_docs)
