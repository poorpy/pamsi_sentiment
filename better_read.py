import os

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

srcript_dir = os.path.dirname(__file__)
snip_rel_path = "zdania.txt"
abs_file_path = os.path.join(srcript_dir, snip_rel_path)

T = Tokenizer(num_words=5000)
docs = open(abs_file_path, 'r+')

T.fit_on_texts(docs)

second_docs = open(abs_file_path, 'r+')

docs_list = [line for line in second_docs]

encoded_docs = T.texts_to_sequences(docs_list)
encoded_docs = pad_sequences(encoded_docs, maxlen=100)

print(encoded_docs)

snip_rel_path = "sentiment.txt"
abs_file_path = os.path.join(srcript_dir, snip_rel_path)

sentiments = open(abs_file_path, 'r+')
sent_list = [line for line in sentiments]
sent_list.extend(['0' for i in range(1, 125)])  # MAŁY PATCH XD

# print(len(sent_list))
# print(str(len(encoded_docs)))
