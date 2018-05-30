import os
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import EnglishStemmer

script_dir = os.path.dirname(__file__)
snip_rel_path = "st/datasetSentences.txt"
snip_abs_path = os.path.join(script_dir, snip_rel_path)

phrase_rel_path = "st/dictionary.txt"
phrase_abs_path = os.path.join(script_dir, phrase_rel_path)

stop_words = stopwords.words('english')
snowball_stemmer = EnglishStemmer()


def tokenize_file(file, delim, positon=1):
    tmp_sentences = []
    tmp_rest = []
    for line in open(file, 'r'):
        tokens = word_tokenize(line.split(delim)[positon])
        tmp_rest.extend([int(item.strip('\n')) for index, item
                         in enumerate(line.split(delim))
                         if index != positon])
        filtered_sentence = [w.lower() for w in tokens
                             if w.lower() not in stop_words and w.isalpha()]
        tmp_sentences.append(list(
            map(snowball_stemmer.stem, filtered_sentence)
        ))
    return (tmp_sentences, tmp_rest)


tokenized_sentences = tokenize_file(snip_abs_path, "\t")[0]
with open("zdania.txt", "w") as f:
    for sentence in tokenized_sentences:
        str_sentence = ""
        for word in sentence:
            str_sentence += (word + " ")
        f.write(str_sentence + "\n")
# (tokenized_phrases, phrase_ids) = tokenize_file(phrase_abs_path, "|", 0)
# tuples_to_dump = []
# for sentence in tokenized_sentences:
    # if sentence in tokenized_phrases:
        # tuples_to_dump.append((sentence, phrase_ids[
            # tokenized_sentences.index(sentence)]))

# pickle.dump(tuples_to_dump, open("zdania_i_id", "w"))
