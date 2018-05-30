import os

from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
from nltk.tokenize import word_tokenize

# nltk.download('punkt') to dowload if not found

script_dir = os.path.dirname(__file__)
snip_rel_path = "st/datasetSentences.txt"
snip_abs_path = os.path.join(script_dir, snip_rel_path)

phrase_rel_path = "st/dictionary.txt"
phrase_abs_path = os.path.join(script_dir, phrase_rel_path)

sentiment_rel_path = "st/sentiment_labels.txt"
sentiment_abs_path = os.path.join(script_dir, sentiment_rel_path)

stop_words = stopwords.words('english')
snowball_stemmer = EnglishStemmer()

sentence_file_name = "zdania.txt"
new_sentiment_filename = "sentiment.txt"


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
    return tmp_sentences, tmp_rest


def save_sentences(file, sentences):
    with open(file, "w") as f:
        for sentence in sentences:
            str_sentence = ""
            for word in sentence:
                str_sentence += (word + " ")
            f.write(str_sentence + "\n")


def count_blank_lines(file):
    with open(file, "r") as f:
        i = 0
        for line in f:
            if line == '\n':
                i += 1
                print("found an end of line %d", i)


def create_sentiments(sentiment_filename, old_sentiment_filename):
    id_sen = {}
    with open(old_sentiment_filename, "r") as sen:
        for line in sen:
            ID, sen = line.split("|")
            id_sen.update({ID: sen})
    with open(sentiment_filename, "w") as sen_dump:
        for ID in ids_to_dump:
            sen_dump.write(str(id_sen[str(ID)]))


if __name__ == "__main__":
    tokenized_sentences = tokenize_file(snip_abs_path, "\t")[0]
    save_sentences(sentence_file_name, tokenized_sentences)

    (tokenized_phrases, phrase_ids) = tokenize_file(phrase_abs_path, "|", 0)
    ids_to_dump = []
    for sentence in tokenized_sentences:
        if sentence in tokenized_phrases:
            ids_to_dump.append((phrase_ids[
                tokenized_sentences.index(sentence)]))

    create_sentiments(new_sentiment_filename, sentiment_abs_path)
    count_blank_lines(sentence_file_name, )
