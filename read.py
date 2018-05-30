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
        # split line into tokens, take only meaningful part
        tokens = word_tokenize(line.split(delim)[positon])

        # put all not intresting parts of line into tmp_rest
        tmp_rest.extend([int(item.strip('\n')) for index, item
                         in enumerate(line.split(delim)) if index != positon])

        # filter out all stopwords
        filtered_sentence = [w.lower() for w in tokens
                             if w.lower() not in stop_words and w.isalpha()]

        # to tmp_sentences append stemmed sentence
        tmp_sentences.append([snowball_stemmer.stem(word)
                              for word in filtered_sentence])

    # filter out all empty sentences
    tmp_sentences = [sentence for sentence in tmp_sentences if sentence]
    return tmp_sentences, tmp_rest


def save_sentences(file, sentences):
    with open(file, "w") as f:
        for sentence in sentences:
            f.write(" ".join(sentence) + "\n")


def count_blank_lines(file):
    with open(file, "r") as f:
        print(sum(line.isspace() for line in f))


def create_sentiments(sentiment_filename, old_sentiment_filename, id_to_match):
    with open(old_sentiment_filename, "r") as old_sentiment:
        id_sen = dict(line.split("|") for line in old_sentiment)
    with open(sentiment_filename, "w") as sen_dump:
        for ID in id_to_match:
            sen_dump.write(id_sen[str(ID)])


def filter_ids():
    pass

# We don't need this main? :P
if __name__ == "__main__":
    tokenized_sentences = tokenize_file(snip_abs_path, "\t")[0]
    save_sentences(sentence_file_name, tokenized_sentences)

    # TODO : Good(readable! XD) function finding sentiment for sentence
    (tokenized_phrases, phrase_ids) = tokenize_file(phrase_abs_path, "|", 0)
    ids_to_dump = []
    for sentence in tokenized_sentences:
        if sentence in tokenized_phrases:
            ids_to_dump.append((phrase_ids[
                tokenized_sentences.index(sentence)]))

    create_sentiments(new_sentiment_filename, sentiment_abs_path)
    count_blank_lines(sentence_file_name)  # Function only for debugging
