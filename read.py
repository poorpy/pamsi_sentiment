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
sentiment_file_name = "sentiment.txt"


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


def read_sentiment(sentiment_filename):
    with open(sentiment_filename, "r") as old_sentiment:
        id_sen = dict(line.strip("\n").split("|") for line in old_sentiment)
    return id_sen


def filter_ids(sentence_token_list, phrase_token_dict):
    return_dict = {}
    for token in sentence_token_list:
        if str(token) in phrase_token_dict:
            return_dict.update({str(token): phrase_token_dict[str(token)]})
    return return_dict


def save_value_to_file(dict_to_save, value_file):
    with open(value_file, "w") as v:
        for value in dict_to_save.values():
            v.write(value + "\n")


if __name__ == "__main__":
    tokenized_sentences = tokenize_file(snip_abs_path, "\t")[0]
    tokenized_phrases, phrase_ids = tokenize_file(phrase_abs_path, "|", 0)
    phrase_id_dict = {str(key): value for key, value
                      in zip(tokenized_phrases, phrase_ids)}
    sentence_id_dict = filter_ids(tokenized_sentences, phrase_id_dict)
    id_sentiment_dict = read_sentiment(sentiment_abs_path)
    sentence_sentiment_dict = {key: id_sentiment_dict[
        str(sentence_id_dict[key])] for key in sentence_id_dict}
    save_sentences(sentence_file_name, tokenized_sentences)
    save_value_to_file(sentence_sentiment_dict, sentiment_file_name)
