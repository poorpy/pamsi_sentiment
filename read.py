import os
import nltk
from nltk.corpus import stopwords
srcript_dir = os.path.dirname(__file__)
snip_rel_path = "st/original_rt_snippets.txt"
abs_file_path = os.path.join(srcript_dir, snip_rel_path)


nltk.download('stopwords')
stop = stopwords.words('english')


def get_tokens(file):
    tokens = []
    with open(file, 'r+') as f:
        for line in f:
            tokens.append(nltk.word_tokenize(line))
    return sum(tokens, [])


def get_filtered_tokens(token_list):
    filtered_tokens = []
    for w in token_list:
        if w not in stop:
            filtered_tokens.append(w)
    return filtered_tokens


print(get_filtered_tokens(get_tokens(abs_file_path)))
