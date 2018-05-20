import os

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

stop = stopwords.words('english')

srcript_dir = os.path.dirname(__file__)
snip_rel_path = "st/original_rt_snippets.txt"
abs_file_path = os.path.join(srcript_dir, snip_rel_path)

porter = PorterStemmer()

przyklad = 'runners like running and thus they run and it is fine for me'


def tokenizer_portert(text):
    return [porter.stem(word) for word in text.split()]


print(tokenizer_portert(przyklad))  ## Tylko proba


def tokenizer(text):
    print(text.split())  ## Do debuggowania, zeby widziec co tam robi
    return text.split()


print(tokenizer(przyklad))  ## rowniez tylko do patrzenia co tu w trawie piszczy


def erase_common_words(text):
    tmp = [w for w in tokenizer_portert(text)
           # tutaj było " [-10:] " ale nie mam zielonego pojęcia co to oznacza, po jej usunięciu działa lepiej? XDDDDD
           if w not in stop]
    print(tmp)  ## Debuggowanie
    return tmp


erase_common_words(przyklad)


def get_tokens(file):
    tokens = []
    with open(file, 'r+') as f:
        for line in f:
            tokens.append(erase_common_words(line))
    return sum(tokens, [])

# print(get_tokens(abs_file_path))
