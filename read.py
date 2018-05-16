import os

srcript_dir = os.path.dirname(__file__)
rel_path_dict = "st/dictionary.txt"
rel_path_label = "st/sentiment_labels.txt"
abs_file_path = os.path.join(srcript_dir, rel_path_dict)

dictionary = {}
with open(abs_file_path, 'r+') as f:
    for line in f:
        (val, key) = line.split("|")
        dictionary[int(key)] = val

labels = {}
