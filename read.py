import os
# import numpy as np

srcript_dir = os.path.dirname(__file__)
rel_path_dict = "st/dictionary.txt"
rel_path_label = "st/sentiment_labels.txt"
abs_file_path = os.path.join(srcript_dir, rel_path_dict)

dictionary = {}
with open(abs_file_path, 'r+') as f:
    for line in f:
        (val, key) = line.split("|")
        if len(val.split()) == 1:
            dictionary[int(key)] = [val.split()]

labels = {}
abs_file_path = os.path.join(srcript_dir, rel_path_label)
with open(abs_file_path, 'r') as f:
    for line in f:
        (key, val) = line.split("|")
        if int(key) in dictionary:
            labels[int(key)] = float(val.strip())

# dictionary_arr = np.array()
# for key, val in dictionary:
    # for word in val:
        # dictionary_arr.append((word, labels[key]))

# filtered_dict = {}
