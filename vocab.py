from os import walk
import pandas as pd
import pickle


"""Script creates 2 dictionaries: conv_dict, which contains word count by conversation and total_dict, which contains
total word count in the data set."""

# Total vocab count
# Vocab count for role
# Vocab count by conversation by role

base_path = '/Users/celestesmith/Docs/GradSchool/sf-Dissertation/roddy_lstm/data/extracted_annotations/'

words_folder = base_path + 'words_advanced_50ms_averaged/'

ix_to_word_file = open(base_path + 'ix_to_word.p','rb')
ix_to_word = pickle.load(ix_to_word_file)

conv_dict = {}
total_dict = {}
for (dirpath, dirnames, filenames) in walk(words_folder):
    for filename in filenames:
        if '.csv' in filename:
            if filename not in conv_dict:
                conv_dict[filename] = {}
            csv = pd.read_csv(words_folder + filename, usecols = ['word'])
            for x in csv['word']:
                if x != 0:
                    if ix_to_word[x] not in total_dict:
                        total_dict[ix_to_word[x]] = 1
                    else:
                        total_dict[ix_to_word[x]] += 1
                    if ix_to_word[x] not in conv_dict[filename]:
                        conv_dict[filename][ix_to_word[x]] = 1
                    else:
                        conv_dict[filename][ix_to_word[x]] += 1

# print(conv_dict)
# print(total_dict)



