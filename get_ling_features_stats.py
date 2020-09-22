from os import walk
import pandas as pd
import pickle
import numpy as np
import sys

"""
Script Modified from script by Lena Smith https://github.com/lnaclst
Script creates 3 dictionaries: conv_dict, which contains word count by conversation, total_dict, which contains
total word count in the data set, and conv_vecs, which contains vectors with the counts of each word by index in each
conversation. Pickles all 3 dictionaries.
"""

# Total vocab count
# Vocab count for role
# Vocab count by conversation by role

base_path = './data/extracted_annotations/'
words_folder = base_path + 'words_advanced_50ms_averaged/'
conversations_list = sys.argv[1]  # complete, testing or training
conversations_list_file = f'./data/splits/{conversations_list}.txt'
if sys.argv[2] == '0':
    include_f = False
else:
    include_f = True
if sys.argv[3] == '0':
    include_g = False
else:
    include_g = True


# get list of file names to consider for the stats
conversations_to_include = []
with open(conversations_list_file, "r") as file:
    for line in file:
        conversations_to_include.append(line)
files_to_include = []
if include_f is True:
    for line in conversations_to_include:
        print(line)
        files_to_include.append(line.strip() + '.f.csv')
        print(files_to_include[-1])
if include_g is True:
    for line in conversations_to_include:
        print(line)
        files_to_include.append(line.strip() + '.g.csv')
        print(files_to_include[-1])


ix_to_word_file = open(base_path + 'ix_to_word.p', 'rb')
ix_to_word = pickle.load(ix_to_word_file)

print(len(ix_to_word))

conv_vecs = {}
conv_dict = {}
total_dict = {}
conv_types = {}
conv_tokens = {}

for word in ix_to_word.values():
    conv_vecs[word] = 0

for filename in files_to_include:
    print(filename)
    if filename not in conv_dict:
        conv_dict[filename] = {}
    csv = pd.read_csv(words_folder + filename, usecols=['word'])

    if filename not in conv_vecs:
        conv_vecs[filename] = [0]*len(ix_to_word)

    for x in csv['word']:
        if x != 0:  # don't count timestamps with no associated word
            conv_vecs[filename][int(x)] += 1

            if ix_to_word[x] not in total_dict:
                total_dict[ix_to_word[x]] = 1
            else:
                total_dict[ix_to_word[x]] += 1
            if ix_to_word[x] not in conv_dict[filename]:
                conv_dict[filename][ix_to_word[x]] = 1
            else:
                conv_dict[filename][ix_to_word[x]] += 1


print(len(conv_vecs))
print(len(conv_dict))
print(len(total_dict))

pickle.dump(conv_vecs, open(base_path + 'conv_vectors.p', 'wb'))
pickle.dump(conv_dict, open(base_path + 'conv_count_dict.p', 'wb'))
pickle.dump(total_dict, open(base_path + 'total_count_dict.p', 'wb'))

# Write stats to file

total_tokens = sum(total_dict.values())
total_types = len(total_dict)
no_conversations = len(conversations_to_include)

types_per_convo = []
tokens_per_convo = []
for conversation in conv_dict:
    current_conv = conv_dict[conversation]
    types_per_convo.append(len(current_conv))
    tokens_per_convo.append(sum(current_conv.values()))

unique_tokens = 0
types_occurring_five_or_less = 0

for key, value in total_dict.items():
    if value == 1:
        unique_tokens += 1
    if 0 < value <= 5:
        types_occurring_five_or_less += 1

with open(f"{conversations_list}_vocab_stats.txt", "a") as file:
    file.write(f"Vocab stats for ~~ {conversations_list} ~~ \n"
               f"include_f is: {str(include_f)}\t include_g is: {str(include_g)}\n"
               f"Number of conversations: {no_conversations}\n"
               f"Total tokens in dataset: {total_tokens}\n"
               f"Total types in dataset: {total_types}\n"
               f"Mean tokens per conversation: {np.mean(tokens_per_convo)}\n"
               f"Min: {min(tokens_per_convo)}\t Max: {max(tokens_per_convo)} \tSt. dev: {np.std(tokens_per_convo)}\n"
               f"Mean types per conversation: {np.mean(types_per_convo)}\n"
               f"Min: {np.min(types_per_convo)}\t Max: {np.max(types_per_convo)} \tSt. dev: {np.std(types_per_convo)}\n"
               
               f"Number of types appearing once only: {unique_tokens}\n"
               f"Percentage of total types: {unique_tokens/total_types:.4%}\t"
               f"Percentage of total tokens: {unique_tokens/total_tokens:.4%}\n\n"
               
               f"Number of types appearing five times or less: {types_occurring_five_or_less}\n"
               f"Percentage of total types: {types_occurring_five_or_less/total_types:.4%}\t"
               f"Percentage of total tokens: {types_occurring_five_or_less/total_tokens:.4%}\n\n")

    file.write("****************************************************\n\n\n")

