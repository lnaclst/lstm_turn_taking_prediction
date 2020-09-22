#Based on script written by Elliott Gruzin

import xml.etree.ElementTree
import os
import numpy as np
import time as t
import pickle
import json
from collections import defaultdict
import nltk
import re
nltk.download('punkt')


def find_nearest(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx
t_1 = t.time()


path_to_features ='./data/signals/gemaps_features_processed_50ms/znormalized/'
path_to_switchboard_annotations = '/group/corpora/public/switchboard/nxt/xml/terminals/'
path_to_maptask_annotations = './data/maptaskv2-1/Data/timed-units/'  # voice activity files
path_to_extracted_annotations = './data/extracted_annotations/voice_activity/'
unknown_word_threshold = 5  # set to 0 if you don't want to have an --unk-- token

files_annotation_list_switchboard = list()
files_annotation_list_maptask = list()
files_feature_list = os.listdir(path_to_features)
files_feature_list_switchboard = list()
files_feature_list_maptask = list()

#split into list of maptask and list of switchboard files
for file in files_feature_list:
    base_name = os.path.basename(file)
    if base_name[0] == 's':
        files_feature_list_switchboard.append(file)
    elif base_name[0] == 'q':
        files_feature_list_maptask.append(file)

########## note that A = g, B = f ########### for switchboard


for file in files_feature_list_switchboard:
    base_name = os.path.basename(file)
    num = base_name.split('.')[0][3:]
    if base_name.split('.')[1] == 'g':
        speaker = 'A'
    elif base_name.split('.')[1] == 'f':
        speaker = 'B'
    files_annotation_list_switchboard.append('/group/corpora/public/switchboard/nxt/xml/terminals/sw{}.{}.terminals.xml'
                                             .format(num, speaker))

for file in files_feature_list_maptask:
    base_name = os.path.basename(file)
    files_annotation_list_maptask.append(os.path.splitext(base_name)[0]+'.timed-units.xml')

no_change, disfluency_count, multi_word_count = 0, 0, 0
words_from_annotations = defaultdict(int)
regex = re.compile(r"-$|--|^-")
# Get vocabulary from switchboard
for i in range(0, len(files_feature_list_switchboard)):
    print('percent done vocab build switchboard:'+str(i/len(files_feature_list_switchboard))[0:4])
    print(files_feature_list_switchboard[i])
    e = xml.etree.ElementTree.parse(files_annotation_list_switchboard[i]).getroot()
    for atype in e.findall('word'):
        target_word = atype.get('orth')
        target_word = target_word.strip()
        target_word = target_word.lower()
        is_disfluency = re.search(regex, target_word)
        if is_disfluency:
            target_word = '--disfluency_token--'
            words_from_annotations[target_word] += 1
            disfluency_count += 1
        else:
            target_words = nltk.word_tokenize(target_word)
            for word in target_words:
                words_from_annotations[word] += 1

# Get vocabulary from maptask
for i in range(0, len(files_feature_list_maptask)):
    print('percent done vocab build maptask:' + str(i/len(files_feature_list_maptask))[0:4])
    print(files_feature_list_maptask[i])
    e = xml.etree.ElementTree.parse(path_to_maptask_annotations+files_annotation_list_maptask[i]).getroot()
    for atype in e.findall('tu'):
        target_word = atype.text
        target_word = target_word.strip()
        target_word = target_word.lower()
        if '--' in target_word:
            target_word ='--disfluency_token--'
            words_from_annotations[target_word] += 1
            disfluency_count += 1
        else:
            target_words = nltk.word_tokenize(target_word)
            for word in target_words:
                words_from_annotations[word] += 1

words_from_annotations["--unk--"] = 0
words_to_remove = set()
for word in words_from_annotations:
    if words_from_annotations[word] <= unknown_word_threshold:
        words_from_annotations["--unk--"] += words_from_annotations[word]
        words_to_remove.add(word)

vocab = set(words_from_annotations.keys())  # turn words into set
print(f'total vocab size is: {len(vocab)}')
vocab.difference_update(words_to_remove)
print(f'vocab size after removing low freq. words: {len(vocab)}')

word_to_ix = {word: i+1 for i, word in enumerate(vocab)} # +1 is because 0 represents no change
ix_to_word = {word_to_ix[wrd]: wrd for wrd in word_to_ix.keys()}
pickle.dump(word_to_ix, open('./data/extracted_annotations/word_to_ix.p', 'wb'))
pickle.dump(ix_to_word, open('./data/extracted_annotations/ix_to_word.p', 'wb'))
json.dump(word_to_ix, open('./data/extracted_annotations/word_to_ix.json', 'w'), indent=4)

print('disfluency count: ' + str(disfluency_count))
print('total_time: ' + str(t.time()-t_1))
