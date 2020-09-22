import xml.etree.ElementTree
import os
import numpy as np
import pandas as pd
import time as t
import pickle
import sys
import json
import nltk
nltk.download('punkt')


def find_nearest(array,value): # function returns the index of an element in an array which most closely matches a specified value
    idx = (np.abs(array-value)).argmin()
    return idx
t_1 = t.time()

path_to_features='./data/signals/gemaps_features_processed_50ms/znormalized/' # uses processed features
path_to_annotations='./data/maptaskv2-1/Data/timed-units/' # voice activity files
# path_to_extracted_annotations='./extracted_annotations/words_advanced_100/'
# if not(os.path.exists(path_to_extracted_annotations)):
#     os.mkdir(path_to_extracted_annotations)
files_feature_list = os.listdir(path_to_features) # file list
files_annotation_list = list()
files_output_list = list()
for file in files_feature_list:
    base_name = os.path.basename(file)
    files_annotation_list.append(os.path.splitext(base_name)[0]+'.timed-units.xml')
    files_output_list.append(os.path.splitext(base_name)[0]+'.csv') # gets all file names in synchronised order

#%% Get vocabulary
no_change, disfluency_count,multi_word_count = 0,0,0
words_from_annotations = []
for i in range(0,len(files_feature_list)):
    # sys.stdout.flush()
    print('percent done vocab build:'+str(i/len(files_feature_list))[0:4])
    e = xml.etree.ElementTree.parse(path_to_annotations+files_annotation_list[i]).getroot() # returns all roots from XML file -- can parse this tree to browse more specific constituents
    for atype in e.findall('tu'): # tu corresponds to transcribed utterances
        target_word = atype.text # harvest the text from the utterances
        target_word = target_word.strip() # strip spaces around
        if '--' in target_word:
            target_word ='--disfluency_token--'
            words_from_annotations.append(target_word)
            disfluency_count += 1
        else:
            target_words = nltk.word_tokenize(target_word) # normalises words, converts into list if multiple
            words_from_annotations.extend( target_words)

vocab = set(words_from_annotations) # turn words into set
word_to_ix = {word: i+1 for i, word in enumerate(vocab)} # +1 is because 0 represents no change
ix_to_word = {word_to_ix[wrd]: wrd for wrd in word_to_ix.keys()}
pickle.dump(word_to_ix,open('./data/extracted_annotations/word_to_ix.p','wb'))
pickle.dump(ix_to_word,open('./data/extracted_annotations/ix_to_word.p','wb'))
json.dump(word_to_ix,open('./data/extracted_annotations/word_to_ix.json','w'),indent=4)

print('disfluency count: '+str(disfluency_count))
print('total_time: '+str(t.time()-t_1))
