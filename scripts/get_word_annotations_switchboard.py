# script written by Elliott Gruzin

import xml.etree.ElementTree
import os
import numpy as np
import pandas as pd
import time as t
import pickle
import sys
import nltk
import re

nltk.download('punkt')

# takes 36 seconds for 50ms version, 2 mins for 10ms version
dataset = "switchboard_data"  # default should be "data" - change if doing cross-corpora tests

# select settings for 50ms (0) or 10ms (1) features
if len(sys.argv) == 2:
    speed_setting = int(sys.argv[1])
else:
    speed_setting = 0 # 0 for 50ms, 1 for 10ms

if speed_setting == 0:
    path_to_features = f'./{dataset}/signals/gemaps_features_processed_50ms/znormalized/'
    path_to_extracted_annotations = f'./{dataset}/extracted_annotations/words_advanced_50ms_raw/'
    frame_delay = 2  # word should only be output 100 ms after it is said
    max_len_setting = 2  # using 2 for the moment for the purpose of speed
elif speed_setting == 1:
    path_to_features = f'./{dataset}/signals/gemaps_features_processed_10ms/znormalized/'
    path_to_extracted_annotations = f'./{dataset}/extracted_annotations/words_advanced_10ms_raw/'
    frame_delay = 10
    max_len_setting = 2


def find_nearest(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx
t_1 = t.time()

path_to_annotations = '/group/corpora/public/switchboard/nxt/xml/terminals/'
word_to_ix = pickle.load(open('./data/extracted_annotations/word_to_ix.p', 'rb'))

if not(os.path.exists(path_to_extracted_annotations)):
    os.mkdir(path_to_extracted_annotations)
files_feature_list = os.listdir(path_to_features)
files_annotation_list = list()
files_output_list = list()

for file in files_feature_list:
    base_name = os.path.basename(file)
    num = base_name.split('.')[0][3:]
    if base_name.split('.')[1] == 'g':
        speaker = 'A'
    elif base_name.split('.')[1] == 'f':
        speaker = 'B'
    files_annotation_list.append('/group/corpora/public/switchboard/nxt/xml/terminals/sw{}.{}.terminals.xml'.format(num,speaker))
    files_output_list.append('sw{}.{}.csv'.format(num, speaker))

#%% Create delayed frame annotations

max_len = 0
for i in range(0, len(files_feature_list)):

    print('percent done files create:'+str(i/len(files_feature_list))[0:4])
    frame_times=np.array(pd.read_csv(path_to_features+files_feature_list[i],delimiter=',',usecols = [0])['frame_time'])
    word_values = np.zeros((len(frame_times), max_len_setting))
    check_next_word_array = np.zeros((len(frame_times),))
    e = xml.etree.ElementTree.parse(files_annotation_list[i]).getroot()
    annotation_data = []
    stored_words = []
    regex = re.compile(r"-$|--|^-")
    for atype in e.findall('word'):
        word_frame_list = []
        target_word = atype.get('orth')
        target_word = target_word.strip()
        target_word = target_word.lower()
        is_disfluency = re.search(regex, target_word)
        if is_disfluency:
            word_frame_list = ['--disfluency_token--']
        else:
            word_frame_list = nltk.word_tokenize(target_word)

        curr_words = []
        for wrd in word_frame_list:
            try:
                curr_words.append(word_to_ix[wrd])
            except KeyError:  # allow for unknown words
                curr_words.append(word_to_ix["--unk--"])

        if len(curr_words) > max_len:
            max_len = len(curr_words)
            curr_words = curr_words[:max_len_setting]

        try:
            end_indx_advanced = find_nearest(frame_times, float(atype.get('{http://nite.sourceforge.net/}end'))) \
                                + frame_delay
        except ValueError:
            if atype.get('{http://nite.sourceforge.net/}end') == 'non-aligned':
                # stored_words.append(target_word) ### for now... ignore -- but this isn't a long term solution
                continue
            if atype.get('{http://nite.sourceforge.net/}end') == 'n/a':
                continue
        if end_indx_advanced < len(word_values):
            arr_strt_indx = np.min(np.where(word_values[end_indx_advanced] == 0)[0])
            arr_end_indx = arr_strt_indx + len(curr_words)
            if arr_end_indx < max_len_setting:

                word_values[end_indx_advanced][arr_strt_indx:arr_end_indx] = np.array(curr_words)

    # output = pd.DataFrame([frame_times,word_values])
    if files_output_list[i][7] == 'A':
        speaker = 'g'
    elif files_output_list[i][7] == 'B':
        speaker = 'f'
    name = files_output_list[i][:2]+'0'+files_output_list[i][2:7]+speaker+'.csv'
    output = pd.DataFrame(np.concatenate([np.expand_dims(frame_times, 1), word_values], 1).transpose())
    output = np.transpose(output)
    output.columns = ['frameTimes'] + [str(n) for n in range(max_len_setting)]
    output.to_csv(path_to_extracted_annotations + name, float_format='%.6f', sep=',', index=False ,header=True)

print('total_time: '+str(t.time()-t_1))
