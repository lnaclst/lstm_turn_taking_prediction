# File written by Elliot Gruzin - use this to get word timings for switchboard, instead of get_VA_annotations.py,
# which is written for the format of the maptask annotations

import glob
import xml.etree.ElementTree
import pickle
import os
import numpy as np
import pandas as pd
import time as t
from io import StringIO

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

# file_jar = open('file_list.pkl', 'rb')
# file_set = pickle.load(file_jar)
# file_list= list(file_set)

# for file in file_list:

t_1 = t.time()
path_to_features = './data/signals/gemaps_features_50ms/'
path_to_annotations = '/group/corpora/public/switchboard/nxt/xml/terminals/'
path_to_extracted_annotations = './data/extracted_annotations/voice_activity/'
files_feature_list = os.listdir(path_to_features)
files_annotation_list = list()
files_output_list = list()



########## note that A = g, B = f ###########


for file in files_feature_list:
    base_name = os.path.basename(file)
    num = base_name.split('.')[0][3:]
    print(base_name)
    print(base_name.split('.'))
    if base_name.split('.')[1] == 'g':
        speaker = 'A'
    elif base_name.split('.')[1] == 'f':
        speaker = 'B'
    files_annotation_list.append(f'/group/corpora/public/switchboard/nxt/xml/terminals/sw{num}.{speaker}.terminals.xml')
    files_output_list.append('./data/extracted_annotations/voice_activity/sw0{}.{}.csv'.format(num,base_name.split('.')[1]))


for i in range(0,len(files_feature_list)):
    frame_times=np.array(pd.read_csv(path_to_features+files_feature_list[i],delimiter=',',usecols = [1])['frameTime'])
    voice_activity = np.zeros((len(frame_times),))
    e = xml.etree.ElementTree.parse(files_annotation_list[i]).getroot()
    # print(e.tag, e.attrib)
    # for k,v in e.iteritems():
    #     if not k:
    #         namespaces['myprefix'] = v
    print(files_annotation_list[i])
    annotation_data = []
    prev_end = 0
    next_start = None
    last_unaligned = False
    for atype in e.findall('word'):
        # print(atype.get('{http://nite.sourceforge.net/}start'))
        # print(atype.get('{http://nite.sourceforge.net/}end'))
        #
        # exit()
        try:
            next_start = float(atype.get('{http://nite.sourceforge.net/}start'))

            if last_unaligned == True:
                annotation_data.append((prev_end, next_start))
                last_unaligned = False

            annotation_data.append((float(atype.get('{http://nite.sourceforge.net/}start')), float(atype.get('{http://nite.sourceforge.net/}end'))))
            prev_end = float(atype.get('{http://nite.sourceforge.net/}end'))

        except ValueError:

            if atype.get('{http://nite.sourceforge.net/}start') == 'non-aligned':
                last_unaligned = True
            elif atype.get('{http://nite.sourceforge.net/}end') == 'n/a' and atype.get('{http://nite.sourceforge.net/}start') == 'n/a':
                continue
            elif atype.get('{http://nite.sourceforge.net/}end') == 'n/a':
                next_start = float(atype.get('{http://nite.sourceforge.net/}start'))
            elif atype.get('{http://nite.sourceforge.net/}start') == 'n/a':
                annotation_data.append((next_start, float(atype.get('{http://nite.sourceforge.net/}end'))))
            else:
                raise

    # Then remove any detections less than 90ms as per ref above
    indx = 1
    less_than_25 = 0
    while indx < len(annotation_data):
        if annotation_data[indx][1]-annotation_data[indx][0] < 0.025:
            annotation_data.pop(indx)
            less_than_25 += 1
        else:
            indx += 1

    # find frames that contain voice activity for at least 50% of their duration (25ms)
    for strt_f,end_f in annotation_data:
        try:
            start_indx = find_nearest(frame_times,strt_f)
            end_indx = find_nearest(frame_times,end_f) - 1
            voice_activity[start_indx:end_indx+1]=1
        except:
            print(annotation_data)
            print(frame_times)
            print(strt_f)
            exit()

    output = pd.DataFrame([frame_times,voice_activity])
    output=np.transpose(output)
    output.columns = ['frameTimes','val']
    # uncomment this!!
    output.to_csv(files_output_list[i], float_format = '%.6f', sep=',', index=False,header=True)

print('total_time: '+str(t.time()-t_1))
#print('merge count: '+str(merge_count))
print('less than 25 count:'+ str(less_than_25))
