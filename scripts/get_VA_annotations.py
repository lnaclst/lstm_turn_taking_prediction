import xml.etree.ElementTree
import os
import numpy as np
import pandas as pd
import time as t

def find_nearest(array,value):
    """Takes an array of values and a value to compare it to, and returns the index of the array value closes to the comparison value."""
    idx = (np.abs(array-value)).argmin()
    return idx
t_1 = t.time()

# define filepaths
path_to_features = './data/signals/gemaps_features_50ms/'
path_to_annotations = './data/maptaskv2-1/Data/timed-units/'
path_to_extracted_annotations = './data/extracted_annotations/voice_activity/'
files_feature_list = os.listdir(path_to_features)

# instantiate lists
files_annotation_list = list()
files_output_list = list()

# check if directory to store annotations exists. If not, create it.
if not(os.path.exists(path_to_extracted_annotations)):
    os.makedirs(path_to_extracted_annotations)
    
# create 2 lists: xml file names for each feature file's time annotations, csv files for each feature file
for file in files_feature_list:
    base_name = os.path.basename(file)
    files_annotation_list.append(os.path.splitext(base_name)[0] + '.timed-units.xml')
    files_output_list.append(os.path.splitext(base_name)[0] + '.csv')

for i in range(0,len(files_feature_list)):
    # create array with 1 column of frame times features from csv
    frame_times = np.array(pd.read_csv(path_to_features + files_feature_list[i], delimiter=',',
                                       usecols=[1])['frameTime'])
    # create empty array with 1 row for each frame
    voice_activity = np.zeros((len(frame_times),))
    # get the set of instances of features for each file
    e = xml.etree.ElementTree.parse(path_to_annotations+files_annotation_list[i]).getroot()

    annotation_data = []
    
    for atype in e.findall('tu'):
        #c reate a list of all start and end (indices, times?) instances of feature 'tu' (which means...someone speaking?)
        annotation_data.append((float(atype.get('start')), float(atype.get('end'))))
    
    # Remove any detections less than 90ms as per ref above
    indx = 1
    less_than_25 = 0
    while indx < len(annotation_data):
        if annotation_data[indx][1]-annotation_data[indx][0] < 0.025:
            annotation_data.pop(indx)
            less_than_25 += 1
        else:
            indx += 1
    
    # Find frames that contain voice activity for at least 50% of their duration (25ms)
    for strt_f,end_f in annotation_data:
        start_indx = find_nearest(frame_times, strt_f)
        end_indx = find_nearest(frame_times, end_f) - 1
        voice_activity[start_indx:end_indx + 1] = 1
    
    output = pd.DataFrame([frame_times, voice_activity])
    output = np.transpose(output)
    output.columns = ['frameTimes', 'val']
    output.to_csv(path_to_extracted_annotations + files_output_list[i], float_format='%.6f', sep=',', index=False,
                  header=True)
        
print('total_time: '+str(t.time()-t_1))
print('less than 25 count:' + str(less_than_25))
