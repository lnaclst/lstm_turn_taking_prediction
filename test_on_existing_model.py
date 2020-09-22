import torch
from lstm_model import LSTMPredictor
from data_loader import TurnPredictionDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import pandas as pd
import numpy as np
from copy import deepcopy
import h5py
from sklearn.metrics import f1_score, roc_curve, confusion_matrix
from pprint import pprint
import pickle
import shutil
import matplotlib.pyplot as plt


num_layers = 1
annotations_dir_train = './data/extracted_annotations/voice_activity/'
annotations_dir_test = './data/extracted_annotations/voice_activity/'
# test_list_path = './data/splits/testing.txt'
# train_list_path = './data/splits/training.txt'  # only used for onsets evaluation to get the classification threshold
prediction_length = 60  # (60 is 3 seconds of prediction)
data_set_select = 0  # 0 for maptask, 1 for mahnob, 2 for switchboard
p_memory = True
train_batch_size = 128


def create_results_directory(directory, test_set, experiment_path):
    print(f"{directory}/{test_set}/{experiment_path}")
    os.makedirs(f"{directory}/{test_set}/{experiment_path}")


def get_train_results_dict(model, train_dataset, train_dataloader, train_list_path):
    # Decide whether to use cuda or not
    use_cuda = torch.cuda.is_available()
    print('Use CUDA: ' + str(use_cuda))
    if use_cuda:
        dtype = torch.cuda.FloatTensor
        dtype_long = torch.cuda.LongTensor
    else:
        dtype = torch.FloatTensor
        dtype_long = torch.LongTensor

    model.eval()
    train_file_list = list(pd.read_csv(train_list_path, header=None, dtype=str)[0])
    train_results_dict = dict()
    train_results_lengths = train_dataset.get_results_lengths()
    for file_name in train_file_list:
        for g_f in ['g','f']:
            # create new arrays for the onset results (the continuous predictions)
            train_results_dict[file_name + '/' + g_f] = np.zeros(
                [train_results_lengths[file_name], prediction_length])
            train_results_dict[file_name + '/' + g_f][:] = np.nan

    for batch_indx, batch in enumerate(train_dataloader):
        # b should be of form: (x,x_i,v,v_i,y,info)
        model_input = []

        for b_i, bat in enumerate(batch): #b_i is each item in the batch i.e. a frame
            if len(bat) == 0:
                model_input.append(bat)
            elif (b_i == 1) or (b_i == 3):
                model_input.append(bat.transpose(0, 2).transpose(1, 2).numpy())
            elif (b_i == 0) or (b_i == 2):
                model_input.append(Variable(bat.type(dtype)).transpose(0, 2).transpose(1, 2))

        y = Variable(batch[4].type(dtype).transpose(0, 2).transpose(1, 2))
        info = batch[5]
        model_output_logits = model(model_input)

        # loss = loss_func_BCE_Logit(model_output_logits,y)
        # loss_list.append(loss.cpu().data.numpy())
        # loss.backward()
        # if grad_clip_bool:
        #     clip_grad_norm(model.parameters(), grad_clip)
        # for opt in optimizer_list:
        #     opt.step()
        file_name_list = info['file_names']
        gf_name_list = info['g_f']
        time_index_list = info['time_indices']
        train_batch_length = y.shape[1]
        #                model_output = torch.transpose(model_output,0,1)
        model_output = torch.transpose(model_output_logits, 0, 1)
        for file_name, g_f_indx, time_indices, batch_indx in zip(file_name_list,
                                                                 gf_name_list,
                                                                 time_index_list,
                                                                 range(train_batch_length)):
            train_results_dict[file_name + '/' + g_f_indx][time_indices[0]:time_indices[1]] = model_output[
                batch_indx].data.cpu().numpy()
    return train_results_dict


def remove_results_directory(directory, test_set, experiment_path):
    """reverses create_results_directory - for use when testing """
    print(f"{directory}/{test_set}/{experiment_path}")
    try:
        os.rmdir(f"{directory}/{test_set}/{experiment_path}")
    except FileNotFoundError:
        pass


def load_args(args_path):
    with open(args_path, "rb") as json_file:
        args_string = json.load(json_file)

    args_dict = json.loads(args_string)
    return args_dict


def load_evaluation_data():
    if 'hold_shift' in locals():
        if isinstance(hold_shift, h5py.File):  # Just HDF5 files
            try:
                hold_shift.close()
            except:
                pass  # Was already closed
    if 'onsets' in locals():
        if isinstance(onsets, h5py.File):  # Just HDF5 files
            try:
                onsets.close()
            except:
                pass  # Was already closed
    if 'overlaps' in locals():
        if isinstance(overlaps, h5py.File):  # Just HDF5 files
            try:
                overlaps.close()
            except:
                pass  # Was already closed

    hold_shift = h5py.File('./data/datasets/hold_shift.hdf5', 'r')
    onsets = h5py.File('./data/datasets/onsets.hdf5', 'r')
    overlaps = h5py.File('./data/datasets/overlaps.hdf5', 'r')

    return hold_shift, onsets, overlaps


def load_model(pickled_model, args_dict, test_data):

    lstm_settings_dict = {  # this works because these variables are set in locals from json_dict
        'no_subnets': args_dict['no_subnets'],
        'hidden_dims': {
            'master': args_dict['hidden_nodes_master'],
            'acous': args_dict['hidden_nodes_acous'],
            'visual': args_dict['hidden_nodes_visual'],
        },
        'uses_master_time_rate': {},
        'time_step_size': {},
        'is_irregular': {},
        'layers': num_layers,
        'dropout': args_dict['dropout_dict'],
        'freeze_glove': args_dict['freeze_glove_embeddings']
    }
    lstm_settings_dict = test_data.get_lstm_settings_dict(
        lstm_settings_dict)  # add some extra items to the lstm settings related to the dataset

    model = LSTMPredictor(lstm_settings_dict=lstm_settings_dict, feature_size_dict=test_data.get_feature_size_dict(),
                          batch_size=train_batch_size, seq_length=args_dict['sequence_length'],
                          prediction_length=prediction_length, embedding_info=test_data.get_embedding_info())
    with open(pickled_model, "rb") as model_file:
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_file))
        else:
            model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))

    return model


def load_test_set(args_dict, test_list_path, test_on_g=True, test_on_f=True):
    test_dataset = TurnPredictionDataset(args_dict['feature_dict_list'], annotations_dir_test, test_list_path,
                                         args_dict['sequence_length'], prediction_length, 'test',
                                         data_select=data_set_select, test_on_f=test_on_f, test_on_g=test_on_g)

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False,
                                 pin_memory=p_memory)

    return test_dataset, test_dataloader


def load_training_set(args_dict, train_list_path, train_on_f=True, train_on_g=True):
    """needed to get the threshold value for prediction at onsets"""
    train_dataset = TurnPredictionDataset(args_dict['feature_dict_list'], annotations_dir_train, train_list_path,
                                          args_dict['sequence_length'], prediction_length, 'train',
                                          data_select=data_set_select, train_on_f=train_on_f, train_on_g=train_on_g)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False, num_workers=0,
                                  drop_last=True, pin_memory=p_memory)
    return train_dataset, train_dataloader


def plot_person_error(name_list, data, results_path, results_key='barchart'):
    y_pos = np.arange(len(name_list))
    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.barh(y_pos, data, align='center', alpha=0.5)
    plt.yticks(y_pos, name_list, fontsize=5)
    plt.xlabel('mean abs error per time frame', fontsize=7)
    plt.xticks(fontsize=7)
    plt.title('Individual Error')
    plt.savefig(results_path + '/' + results_key + '.pdf')


def test(model, test_dataset, test_dataloader, train_results_dict, train_dataset, onset_test_flag=True,
         onset_test_length=[0,60], prediction_at_overlap_flag=True, error_per_person_flag=True,
         test_list_path=None, train_list_path=None):
    losses_test = list()
    results_dict = dict()
    losses_dict = dict()
    batch_sizes = list()
    losses_mse, losses_l1 = [], []

    pause_str_list = ['50ms', '250ms', '500ms']
    overlap_str_list = ['overlap_hold_shift', 'overlap_hold_shift_exclusive']
    onset_str_list = ['short_long']
    results_save = dict()
    for pause_str in pause_str_list + overlap_str_list + onset_str_list:
        results_save['f_scores_' + pause_str] = list()
        results_save['tn_' + pause_str] = list()
        results_save['fp_' + pause_str] = list()
        results_save['fn_' + pause_str] = list()
        results_save['tp_' + pause_str] = list()
    results_save['train_losses'], results_save['test_losses'], results_save['indiv_perf'], results_save[
        'test_losses_l1'] = [], [], [], []

    # Decide whether to use cuda or not
    use_cuda = torch.cuda.is_available()
    print('Use CUDA: ' + str(use_cuda))
    if use_cuda:
        dtype = torch.cuda.FloatTensor
        dtype_long = torch.cuda.LongTensor
    else:
        dtype = torch.FloatTensor
        dtype_long = torch.LongTensor

    # initialise loss functions
    loss_func_L1 = nn.L1Loss()
    loss_func_L1_no_reduce = nn.L1Loss(reduce=False)
    loss_func_BCE = nn.BCELoss()

    model.eval()
    # setup results_dict
    results_lengths = test_dataset.get_results_lengths()
    out_test_list = []
    y_test_list = []

    test_file_list = list(pd.read_csv(test_list_path, header=None, dtype=str)[0])
    data_select_dict = {0: ['f', 'g'],
                        1: ['c1', 'c2'],
                        2: ['A', 'B']}
    for file_name in test_file_list:
        #        for g_f in ['g','f']:
        for g_f in data_select_dict[data_set_select]:
            # create new arrays for the results
            results_dict[file_name + '/' + g_f] = np.zeros([results_lengths[file_name], prediction_length])
            losses_dict[file_name + '/' + g_f] = np.zeros([results_lengths[file_name], prediction_length])

    for batch_indx, batch in enumerate(test_dataloader):

        model_input = []

        for b_i, bat in enumerate(batch):
            if len(bat) == 0:
                model_input.append(bat)
            elif (b_i == 1) or (b_i == 3):
                model_input.append(torch.squeeze(bat, 0).transpose(0, 2).transpose(1, 2).numpy())
            elif (b_i == 0) or (b_i == 2):
                model_input.append(Variable(torch.squeeze(bat, 0).type(dtype)).transpose(0, 2).transpose(1, 2))

        y_test = Variable(torch.squeeze(batch[4].type(dtype), 0))

        info_test = batch[-1]
        batch_length = int(info_test['batch_size'])
        if batch_indx == 0:
            model.change_batch_size_reset_states(batch_length)
        else:
            model.change_batch_size_no_reset(batch_length)

        out_test = model(model_input)
        out_test = torch.transpose(out_test, 0, 1)

        if test_dataset.set_type == 'test':
            file_name_list = [info_test['file_names'][i][0] for i in range(len(info_test['file_names']))]
            gf_name_list = [info_test['g_f'][i][0] for i in range(len(info_test['g_f']))]
            time_index_list = [info_test['time_indices'][i][0] for i in range(len(info_test['time_indices']))]
        else:
            file_name_list = info_test['file_names']
            gf_name_list = info_test['g_f']
            time_index_list = info_test['time_indices']

        # Should be able to make other loss calculations faster
        # Too many calls to transpose as well. Should clean up loss pipeline
        y_test = y_test.permute(2, 0, 1)
        loss_no_reduce = loss_func_L1_no_reduce(out_test, y_test.transpose(0, 1))


        out_test_list.append(out_test)
        y_test_list.extend(y_test)

        for file_name, g_f_indx, time_indices, batch_indx in zip(file_name_list,
                                                                 gf_name_list,
                                                                 time_index_list,
                                                                 range(batch_length)):

            results_dict[file_name + '/' + g_f_indx][time_indices[0]:time_indices[1]] = out_test[
                batch_indx].data.cpu().numpy()
            losses_dict[file_name + '/' + g_f_indx][time_indices[0]:time_indices[1]] = loss_no_reduce[
                batch_indx].data.cpu().numpy()

        loss = loss_func_BCE(F.sigmoid(out_test), y_test.transpose(0, 1))
        # loss = loss_func_BCE_Logit(out_test,y_test.transpose(0,1))
        losses_test.append(loss.data.cpu().numpy())
        batch_sizes.append(batch_length)

        loss_l1 = loss_func_L1(out_test, y_test.transpose(0, 1))
        losses_l1.append(loss_l1.data.cpu().numpy())

    # get weighted mean
    loss_weighted_mean = np.sum(np.array(batch_sizes) * np.squeeze(np.array(losses_test))) / np.sum(batch_sizes)
    loss_weighted_mean_l1 = np.sum(np.array(batch_sizes) * np.squeeze(np.array(losses_l1))) / np.sum(batch_sizes)
    #    loss_weighted_mean_mse = np.sum( np.array(batch_sizes)*np.squeeze(np.array(losses_mse))) / np.sum( batch_sizes )

    for conv_key in test_file_list:

        results_dict[conv_key + '/' + data_select_dict[data_set_select][1]] = np.array(
            results_dict[conv_key + '/' + data_select_dict[data_set_select][1]]).reshape(-1, prediction_length)
        results_dict[conv_key + '/' + data_select_dict[data_set_select][0]] = np.array(
            results_dict[conv_key + '/' + data_select_dict[data_set_select][0]]).reshape(-1, prediction_length)

    hold_shift, onsets, overlaps = load_evaluation_data()

    # get hold-shift f-scores
    length_of_future_window = 20  # (1 second)

    for pause_str in pause_str_list:
        true_vals = list()
        predicted_class = list()
        for conv_key in test_file_list:
            for g_f_key in list(hold_shift[pause_str + '/hold_shift' + '/' + conv_key].keys()):
                g_f_key_not = deepcopy(data_select_dict[data_set_select])
                g_f_key_not.remove(g_f_key)
                for frame_indx, true_val in hold_shift[pause_str + '/hold_shift' + '/' + conv_key + '/' + g_f_key]:
                    # make sure the index is not out of bounds
                    if frame_indx < len(results_dict[conv_key + '/' + g_f_key]):
                        true_vals.append(true_val)
                        if np.sum( # using model outputs to decide a prediction of hold or shift
                                results_dict[conv_key + '/' + g_f_key][frame_indx, 0:length_of_future_window]) > np.sum(
                                results_dict[conv_key + '/' + g_f_key_not[0]][frame_indx, 0:length_of_future_window]):
                            predicted_class.append(0)
                        else:
                            predicted_class.append(1)
        f_score = f1_score(true_vals, predicted_class, average='weighted')
        results_save['f_scores_' + pause_str].append(f_score)
        tn, fp, fn, tp = confusion_matrix(true_vals, predicted_class).ravel() # true negative, false positive etc.
        results_save['tn_' + pause_str].append(tn)
        results_save['fp_' + pause_str].append(fp)
        results_save['fn_' + pause_str].append(fn)
        results_save['tp_' + pause_str].append(tp)
        print('majority vote f-score(' + pause_str + '):' + str(
            f1_score(true_vals, np.zeros([len(predicted_class)]).tolist(), average='weighted')))
    # get prediction at onset f-scores
    # first get best threshold from training data
    g_f_keys = [] #only do this on the speaker role(s) that are in the training set
    if train_dataset.train_on_g: g_f_keys.append('g')
    if test_dataset.train_on_f: g_f_keys.append('f')
    train_file_list = list(pd.read_csv(train_list_path, header=None, dtype=str)[0])
    if onset_test_flag:
        onset_train_true_vals = list()
        onset_train_mean_vals = list()
        onset_threshs = []
        for conv_key in list(set(train_file_list).intersection(onsets['short_long'].keys())):
            for g_f_key in g_f_keys:
                for frame_indx, true_val in onsets['short_long' + '/' + conv_key + '/' + g_f_key]:
                    # make sure the index is not out of bounds

                    if (frame_indx < len(train_results_dict[conv_key + '/' + g_f_key])) and not (
                            np.isnan(np.mean(train_results_dict[conv_key + '/' + g_f_key][frame_indx, :]))):
                        onset_train_true_vals.append(true_val)
                        vals_to_average_train = train_results_dict[conv_key + '/' + g_f_key][frame_indx, :]
                        onset_train_mean_vals.append(
                            np.mean(vals_to_average_train[onset_test_length[0]:onset_test_length[1]]))
        if not(len(onset_train_true_vals) == 0):
            fpr, tpr, thresholds = roc_curve(np.array(onset_train_true_vals), np.array(onset_train_mean_vals))
        else:
            fpr, tpr, thresholds = 0, 0, [0]
        thresh_indx = np.argmax(tpr - fpr)
        onset_thresh = thresholds[thresh_indx]
        onset_threshs.append(onset_thresh)

        g_f_keys = []  # only do this on the speaker role(s) that are in the test set
        if test_dataset.test_on_g == True: g_f_keys.append('g')
        if test_dataset.test_on_f == True: g_f_keys.append('f')
        true_vals_onset, onset_test_mean_vals, predicted_class_onset = [], [], []
        for conv_key in list(set(test_file_list).intersection(onsets['short_long'].keys())):
            for g_f_key in g_f_keys:
                for frame_indx, true_val in onsets['short_long' + '/' + conv_key + '/' + g_f_key]:
                    # make sure the index is not out of bounds
                    if (frame_indx < len(results_dict[conv_key + '/' + g_f_key])) and not (
                    np.isnan(np.mean(results_dict[conv_key + '/' + g_f_key][frame_indx, :]))):
                        true_vals_onset.append(true_val)
                        vals_to_average_test = results_dict[conv_key + '/' + g_f_key][frame_indx, :]
                        onset_mean = np.mean(vals_to_average_test[onset_test_length[0]:onset_test_length[1]])
                        onset_test_mean_vals.append(onset_mean)
                        if onset_mean > onset_thresh:
                            predicted_class_onset.append(1)  # long
                        else:
                            predicted_class_onset.append(0)  # short
        f_score = f1_score(true_vals_onset, predicted_class_onset, average='weighted')
        print(onset_str_list[0] + ' f-score: ' + str(f_score))
        print('majority vote f-score:' + str(
            f1_score(true_vals_onset, np.zeros([len(true_vals_onset), ]).tolist(), average='weighted')))
        results_save['f_scores_' + onset_str_list[0]].append(f_score)
        if not(len(true_vals_onset) == 0):
            tn, fp, fn, tp = confusion_matrix(true_vals_onset, predicted_class_onset).ravel()
        else:
            tn,fp,fn,tp, = 0,0,0,0
        results_save['tn_' + onset_str_list[0]].append(tn)
        results_save['fp_' + onset_str_list[0]].append(fp)
        results_save['fn_' + onset_str_list[0]].append(fn)
        results_save['tp_' + onset_str_list[0]].append(tp)

    # get prediction at overlap f-scores
    if prediction_at_overlap_flag:
        short_class_length = 20
        overlap_min = 2
        eval_window_start_point = short_class_length - overlap_min
        eval_window_length = 10
        for overlap_str in overlap_str_list:
            true_vals_overlap, predicted_class_overlap = [], []

            for conv_key in list(set(list(overlaps[overlap_str].keys())).intersection(set(test_file_list))):
                for g_f_key in list(overlaps[overlap_str + '/' + conv_key].keys()):
                    g_f_key_not = deepcopy(data_select_dict[data_set_select])
                    g_f_key_not.remove(g_f_key)
                    for eval_indx, true_val in overlaps[overlap_str + '/' + conv_key + '/' + g_f_key]:
                        # make sure the index is not out of bounds
                        if eval_indx < len(results_dict[conv_key + '/' + g_f_key]):
                            true_vals_overlap.append(true_val)
                            if np.sum(results_dict[conv_key + '/' + g_f_key][eval_indx,
                                      eval_window_start_point: eval_window_start_point + eval_window_length]) \
                                    > np.sum(results_dict[conv_key + '/' + g_f_key_not[0]][eval_indx,
                                             eval_window_start_point: eval_window_start_point + eval_window_length]):
                                predicted_class_overlap.append(0)
                            else:
                                predicted_class_overlap.append(1)
            f_score = f1_score(true_vals_overlap, predicted_class_overlap, average='weighted')
            print(overlap_str + ' f-score: ' + str(f_score))

            print('majority vote f-score:' + str(
                f1_score(true_vals_overlap, np.ones([len(true_vals_overlap), ]).tolist(), average='weighted')))
            results_save['f_scores_' + overlap_str].append(f_score)
            tn, fp, fn, tp = confusion_matrix(true_vals_overlap, predicted_class_overlap).ravel()
            results_save['tn_' + overlap_str].append(tn)
            results_save['fp_' + overlap_str].append(fp)
            results_save['fn_' + overlap_str].append(fn)
            results_save['tp_' + overlap_str].append(tp)

    # get error per person (to use with plot_person_error())
    if error_per_person_flag:
        bar_chart_labels = []
        bar_chart_vals = []
        for conv_key in test_file_list:
            #        for g_f in ['g','f']:
            for g_f in data_select_dict[data_set_select]:
                losses_dict[conv_key + '/' + g_f] = np.array(losses_dict[conv_key + '/' + g_f]).reshape(-1,
                                                                                                        prediction_length)
                bar_chart_labels.append(conv_key + '_' + g_f)
                bar_chart_vals.append(np.mean(losses_dict[conv_key + '/' + g_f]))

        results_save['test_losses'].append(loss_weighted_mean)
        results_save['test_losses_l1'].append(loss_weighted_mean_l1)
        #    results_save['test_losses_mse'].append(loss_weighted_mean_mse)

        indiv_perf = {'bar_chart_labels': bar_chart_labels,
                      'bar_chart_vals': bar_chart_vals}
        results_save['indiv_perf'].append(indiv_perf)
    # majority baseline:
    # f1_score(true_vals,np.zeros([len(true_vals),]).tolist(),average='weighted')
    return results_save


def get_test_set_name(f, g):
    if f is True and g is True:
        return 'test_on_both'
    elif f is True:
        return 'test_on_f'
    else:
        return 'test_on_g'


def test_on_existing_models(trial_path, test_on_g=True, test_on_f=True, trained_on_g=True, trained_on_f=True,
                            onset_prediction_frames=[0, 60], report_dict_name=None,
                            train_list_path='./data/splits/testing.txt', test_list_path='./data/splits/training.txt'):
    test_path = f'{trial_path}/test'
    # Loop through all the trained models in this trial path
    results_dicts = []
    for directory in os.listdir(test_path):
        if report_dict_name:
            test_set_name = report_dict_name
        else:
            test_set_name = get_test_set_name(test_on_f, test_on_g)
        # paths to stored models, settings, and location for new results
        model_path = f'{test_path}/{directory}/best_model.p'
        settings_path = f'{test_path}/{directory}/settings.json'
        results_path = f"{trial_path}/{test_set_name}/{directory}"

        # remove existing directories (only needed for debugging purposes)
        try:
            shutil.rmtree(results_path)
        except FileNotFoundError:
            pass

        # load settings, model, data and create directories for results
        args = load_args(settings_path)
        test_set, test_loader = load_test_set(args, test_list_path, test_on_g=test_on_g, test_on_f=test_on_f)
        model = load_model(model_path, args, test_set)
        os.makedirs(results_path)

        # use training set to get threshold for onset evaluation
        train_set, train_loader = load_training_set(args, train_list_path, train_on_g=trained_on_g,
                                                    train_on_f=trained_on_f)  # needs to be how model was originally trained

        train_results_dict = get_train_results_dict(model, train_set, train_loader, train_list_path)

        # perform test on loaded model
        model.eval()
        test_results = test(model, test_set, test_loader, train_results_dict, train_set,
                            onset_test_length=onset_prediction_frames,
                            train_list_path=train_list_path, test_list_path=test_list_path)
        with open(results_path + '/results.txt', 'w') as file:
            file.write(str(test_results))
        pickle.dump(test_results, open(results_path + '/results.p', 'wb'))
        plot_person_error(test_results['indiv_perf'][-1]['bar_chart_labels'],
                          test_results['indiv_perf'][-1]['bar_chart_vals'], results_path,
                          results_key='person_error_barchart')

        # store metrics to average across trials
        results_dicts.append(test_results)

        # combine metrics across trials
        eval_metric_list = ['f_scores_50ms', 'f_scores_250ms', 'f_scores_500ms', 'f_scores_overlap_hold_shift',
                            'f_scores_overlap_hold_shift_exclusive', 'f_scores_short_long', 'train_losses',
                            'test_losses', 'test_losses_l1']
        combined_results = {}
        for metric in eval_metric_list:
            combined_results[metric] = []
        for results_dict in results_dicts:
            for metric in eval_metric_list:
                combined_results[metric].append(results_dict[metric])
        # get average across trials
        averaged_results = {}
        for metric in eval_metric_list:
            averaged_results[metric] = np.mean(combined_results[metric])
        combined_results['means'] = averaged_results
        pprint(combined_results)

        json.dump(combined_results, open(trial_path + f'/report_dict_{test_set_name}.json', 'w'), indent=4, sort_keys=True)

if __name__ == "__main__":
 # to run on initial models:
 #    trial_path = './no_subnets/2_Acous_10ms'
 #    test_on_existing_models(trial_path, test_on_f=True, test_on_g=True)
 #    test_on_existing_models(trial_path, test_on_f=False, test_on_g=True)
 #    test_on_existing_models(trial_path, test_on_f=True, test_on_g=False)

    # trial_path = './two_subnets/2_Acous_10ms_Ling_50ms'
    # test_on_existing_models(trial_path, test_on_f=True, test_on_g=True)
    # test_on_existing_models(trial_path, test_on_f=False, test_on_g=True)
    # test_on_existing_models(trial_path, test_on_f=True, test_on_g=False)
    #
    # trial_path = './no_subnets/3_Ling_50ms'
    # test_on_existing_models(trial_path, test_on_f=True, test_on_g=True)
    # test_on_existing_models(trial_path, test_on_f=False, test_on_g=True)
    # test_on_existing_models(trial_path, test_on_f=True, test_on_g=False)

    # trial_path = './f_and_g_no_subnets/3_Acous_10ms_ftrain'
    # test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, trained_on_f=True, trained_on_g=False)
    # test_on_existing_models(trial_path, test_on_f=False, test_on_g=True, trained_on_f=True, trained_on_g=False)
    # test_on_existing_models(trial_path, test_on_f=True, test_on_g=False, trained_on_f=True, trained_on_g=False)

    trial_path = './f_and_g_no_subnets/5_Ling_50ms_ftrain'
    test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, trained_on_f=True, trained_on_g=False)
    test_on_existing_models(trial_path, test_on_f=False, test_on_g=True, trained_on_f=True, trained_on_g=False)
    test_on_existing_models(trial_path, test_on_f=True, test_on_g=False, trained_on_f=True, trained_on_g=False)

    trial_path = './f_and_g_no_subnets/6_Ling_50ms_gtrain'
    test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, trained_on_f=False, trained_on_g=True)
    test_on_existing_models(trial_path, test_on_f=False, test_on_g=True, trained_on_f=False, trained_on_g=True)
    test_on_existing_models(trial_path, test_on_f=True, test_on_g=False, trained_on_f=False, trained_on_g=True)

    trial_path = './f_and_g_two_subnets/1_Acous_10ms_Ling_50ms_ftrain'
    test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, trained_on_f=True, trained_on_g=False)
    test_on_existing_models(trial_path, test_on_f=False, test_on_g=True, trained_on_f=True, trained_on_g=False)
    test_on_existing_models(trial_path, test_on_f=True, test_on_g=False, trained_on_f=True, trained_on_g=False)

    trial_path = './f_and_g_two_subnets/2_Acous_10ms_Ling_50ms_gtrain'
    test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, trained_on_f=False, trained_on_g=True)
    test_on_existing_models(trial_path, test_on_f=False, test_on_g=True, trained_on_f=False, trained_on_g=True)
    test_on_existing_models(trial_path, test_on_f=True, test_on_g=False, trained_on_f=False, trained_on_g=True)

# #to get results with different onset prediction times: TODO: add report_dict_name to prevent saving over previous results
#     onset_prediction_length = [None, None]  # default is [0, 60] (60 frames is 3 seconds)
#     trial_path = './no_subnets/2_Acous_10ms'
#     test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, onset_prediction_frames=onset_prediction_length)
#     test_on_existing_models(trial_path, test_on_f=False, test_on_g=True,onset_prediction_frames=onset_prediction_length)
#     test_on_existing_models(trial_path, test_on_f=True, test_on_g=False, onset_prediction_frames=onset_prediction_length)
#
#     trial_path = './two_subnets/2_Acous_10ms_Ling_50ms'
#     test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, onset_prediction_frames=onset_prediction_length)
#     test_on_existing_models(trial_path, test_on_f=False, test_on_g=True, onset_prediction_frames=onset_prediction_length)
#     test_on_existing_models(trial_path, test_on_f=True, test_on_g=False, onset_prediction_frames=onset_prediction_length)
#
#     trial_path = './no_subnets/3_Ling_50ms'
#     test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, onset_prediction_frames=onset_prediction_length)
#     test_on_existing_models(trial_path, test_on_f=False, test_on_g=True, onset_prediction_frames=onset_prediction_length)
#     test_on_existing_models(trial_path, test_on_f=True, test_on_g=False, onset_prediction_frames=onset_prediction_length)
#
#     trial_path = './f_and_g_no_subnets/3_Acous_10ms_ftrain'
#     test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, trained_on_f=True, trained_on_g=False,
#                             onset_prediction_frames=onset_prediction_length)
#     test_on_existing_models(trial_path, test_on_f=False, test_on_g=True, trained_on_f=True, trained_on_g=False,
#                             onset_prediction_frames=onset_prediction_length)
#     test_on_existing_models(trial_path, test_on_f=True, test_on_g=False, trained_on_f=True, trained_on_g=False,
#                             onset_prediction_frames=onset_prediction_length)
#
#     trial_path = './f_and_g_no_subnets/5_Ling_50ms_ftrain'
#     test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, trained_on_f=True, trained_on_g=False,
#                             onset_prediction_frames=onset_prediction_length)
#     test_on_existing_models(trial_path, test_on_f=False, test_on_g=True, trained_on_f=True, trained_on_g=False,
#                             onset_prediction_frames=onset_prediction_length)
#     test_on_existing_models(trial_path, test_on_f=True, test_on_g=False, trained_on_f=True, trained_on_g=False,
#                             onset_prediction_frames=onset_prediction_length)
#
#     trial_path = './f_and_g_no_subnets/4_Acous_10ms_gtrain'
#     test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, trained_on_f=False, trained_on_g=True,
#                             onset_prediction_frames=onset_prediction_length)
#     test_on_existing_models(trial_path, test_on_f=False, test_on_g=True, trained_on_f=False, trained_on_g=True,
#                             onset_prediction_frames=onset_prediction_length)
#     test_on_existing_models(trial_path, test_on_f=True, test_on_g=False, trained_on_f=False, trained_on_g=True,
#                             onset_prediction_frames=onset_prediction_length)
#
#     trial_path = './f_and_g_no_subnets/6_Ling_50ms_gtrain'
#     test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, trained_on_f=False, trained_on_g=True,
#                             onset_prediction_frames=onset_prediction_length)
#     test_on_existing_models(trial_path, test_on_f=False, test_on_g=True, trained_on_f=False, trained_on_g=True,
#                             onset_prediction_frames=onset_prediction_length)
#     test_on_existing_models(trial_path, test_on_f=True, test_on_g=False, trained_on_f=False, trained_on_g=True,
#                             onset_prediction_frames=onset_prediction_length)
#
#     trial_path = './f_and_g_two_subnets/1_Acous_10ms_Ling_50ms_ftrain'
#     test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, trained_on_f=True, trained_on_g=False,
#                             onset_prediction_frames=onset_prediction_length)
#     test_on_existing_models(trial_path, test_on_f=False, test_on_g=True, trained_on_f=True, trained_on_g=False,
#                             onset_prediction_frames=onset_prediction_length)
#     test_on_existing_models(trial_path, test_on_f=True, test_on_g=False, trained_on_f=True, trained_on_g=False,
#                             onset_prediction_frames=onset_prediction_length)
#
#     trial_path = './f_and_g_two_subnets/2_Acous_10ms_Ling_50ms_gtrain'
#     test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, trained_on_f=False, trained_on_g=True,
#                             onset_prediction_frames=onset_prediction_length)
#     test_on_existing_models(trial_path, test_on_f=False, test_on_g=True, trained_on_f=False, trained_on_g=True,
#                             onset_prediction_frames=onset_prediction_length)
#     test_on_existing_models(trial_path, test_on_f=True, test_on_g=False, trained_on_f=False, trained_on_g=True,
#                             onset_prediction_frames=onset_prediction_length)

# to test models trained on both sets, on different datasets
#     trial_path = './no_subnets/2_Acous_10ms'
#     test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, report_dict_name='test_on_maptask',
#                             test_list_path="./data/splits/testing_maptask.txt")
#     test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, report_dict_name='test_on_switchboard',
#                             test_list_path="./data/splits/testing_switchboard.txt")
#     test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, report_dict_name='test_on_both_datasets',
#                             test_list_path="./data/splits/testing_both.txt")
#
#     trial_path = './two_subnets/2_Acous_10ms_Ling_50ms'
#     test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, report_dict_name='test_on_maptask',
#                             test_list_path="./data/splits/testing_maptask.txt")
#     test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, report_dict_name='test_on_switchboard',
#                             test_list_path="./data/splits/testing_switchboard.txt")
#     test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, report_dict_name='test_on_both_datasets',
#                             test_list_path="./data/splits/testing_both.txt")
#
#     trial_path = './no_subnets/3_Ling_50ms'
#     test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, report_dict_name='test_on_maptask',
#                             test_list_path="./data/splits/testing_maptask.txt")
#     test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, report_dict_name='test_on_switchboard',
#                             test_list_path="./data/splits/testing_switchboard.txt")
#     test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, report_dict_name='test_on_both_datasets',
#                             test_list_path="./data/splits/testing_both.txt")

# to test models trained on Maptask, on different datasets
#     trial_path = '/group/project/cstr1/mscslp/2019-20/s0910315_Sarah_Burne_James/dev/no_subnets/2_Acous_10ms'
#     test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, report_dict_name='test_on_maptask',
#                             test_list_path="./data/splits/testing_maptask.txt",
#                             train_list_path="./data/splits/training_maptask.txt")
#     test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, report_dict_name='test_on_switchboard',
#                             test_list_path="./data/splits/testing_switchboard.txt",
#                             train_list_path="./data/splits/training_maptask.txt")
#     test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, report_dict_name='test_on_both_datasets',
#                             test_list_path="./data/splits/testing_both.txt",
#                             train_list_path="./data/splits/training_maptask.txt")

    # trial_path = '/group/project/cstr1/mscslp/2019-20/s0910315_Sarah_Burne_James/dev/no_subnets/3_Ling_50ms'
    # test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, report_dict_name='test_on_maptask',
    #                         test_list_path="./data/splits/testing_maptask.txt",
    #                         train_list_path="./data/splits/training_maptask.txt")
    # test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, report_dict_name='test_on_switchboard',
    #                         test_list_path="./data/splits/testing_switchboard.txt",
    #                         train_list_path="./data/splits/training_maptask.txt")
    # test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, report_dict_name='test_on_both_datasets',
    #                         test_list_path="./data/splits/testing_both.txt",
    #                         train_list_path="./data/splits/training_maptask.txt")
    #
    # trial_path = '/group/project/cstr1/mscslp/2019-20/s0910315_Sarah_Burne_James/dev/two_subnets/2_Acous_10ms_Ling_50ms'
    # test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, report_dict_name='test_on_maptask',
    #                         test_list_path="./data/splits/testing_maptask.txt",
    #                         train_list_path="./data/splits/training_maptask.txt")
    # test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, report_dict_name='test_on_switchboard',
    #                         test_list_path="./data/splits/testing_switchboard.txt",
    #                         train_list_path="./data/splits/training_maptask.txt")
    # test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, report_dict_name='test_on_both_datasets',
    #                         test_list_path="./data/splits/testing_both.txt",
    #                         train_list_path="./data/splits/training_maptask.txt")


# to test models trained on Switchboard, on different datasets
#     trial_path = '/group/project/cstr1/mscslp/2019-20/s0910315_Sarah_Burne_James/switchboard_dev/no_subnets/2_Acous_10ms'
#     test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, report_dict_name='test_on_maptask',
#                             test_list_path="./data/splits/testing_maptask.txt",
#                             train_list_path="./data/splits/training_switchboard.txt")
#     test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, report_dict_name='test_on_switchboard',
#                             test_list_path="./data/splits/testing_switchboard.txt",
#                             train_list_path="./data/splits/training_switchboard.txt")
#     test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, report_dict_name='test_on_both_datasets',
#                             test_list_path="./data/splits/testing_both.txt",
#                             train_list_path="./data/splits/training_switchboard.txt")

    # trial_path = '/group/project/cstr1/mscslp/2019-20/s0910315_Sarah_Burne_James/switchboard_dev/two_subnets/2_Acous_10ms_Ling_50ms'
    # test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, report_dict_name='test_on_maptask',
    #                         test_list_path="./data/splits/testing_maptask.txt",
    #                         train_list_path="./data/splits/training_switchboard.txt")
    # test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, report_dict_name='test_on_switchboard',
    #                         test_list_path="./data/splits/testing_switchboard.txt",
    #                         train_list_path="./data/splits/training_switchboard.txt")
    # test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, report_dict_name='test_on_both_datasets',
    #                         test_list_path="./data/splits/testing_both.txt",
    #                         train_list_path="./data/splits/training_switchboard.txt")
    #
    # trial_path = '/group/project/cstr1/mscslp/2019-20/s0910315_Sarah_Burne_James/switchboard_dev/no_subnets/3_Ling_50ms'
    # test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, report_dict_name='test_on_maptask',
    #                         test_list_path="./data/splits/testing_maptask.txt",
    #                         train_list_path="./data/splits/training_switchboard.txt")
    # test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, report_dict_name='test_on_switchboard',
    #                         test_list_path="./data/splits/testing_switchboard.txt",
    #                         train_list_path="./data/splits/training_switchboard.txt")
    # test_on_existing_models(trial_path, test_on_f=True, test_on_g=True, report_dict_name='test_on_both_datasets',
    #                         test_list_path="./data/splits/testing_both.txt",
    #                         train_list_path="./data/splits/training_switchboard.txt")
