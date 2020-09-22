import pickle
from pprint import pprint
from glob import glob
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_person_error(name_list, data, results_path, architecture, results_key='barchart',f_mean=None, g_mean=None,
                      colour_list=None):
    y_pos = np.arange(len(name_list))
    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.xlim(0, 3)
    plt.barh(y_pos, data, align='center', alpha=0.5, color=colour_list)
    plt.yticks(y_pos, name_list, fontsize=5)
    plt.xlabel('mean abs error per time frame', fontsize=7)
    plt.xticks(fontsize=7)
    plt.title(f'Individual Error - {architecture[2:]}')
    if f_mean:
        plt.axvline(x=f_mean, color='black', linestyle=":", label="f mean")
    if g_mean:
        plt.axvline(x=g_mean, color='black', linestyle="--", label = "g mean")
        plt.legend(["f_mean", "g_mean"])
    plt.savefig(results_path + '/' + results_key + '.pdf')


def plot_mean_person_error(test_directory, architecture):
    results_directory = glob(f'{test_directory}/*')
    number_of_models = len(results_directory)
    print(f"number of models is {number_of_models}")

    # get error for each speaker, in every model
    individual_results_dict = defaultdict(list)
    for model in results_directory:
        pickled_results = f"{model}/results.p"

        with open(pickled_results, "rb") as results_file:
            results = pickle.load(results_file)
            # pprint(results)
            individual_labels = results['indiv_perf'][0]['bar_chart_labels']
            individual_results = results['indiv_perf'][0]['bar_chart_vals']
            # pprint(individual_results)
            for speaker in range(len(individual_labels)):
                individual_results_dict[individual_labels[speaker]].append(individual_results[speaker])

    # pprint(individual_results_dict)

    # get mean error for each speaker
    mean_results_dict = {}

    for speaker, results in individual_results_dict.items():
        mean_results_dict[speaker] = np.mean(results)

    #get overall mean for f and g
    g_values = []
    f_values = []
    g_results_list = []
    f_results_list = []

    for key in mean_results_dict.keys():
        if key[-1] == 'f':
            g_values.append(mean_results_dict[key])
            g_results_list.append((key, mean_results_dict[key]))
        if key[-1] == 'g':
            f_values.append(mean_results_dict[key])
            f_results_list.append((key, mean_results_dict[key]))

    g_colourlist = ["blue"] * len(g_results_list)
    f_colourlist = ["green"] * len(f_results_list)
    colourlist = g_colourlist + f_colourlist

    g_mean = np.mean(g_values)
    f_mean = np.mean(f_values)

    mean_results_list = g_results_list + f_results_list
    # mean_results_list = sorted(mean_results_dict.items())
    bar_chart_labels, bar_chart_vals = map(list, zip(*mean_results_list))
    # bar_chart_labels.extend(["f mean", "g mean"])
    # bar_chart_vals.extend([f_mean, g_mean])

    plot_person_error(bar_chart_labels, bar_chart_vals, test_directory, architecture, results_key='mean_person_error',
                      f_mean=f_mean, g_mean=g_mean, colour_list=colourlist)


if __name__ == '__main__':
    #train on both no subnets models
    for architecture in ["2_Acous_10ms", "3_Ling_50ms"]:
        file_path = f"no_subnets/{architecture}/test_on_both"
        try:
            os.remove(f"{file_path}/mean_person_error.pdf")
        except FileNotFoundError:
            pass
        plot_mean_person_error(file_path, architecture)

    #train on both two subnets models
    for architecture in ["2_Acous_10ms_Ling_50ms"]:
        file_path = f"two_subnets/{architecture}/test_on_both"
        try:
            os.remove(f"{file_path}/mean_person_error.pdf")
        except FileNotFoundError:
            pass
        plot_mean_person_error(file_path, architecture)

    #f and g two subnets models
    for architecture in ["1_Acous_10ms_Ling_50ms_ftrain", "2_Acous_10ms_Ling_50ms_gtrain"]:
        file_path = f"f_and_g_two_subnets/{architecture}/test_on_both"
        try:
            os.remove(f"{file_path}/mean_person_error.pdf")
        except FileNotFoundError:
            pass
        plot_mean_person_error(file_path, architecture)

    # f and g no subnets models
    for architecture in ["3_Acous_10ms_ftrain", "4_Acous_10ms_gtrain", "5_Ling_50ms_ftrain", "6_Ling_50ms_gtrain"]:
        file_path = f"f_and_g_no_subnets/{architecture}/test_on_both"
        try:
            os.remove(f"{file_path}/mean_person_error.pdf")
        except FileNotFoundError:
            pass
        plot_mean_person_error(file_path, architecture)
