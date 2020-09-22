# to load in pickled results files and inspect them
import pickle
import numpy as np
from pprint import pprint

def get_f_g_stats(pickled_results):
    """Takes in pickled results file, prints stats about mean absolute error for f (instruction followers) and for g
    (instruction givers) and for all speakers."""
    with open(pickled_results, "rb") as results_file:
        results = pickle.load(results_file)
        pprint(results)

    all_errors = results['indiv_perf'][0]['bar_chart_vals']
    f_errors = []  # 0,2,4,6 etc.
    g_errors = []  # 1,3,5,7 etc.

    for index, error in enumerate(results['indiv_perf'][0]['bar_chart_vals']):
        if index % 2 == 0:
            f_errors.append(error)
        else:
            g_errors.append(error)

    mean_f_error = sum(f_errors)/len(f_errors)
    mean_g_error = sum(g_errors)/len(g_errors)
    mean_all_error = sum(all_errors)/len(all_errors)
    std_f_error = np.std(f_errors)
    std_g_error = np.std(g_errors)
    std_all_error = np.std(all_errors)

    print(f'average error on f: {mean_f_error:.3f} standard deviation: {std_f_error:.3f}')
    print(f'average error on g: {mean_g_error:.3f} standard deviation: {std_g_error:.3f}')
    print(f'average error on both: {mean_all_error:.3f} standard deviation: {std_all_error:.3f}')
    print(f'Difference between average error f and g: {abs(mean_f_error-mean_g_error):.3f}')

print("\n Train on both")
get_f_g_stats("results/f_g_test/results_train_on_both.p")
print("\n Train on f")
get_f_g_stats("results/f_g_test/results_train_on_f.p")
print("\n Train on g")
get_f_g_stats("results/f_g_test/results_train_on_g.p")