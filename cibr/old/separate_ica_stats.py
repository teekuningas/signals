import sys
import os
import csv

import argparse

import mne
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from scipy import stats

# set font size
import matplotlib
matplotlib.rc('font', size=6)

def get_component_data(selections, data, columns):
    component_data = []

    for key, value in selections:
        if key in data:
            for column in columns:
                component_data.append(data[key][column][value-1])

    component_data = np.reshape(np.array(component_data), 
        (len(component_data)/len(columns), len(columns)))
    return pd.DataFrame(component_data)
            

def draw_component_info(frame, show=True):
    # extract state data
    mind_column = frame[0]
    rest_column = frame[1]
    plan_column = frame[2]
    anx_column = frame[3]

    # create four subplot columns
    fig_, axarr = plt.subplots(1, 4)

    # Mind, plan and anx and for the first subplot
    axarr[0].set_title("Mind, plan, anx")

    columns = [mind_column, plan_column, anx_column]
    axarr[0].boxplot(columns, 0, '', 1, showmeans=True)

    # this reduces the number of ticks
    axarr[0].locator_params(nbins=4, axis='x')

    def plot_comparison(ax, column1, column2, column1_name, column2_name):
        """ helper function to make a subplot for comparing two states """
      
        # get p value of the paired t test
        pvalue = stats.ttest_1samp(column1 - column2, 0).pvalue

        ax.boxplot(column1 - column2, 0, '', 1, showmeans=True)
        ax.set_title(column1_name + " vs " + column2_name)

        # place a pvalue text box in upper left in axes coords
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, "p = " + '%.5f' % pvalue, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)

        # add horizontal zero-line
        ax.axhline(0, linestyle='dashed')

    # do three subplots for comparing state means
    plot_comparison(axarr[1], mind_column, plan_column, "mind", "plan")
    plot_comparison(axarr[2], mind_column, anx_column, "mind", "anx")
    plot_comparison(axarr[3], plan_column, anx_column, "plan", "anx")

    if show:
        plt.show()

    return fig_


def read_component_selections(input_path):
    with open(input_path + 'component_selections.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        component_selections = [line for line in reader]
    
    returnable = {}

    headers = component_selections[0][1:]
    for header in headers:
        returnable[header] = []

    for row in component_selections[1:]:
        for j in range(len(headers)):
            subject_id = row[0]
            val = row[j+1]
            if val:
                returnable[headers[j]].append((subject_id, int(val)))

    return returnable


def load_data(input_path):

    data = {}

    for subject_folder in os.listdir(input_path):
        if not os.path.isdir(os.path.join(input_path, subject_folder)):
            continue

        # folder names of format "003", "004", etc.
        if len(subject_folder) != 3:
            continue

        csv_fnames = [fname for fname in 
                      os.listdir(os.path.join(input_path, subject_folder)) 
                      if fname.endswith('.csv')]

        if not subject_folder in data:
            data[subject_folder] = {}

        for fname in csv_fnames:

            peak_values = np.loadtxt(
                os.path.join(input_path, subject_folder, fname))
            data[subject_folder][fname.split('.csv')[0]] = peak_values

    return data


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('input_path')
    parser.add_argument('--save_path')
    cli_args = parser.parse_args()

    save_path = cli_args.save_path
    input_path = cli_args.input_path

    data = load_data(input_path)

    selections = read_component_selections(input_path)

    left_temporal_low = get_component_data(selections['left_temporal_low'], data,
        columns=['mind_peak_values', 'rest_peak_values', 'plan_peak_values', 'anx_peak_values'])
    left_temporal_high = get_component_data(selections['left_temporal_high'], data,
        columns=['mind_peak_values', 'rest_peak_values', 'plan_peak_values', 'anx_peak_values'])

    draw_component_info(np.log10(left_temporal_high))
    draw_component_info(np.log10(left_temporal_low))

