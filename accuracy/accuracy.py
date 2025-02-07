#!/usr/bin/env python3
import sys
import os
import glob
import argparse as ag
import numpy as np
import pandas as pd

# Set up command-line arguments
argv = ag.ArgumentParser()
argv.add_argument('input_dir', type=str)  # Input directory containing Excel files
argv.add_argument('output_dir', type=str, nargs='?', default='output')  # Output directory for CSV results
argv.add_argument('-m', '--master', type=str, nargs='?', const='id', default='')
argv.add_argument('-s', '--sum', action='store_true')

# Define scoring rules for stimulus types
scoring = {
    'Center': [0, 0],
    'Left': [0, 1],
    'Right': [1, 0],
}

# Class to compute accuracy metrics
class accuracy:
    def __init__(self, data):
        self.data = data  # Input data
        self.point = ['Center', 'Left', 'Right']  # Stimulus types
        self.dir = ['In', 'Out']  # Directions
        self.totals = {}  # Totals for each event
        self.scores = {}  # Scores (correct/incorrect) for each event

        self.find_summary()

    def find_summary(self):
        # Locate the summary data section in the input
        idx = self.data.index[self.data['0'] == 'SUMMARY DATA'][0]
        summary = self.data.iloc[idx:, :9].reset_index(drop=True)

        # Locate the 'Condition' row to extract stimulus-related data
        idx = summary.index[summary['0'] == 'Condition'][0]
        self.data = summary.iloc[idx:].reset_index(drop=True)

        # Set proper column headers
        self.data.columns = self.data.iloc[0]
        self.data = self.data.iloc[1:].reset_index(drop=True)

    def sum_stimulus(self, idx):
        # Process a specific stimulus event
        event = self.data['Stimulus'][idx]
        presses = {}  # Dictionary to store counts for each press
        p_type = None
        direction = None
        w_accel = False

        # Identify stimulus type and direction
        for p in self.point:
            if p in event:
                p_type = p
                break
        for d in self.dir:
            if d in event:
                direction = d

        if 'Accel' in event: # edge case just to be safe
            w_accel = True

        if p_type is None or direction is None:
            # Raise error if event format is not recognized
            raise ValueError(f'Unrecognized event: {event}')


        if w_accel:
            event = p_type + direction + 'Accel'
        else:
            event = p_type + direction  # Combine type and direction to create event key

        n_events = self.data['n Events'][idx]  # Total number of events
        n_c_event = 0  # Counter for current events processed

        for i in [1,2]:
            n_c_event = self.data['n Events'][idx + i]
            presses[i] = n_c_event

        # Score the presses for the current stimulus
        split = self.score_presses(presses, p_type, n_events)

        # Update scores and totals for this event
        if event in self.scores:
            c_correct, c_incorrect = self.scores[event]
            self.scores[event] = (
                c_correct + split[0],
                c_incorrect + split[1]
            )
        else:
            self.scores.setdefault(event, (0,0))
            self.scores[event] = split

        self.totals.setdefault(event, 0)
        self.totals[event] += n_events
            

    def score_presses(self, presses, p_type, total):
        # Compute correct and incorrect counts based on scoring rules
        correct = 0
        incorrect = 0
        n_presses = 0
        t_presses = sum(presses.values())

        # Iterate through presses dictionary
        for idx, press_count in presses.items():
            weight = scoring[p_type]  # Use scoring weights for the current type
            if idx - 1 < len(weight):  # Prevent out-of-bounds error
                if weight[idx - 1]:  # If scoring weight is 1 (correct press)
                    correct += press_count
                else:  # If scoring weight is 0 (incorrect press)
                    incorrect += press_count

            n_presses += press_count
        if p_type in ['Left', 'Right']:
            # Account for missed presses in directional stimuli
            missed_presses = abs(total - n_presses)
            if correct > total:
                missed_presses += correct - total
                correct = total
            incorrect += missed_presses

        if p_type == 'Center':
            # Adjust correct count for 'Center' type
            correct = total - incorrect
            if correct < 0:
                incorrect += abs(correct)
                correct = 0

        percent = correct / (correct + incorrect)

        return [correct, incorrect]

    def accuracy(self):
        for i in range(len(self.data['Stimulus'])):
            if type(self.data['Stimulus'][i]) == str:
                self.sum_stimulus(i)

def process_directory(input_dir, output_dir, master_file=''):
    os.makedirs(output_dir, exist_ok=True)

    master_list = []
    # Process each Excel file in the input directory
    for file_path in glob.glob(os.path.join(input_dir, '*.xlsx')):
        try:
            print(f"Processing {file_path}...")
            data = pd.read_excel(file_path)
            cols = np.array(range(data.shape[1]), dtype=str)  # Generate column names
            data.columns = cols
            
            # Compute accuracy
            acc = accuracy(data)
            acc.accuracy()

            # Prepare output file name
            file_name = os.path.basename(file_path).replace('.xlsx', '_accuracy.csv')
            output_path = os.path.join(output_dir, file_name)

            # Save scores to CSV
            scores_df = pd.DataFrame.from_dict(acc.scores, orient='index', columns=['Correct', 'Incorrect'])
            scores_df['Percent'] = (scores_df['Correct'] / (scores_df['Correct'] + scores_df['Incorrect'])).round(4)
            scores_df.index.name = 'Event'
            scores_df.to_csv(output_path)
            print(f"Saved accuracy scores to {output_path}")

            if master_file != '':
                scores_df['source'] = file_path
                master_list.append(scores_df)
        except Exception as e:
            print(f'Error proccessing {file_path}: {e}')
    if master_file != '':
        return pd.concat(master_list)
    return None
    
def main(args):
    input_dir = args.input_dir  # Input directory argument
    output_dir = args.output_dir  # Output directory argument
    master_file = args.master
    do_sum = args.sum

    if master_file != '':
        if master_file == 'id':
            master_file = input_dir
        if '.csv' not in master_file:
            master_file = master_file + '.csv'
        print(f'Processing directory {input_dir} with masterfile: {master_file}')
    # Process the directory
    master_df = process_directory(input_dir, output_dir, master_file)

    if master_df is not None:
        if do_sum: # sum up all data and recalculate percentages
            master_df = master_df.drop(columns=['source'])
            master_df = master_df.groupby(level=0).sum()
            master_df['Percent'] = (master_df['Correct'] / (master_df['Correct'] + master_df['Incorrect'])).round(4)
            master_df['input dir'] = input_dir
        master_df.to_csv(master_file)
        print(f'Saved masterfile to: {master_file}')

if __name__ == '__main__':
    # Parse arguments and execute the script
    args = argv.parse_args()
    main(args)
