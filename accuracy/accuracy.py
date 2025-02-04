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

        # Initialize totals for each event type
        for p in self.point:
            for d in self.dir:
                e = p + d
                self.totals[e] = 0

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
        type = None
        direction = None

        # Identify stimulus type and direction
        for p in self.point:
            if p in event:
                type = p
                break
        for d in self.dir:
            if d in event:
                direction = d

        if type is None or direction is None:
            # Raise error if event format is not recognized
            raise ValueError(f'Unrecognized event: {event}')

        event = type + direction  # Combine type and direction to create event key

        n_events = self.data['n Events'][idx]  # Total number of events
        n_c_event = 0  # Counter for current events processed

        i = 1  # Offset to locate additional rows for this stimulus
        while n_c_event != n_events:
            n_c_event = self.data['n Events'][idx + i]
            presses[i] = n_c_event  # Store count of presses for current row
            i = i + 1

        # Score the presses for the current stimulus
        split = self.score_presses(presses, type, n_events)

        # Update scores and totals for this event
        if event in self.scores:
            c_correct, c_incorrect = self.scores[event]
            self.scores[event] = (
                c_correct + split[0],
                c_incorrect + split[1]
            )
        else:
            self.scores[event] = split

        self.totals[event] += n_events

    def score_presses(self, presses, type, total):
        # Compute correct and incorrect counts based on scoring rules
        correct = 0
        incorrect = 0
        n_presses = 0

        # Iterate through presses dictionary
        for idx, press_count in presses.items():
            if idx == 3:  # Skip idx 3 if reached
                continue
            weight = scoring[type]  # Use scoring weights for the current type
            if idx - 1 < len(weight):  # Prevent out-of-bounds error
                if weight[idx - 1]:  # If scoring weight is 1 (correct press)
                    correct += press_count
                else:  # If scoring weight is 0 (incorrect press)
                    incorrect += press_count

            n_presses += press_count
        if type in ['Left', 'Right']:
            # Account for missed presses in directional stimuli
            missed_presses = total - n_presses
            incorrect += missed_presses

        if type == 'Center':
            # Adjust correct count for 'Center' type
            correct = total - incorrect

        percent = correct / (correct + incorrect)

        return [correct, incorrect]

    def accuracy(self):
        for i in range(len(self.data['Stimulus'])):
            if type(self.data['Stimulus'][i]) == str:
                self.sum_stimulus(i)

def process_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

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
        except Exception as e:
            print(f'Error proccessing {file_path}: {e}')

def main(args):
    input_dir = args.input_dir  # Input directory argument
    output_dir = args.output_dir  # Output directory argument

    # Process the directory
    process_directory(input_dir, output_dir)

if __name__ == '__main__':
    # Parse arguments and execute the script
    args = argv.parse_args()
    main(args)
