#!/usr/bin/env python3
import sys
import os
import glob
import argparse as ag
import numpy as np
import pandas as pd

argv = ag.ArgumentParser()
argv.add_argument('input_file', type=str)
argv.add_argument('output_file', type=str, default='*/')

scoring = {
	'Center': [0,0],
	'Left': [0,1],
	'Right': [1,0],
}

class accuracy:

	def __init__(self, data):
		self.data = data
		self.point = ['Center', 'Left', 'Right']
		self.dir = ['In', 'Out']
		self.totals = {}
		self.scores = {}

		for p in self.point:
			for d in self.dir:
				e = p + d
				self.totals[e] = 0

		self.find_summary()

	def find_summary(self):
		idx = self.data.index[self.data['0'] == 'SUMMARY DATA'][0]
		summary = self.data.iloc[idx:,:9].reset_index(drop=True)

		idx = summary.index[summary['0'] == 'Condition'][0]
		self.data = summary.iloc[idx:].reset_index(drop=True)

		self.data.columns = self.data.iloc[0]
		self.data = self.data.iloc[1:].reset_index(drop=True)

	def sum_stimulus(self, idx):
		event = self.data['Stimulus'][idx]
		presses = {}
		type = None
		direction = None

		for p in self.point:
			if p in event:
				type = p
				break
		for d in self.dir:
			if d in event:
				direction = d

		if type is None or direction is None:
			# untested
			raise ValueError(f'Unrecognized event: {event}')

		event = type + direction

		n_events = self.data['n Events'][idx]
		n_c_event = 0

		i = 1

		while n_c_event != n_events:
			n_c_event = self.data['n Events'][idx+i]
			presses[i] = n_c_event
			i = i + 1

		split = self.score_presses(presses, type, n_events)

		if event in self.scores:
			c_correct, c_incorrect = self.scores[event]
			self.scores[event] = (
				c_correct + split[0],
				c_incorrect + split[1]
			)
		else:
			self.scores[event] = split

		self.totals[event] = self.totals[event] + n_events

	def score_presses(self, presses, type, total):
	    correct = 0
	    incorrect = 0
	    n_presses = 0

    # Iterate through presses dictionary
	    for idx, press_count in presses.items():
	        # Use scoring weights for the current type
	        if idx == 3:
	        	continue
	        weight = scoring[type]
	        if idx - 1 < len(weight):  # Prevent out-of-bounds error
	            if weight[idx - 1]:  # If scoring weight is 1 (correct press)
	                correct += press_count
	            else:  # If scoring weight is 0 (incorrect press)
	                incorrect += press_count

	        n_presses += press_count
	    if type in ['Left', 'Right']:
        	missed_presses = total - n_presses
        	incorrect += missed_presses

	    # Adjust correct count for 'Center' type
	    if type == 'Center':
	        correct = total - incorrect

	    return (correct, incorrect)

	def accuracy(self):
		for i in range(len(self.data['Stimulus'])):
			if type(self.data['Stimulus'][i]) == str:
				self.sum_stimulus(i)

def process_directory(input_dir, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each Excel file in the input directory
    for file_path in glob.glob(os.path.join(input_dir, '*.xlsx')):
        print(f"Processing {file_path}...")
        data = pd.read_excel(file_path)
        cols = np.array(range(data.shape[1]), dtype=str)
        data.columns = cols

        # Compute accuracy
        acc = accuracy(data)
        acc.accuracy()

        # Prepare output file name
        file_name = os.path.basename(file_path).replace('.xlsx', '_accuracy.csv')
        output_path = os.path.join(output_dir, file_name)

        # Save scores to CSV
        scores_df = pd.DataFrame.from_dict(acc.scores, orient='index', columns=['Correct', 'Incorrect'])
        scores_df.index.name = 'Event'
        scores_df.to_csv(output_path)
        print(f"Saved accuracy scores to {output_path}")

def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir

    # Process the directory
    process_directory(input_dir, output_dir)

if __name__ == '__main__':
    argv = ag.ArgumentParser()
    argv.add_argument('input_dir', type=str, help='Directory containing input Excel files')
    argv.add_argument('output_dir', type=str, help='Directory to save output CSV files')
    args = argv.parse_args()

    main(args)
