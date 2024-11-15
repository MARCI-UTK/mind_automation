#!/home/karim/anaconda3/envs/mind/bin/python
import sys
import argparse as ag
import numpy as np
import pandas as pd

argv = ag.ArgumentParser()
argv.add_argument('input_file', type=str)
argv.add_argument('output_file', type=str, default='*/')

class accuracy:

	def __init__(self, data):
		self.data = data
		self.point = ['Center', 'Left', 'Right']
		self.dir = ['In', 'Out']
		self.totals = {}

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

		for p in self.point:
			if p in event:
				type = p
				break
		for d in self.dir:
			if d in event:
				direction = d
		event = type + direction

		n_events = self.data['n Events'][idx]
		n_c_event = 0

		i = 1
		print(event)

		while n_c_event != n_events:
			n_c_event = self.data['n Events'][idx+i]
			presses[i] = n_c_event
			i = i + 1

		print(presses)
		if type == 'Center':
			pass

		self.totals[event] = self.totals[event] + n_events 

	def accuracy(self):
		for i in range(len(self.data['Stimulus'])):
			if type(self.data['Stimulus'][i]) == str:
				self.sum_stimulus(i)

def main(args):
	i_path = args.input_file
	data = pd.read_excel(i_path)
	cols = np.array(range(data.shape[1]), dtype=str)
	data.columns = cols

	acc = accuracy(data)

	acc.accuracy()

	print(acc.totals)

if __name__ == '__main__':
	args = argv.parse_args()
	main(args)