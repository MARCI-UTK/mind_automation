#!/home/karim/anaconda3/envs/mind/bin/python
import sys
import argparse as ag
import numpy as np
import pandas as pd

stimulus = {
	'CenterMO_IN': 0,
	'CenterMO_OUT': 0,
	'LeftMO_IN': 0,
	'LeftMO_OUT': 0,
	'RightMO_IN': 0,
	'RightMO_OUT': 0,
}

argv = ag.ArgumentParser()
argv.add_argument('input_file', type=str)
argv.add_argument('output_file', type=str, default='*/')

def main(args):
	i_path = args.input_file
	data = pd.read_excel(i_path)
	cols = np.array(range(data.shape[1]), dtype=str)
	data.columns = cols

	idx = data.index[data['0'] == 'SUMMARY DATA'][0]
	print(data.iloc[idx:].head())

if __name__ == '__main__':
	args = argv.parse_args()
	main(args)