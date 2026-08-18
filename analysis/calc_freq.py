#!/bin/env python

import numpy as np
import pandas as pd
import sys

def main():
	if len(sys.argv) != 2:
		print(f"usage: {sys.argv[0]} csv_file")
		return

	df = pd.read_csv(sys.argv[1])

	col_cur = "Current(uA)"
	col_time = "Timestamp(ms)"

	act_thresh = 270000
	dct_thresh = 200000

	mask = pd.Series(False, index=df.index)

	currents = df[col_cur].values

	act = False
	pmi = None
	pmc = 0

	for i, c in enumerate(df[col_cur].values):
		if act and c < dct_thresh:
			mask[pmi] = True
			act = False

		elif not act and c > act_thresh:
			act = True
			pmi = None
			pmc = 0

		if act and c > pmc:
			pmi = i
			pmc = c

	peaks = df[mask]

	periods_ms = peaks[col_time].diff()

	avg_period = periods_ms.mean()

	avg_freq = 1000 / avg_period

	print(f"{avg_freq} Hz ({avg_period} ms), calculated using {len(df)} samples ({len(peaks)} peaks) over {df[col_time].iloc[-1]}ms")

if __name__ == "__main__":
	main()
