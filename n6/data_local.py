#!/bin/env python

import sys
import os
import numpy as np

def main():
	if len(sys.argv) != 2:
		print(f"Usage: {sys.argv[0]} input")
		return


	files = sorted([f for f in os.listdir(sys.argv[1]) if f.endswith(".bin")])

	chunks = [np.fromfile(f"{sys.argv[1]}/{name}", dtype=np.uint16).reshape(-1, 6) for name in files]

	events = np.concatenate(chunks, axis=0)
	print(events)

if __name__ == "__main__":
	main()
