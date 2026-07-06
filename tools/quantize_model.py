#!/bin/env python

import sys
import onnxruntime
from onnxruntime.quantization import QuantFormat, QuantType, CalibrationDataReader, quantize_static
import torch

class DataReader(CalibrationDataReader):
	def __init__(self, path):
		self.data = iter(torch.load(path).numpy())

	def get_next(self):
		try:
			return {"inputs": next(self.data)}
		except StopIteration:
			return None

def usage():
	print(f"usage: {sys.argv[0]} intput output calib")

def main():
	if len(sys.argv) != 4:
		usage()
		return -1

	quantize_static(
		sys.argv[1],
		sys.argv[2],
		DataReader(sys.argv[3]),
		quant_format=QuantFormat.QDQ,
		per_channel=True,
		activation_type=QuantType.QInt8,
		weight_type=QuantType.QInt8,
	)

	print(f"Model ({sys.argv[1]}) quantized to {sys.argv[2]}")

	return 0

if __name__ == "__main__":
	sys.exit(main())
