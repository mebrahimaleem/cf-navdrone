#!/bin/env python

import sys
import onnx
from onnxsim import simplify


def usage():
	print(f"usage: {sys.argv[0]} input output")

def main():
	if len(sys.argv) != 3:
		usage()
		return -1

	model = onnx.load(sys.argv[1])
	
	simplified_model, check = simplify(model)

	if not check:
		raise RuntimeError("Failed to simplify model")

	onnx.save(simplified_model, sys.argv[2])

	print(f"Model ({sys.argv[1]}) simplified to {sys.argv[2]}")

	return 0

if __name__ == "__main__":
	sys.exit(main())


