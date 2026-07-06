#!/bin/env python

import sys
from onnx import version_converter
import onnxscript
from onnxscript.optimizer import optimize
import onnx.inliner


def usage():
	print(f"usage: {sys.argv[0]} net file")
	print(f"known networks:", ", ".join(net_table.keys()))

def main():
	if len(sys.argv) != 3:
		usage()
		return -1

	model = onnx.load(sys.argv[1])
	
	decomposed_model = onnx.inliner.inline_selected_functions(
		model,
		function_ids=[],
		exclude=True,
		inline_schema_functions=True
	)

	onnx.save(decomposed_model, sys.argv[2])

	print(f"Model ({sys.argv[1]}) inlined to {sys.argv[2]}")

	return 0

if __name__ == "__main__":
	sys.exit(main())

