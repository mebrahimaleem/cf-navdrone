#!/bin/env python

import sys
import torch
from net import net_table

def usage():
	print(f"usage: {sys.argv[0]} net file calib [weights]")
	print(f"known networks:", ", ".join(net_table.keys()))

def main():
	if len(sys.argv) != 4 and len(sys.argv) != 5:
		usage()
		return

	net_class = net_table.get(sys.argv[1], None)

	if net_class == None:
		print(f"Unkown net {sys.argv[1]}")
		usage()
		return

	torch_model = net_class()

	if len(sys.argv) == 5:
		chk = torch.load(sys.argv[4], weights_only=True, map_location=torch.device("cpu"))
		torch_model.load_state_dict(chk)

	torch_model.eval()

	with open(sys.argv[2], "wb") as f:
		onnx_model = torch.onnx.export(
			torch_model,
			torch_model.example_inputs(),
			f,
			dynamo=False,
			input_names=["inputs"],
			output_names=["outputs"],
			export_params=True,
			opset_version=13
		)

	print(f"Model ({sys.argv[1]}) saved to {sys.argv[2]}")

	torch.save(torch_model.gen_calib_data(), sys.argv[3])
	print(f"Model calibration data saved to {sys.argv[3]}")

if __name__ == "__main__":
	main()
