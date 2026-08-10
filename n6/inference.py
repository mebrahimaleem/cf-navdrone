import ml
import time
from ulab import numpy as np
import gc

h0 = np.zeros((1, 10, 10, 128), dtype=np.int8)
s0 = np.zeros((1, 10, 10, 128), dtype=np.int8)
h1 = np.zeros((1, 10, 10, 128), dtype=np.int8)
s1 = np.zeros((1, 10, 10, 128), dtype=np.int8)
h2 = np.zeros((1, 10, 10, 128), dtype=np.int8)
s2 = np.zeros((1, 10, 10, 128), dtype=np.int8)
h3 = np.zeros((1, 10, 10, 128), dtype=np.int8)
s3 = np.zeros((1, 10, 10, 128), dtype=np.int8)

class PostProcess:
	def __init__(self):
		pass

	def __call__(self, model, inputs, outputs):
		global h0, s0, h1, s1, h2, s2, h3, s3

		vel, depth, h0, s0, h1, s1, h2, s2, h3, s3 = outputs
		return vel, depth

bem = np.zeros((1, 320, 320, 1), dtype=np.int8)

model = ml.Model("/rom/model.onnx", postprocess=PostProcess())
print(model)

clock = time.clock()

gc.collect()
gc.disable()

i = 0

while True:
	clock.tick()

	_, _ = model.predict([bem, h0, s0, h1, s1, h2, s2, h3, s3])

	i += 1
	if i == 5:
		gc.collect()
		i = 0

	print(clock.fps())
