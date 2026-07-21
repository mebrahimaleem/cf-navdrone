def main():
	import os
	import csi
	from ulab import numpy as np

	BIN_SIZE = 1000000
	BUF_SIZE = 65536
	BIN_THRESH = BIN_SIZE - BUF_SIZE
	events_bin = np.empty((BIN_SIZE, 6), dtype=np.uint16)
	events_index = 0

	# Initialize the sensor.
	csi0 = csi.CSI(cid=csi.GENX320)
	csi0.reset()
	csi0.ioctl(csi.IOCTL_GENX320_SET_MODE, csi.GENX320_MODE_EVENT, BUF_SIZE)
	csi0.ioctl(csi.IOCTL_GENX320_SET_BIASES, csi.GENX320_BIASES_DEFAULT)

	meta = "meta.txt"
	dir_index = 0
	file_index = 0

	try:
		with open(meta, "r") as f:
			dir_index = int(f.read().strip())
	except:
		pass

	with open(meta, "w") as f:
		f.write(str(dir_index + 1))

	directory = f"events_{dir_index}"
	os.mkdir(directory)
	print(f"Saving results to {directory}")

	while True:
		events = events_bin[events_index:events_index + BUF_SIZE]
		event_count = csi0.ioctl(csi.IOCTL_GENX320_READ_EVENTS, events)

		if event_count > 0:
			events_index += event_count

		if events_index >= BIN_THRESH:
			file_path = f"{directory}/events_{file_index:010d}.bin"
			try:
				with open(file_path, "wb") as f:
					f.write(events_bin[:events_index])
					print(f"Saved {events_index} events to {file_path}")
			except:
				print(f"Failed to save to {file_path}")
				try:
					os.remove(file_path)
				except OSError:
					pass

			file_index += 1
			events_index = 0

try:
	main()
except KeyboardInterrupt:
	pass
