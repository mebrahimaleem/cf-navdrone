import asyncio
import csi
import gc
import ml
from pyb import LED, UART
import struct
import sys
import time
from ulab import numpy as np

FORCE_HEARTBEAT = True


class LEDManager:
	def __init__(self):
		self.red_led = LED(1)
		self.green_led = LED(2)
		self.blue_led = LED(3)
		self.red_led.off()
		self.green_led.off()
		self.blue_led.off()
		self.parity = False

	def led_disconnected(self):
		self.red_led.on()

	def led_connected(self):
		self.red_led.off()

	def led_heartbeat(self):
		if self.parity:
			self.green_led.off()
		else:
			self.green_led.on()
		self.parity = not self.parity


led_manager = LEDManager()


class CFConnection:
	_RESET = 0xB8
	_MAGIC = 0x3E
	_FLIGHT_CMD = 0x01
	_LOG_FPS = 0x02
	_LOG_ERROR = 0x03
	FLIGHT_CMD_IDLE = 0
	FLIGHT_CMD_FWD = 1
	FLIGHT_CMD_BCK = 2
	FLIGHT_CMD_RIGHT = 4
	FLIGHT_CMD_LEFT = 8
	FLIGHT_CMD_UP = 16
	FLIGHT_CMD_DOWN = 32
	FLIGHT_CMD_YRIGHT = 64
	FLIGHT_CMD_YLEFT = 128

	_FPS_MAX = 255

	def __init__(self):
		self._con = UART(4, 115200)
		self.connected = False
		self.connected_event = asyncio.Event()
		self.fps = CFConnection._FPS_MAX

	async def reset(self):
		self.connected_event.clear()
		readback = False
		while not readback:
			while self._con.any() < 1:
				self._con.write(struct.pack("<B", CFConnection._RESET))
				await asyncio.sleep_ms(50)
			while self._con.any() > 0:
				if struct.unpack("<B", self._con.read(1))[0] == CFConnection._RESET:
					readback = True
					break
		readback = False
		while not readback:
			while self._con.any() > 0:
				if struct.unpack("<B", self._con.read(1))[0] == CFConnection._MAGIC:
					readback = True
					break
			await asyncio.sleep_ms(10)
		self.connected = True
		led_manager.led_connected()
		self.connected_event.set()

	def write_flight_command(self, cmd):
		if not self.connected:
			return False
		self._con.write(struct.pack("<BBB", CFConnection._MAGIC, CFConnection._FLIGHT_CMD, cmd))
		return True

	def track_fps(self, fps):
		self.fps = min(fps, self.fps)

	def log_fps(self):
		try:
			if self.fps > CFConnection._FPS_MAX:
				self.fps = CFConnection._FPS_MAX

			fps = self.fps
			self.fps = CFConnection._FPS_MAX

			if not self.connected:
				return False
			self._con.write(struct.pack("<BBB", CFConnection._MAGIC, CFConnection._LOG_FPS, int(fps)))
			return True
		except:
			return False

	def log_error(self, state=0):
		if not self.connected:
			return False
		self._con.write(struct.pack("<BBB", CFConnection._MAGIC, CFConnection._LOG_ERROR, state))
		return True


cf_con = CFConnection()

async def pipeline():
	h0 = np.zeros((1, 10, 10, 128), dtype=np.int8)
	s0 = np.zeros((1, 10, 10, 128), dtype=np.int8)
	h1 = np.zeros((1, 10, 10, 128), dtype=np.int8)
	s1 = np.zeros((1, 10, 10, 128), dtype=np.int8)
	h2 = np.zeros((1, 10, 10, 128), dtype=np.int8)
	s2 = np.zeros((1, 10, 10, 128), dtype=np.int8)
	h3 = np.zeros((1, 10, 10, 128), dtype=np.int8)
	s3 = np.zeros((1, 10, 10, 128), dtype=np.int8)

	bem = np.zeros((1, 320, 320, 1), dtype=np.int8)

	class PostProcess:
		def __init__(self):
			pass

		def __call__(self, model, inputs, outputs):
			global h0, s0, h1, s1, h2, s2, h3, s3

			# outputs tensors are stored on npuRAM. Copy by reference prevents expensive DMA reads
			vel, h0, s0, h1, s1, h2, s2, h3, s3 = outputs
			#vel, depth, h0, s0, h1, s1, h2, s2, h3, s3 = outputs

			return vel
			#return vel, depth


	try:
		model = ml.Model("/rom/model.onnx", postprocess=PostProcess())
		events = np.zeros((2048, 6), dtype=np.uint16)

		csi0 = csi.CSI(cid=csi.GENX320)
		csi0.reset()
		csi0.ioctl(csi.IOCTL_GENX320_SET_MODE, csi.GENX320_MODE_EVENT, events.shape[0])
		csi0.ioctl(csi.IOCTL_GENX320_SET_BIASES, csi.GENX320_BIASES_DEFAULT)

		cmd = CFConnection.FLIGHT_CMD_IDLE

		clock = time.clock()

		gc.collect()
		gc.disable()

		GC_COUNTER_THRESH = 5

		gc_counter = 0

		clock.tick()
		while True:
			await asyncio.sleep_ms(0)

			event_count = csi0.ioctl(csi.IOCTL_GENX320_READ_EVENTS, events)

			@micropython.viper
			def create_bem(bem_ptr: ptr8, events_ptr: ptr16, num_events: int, width: int, pix_on: int, pix_off: int):
				for i in range(width * width):
					bem_ptr[i] = 0

				for i in range(num_events):
					offset = i * 6
					t = events_ptr[offset]
					x = events_ptr[offset + 5] # camera is mounted sideways
					y = events_ptr[offset + 4]

					bem_idx = y * width + x

					if t == pix_on:
						bem_ptr[bem_idx] = 1
					elif t == pix_off:
						bem_ptr[bem_idx] = 255 # -1 uint8
			
			create_bem(bem, events, event_count, 320, csi.PIX_ON_EVENT, csi.PIX_OFF_EVENT)

			vel = model.predict([bem, h0, s0, h1, s1, h2, s2, h3, s3])
			#vel, depth = model.predict([bem, h0, s0, h1, s1, h2, s2, h3, s3])

			cf_con.write_flight_command(cmd)
			cf_con.track_fps(clock.fps())
			clock.tick()

			# Pipeline Throughputs (skip decoder, FastPixelShuffle)
			# With depth: ~26fps
			# Without depth: ~31fps
			# Throughtputs vary based on number of events. Can be improved by tweaking DMA buffer
			# Binning may need to span multiple frames to increase bem details

			gc_counter += 1
			if gc_counter == GC_COUNTER_THRESH:
				gc.collect()
				gc_counter = 0


	except Exception as e:
		print("Pipeline Crashed")
		sys.print_exception(e)
		cf_con.log_error(1)
		await asyncio.sleep_ms(1000)
		asyncio.create_task(pipeline())

async def heartbeat():
	while True:
		if not cf_con.connected and not FORCE_HEARTBEAT:
			led_manager.led_disconnected()
			await cf_con.reset()

		led_manager.led_heartbeat()
		cf_con.log_fps()
		await asyncio.sleep_ms(500)

async def main():
	asyncio.create_task(pipeline())

	await heartbeat()

asyncio.run(main())

