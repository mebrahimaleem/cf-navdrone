import asyncio

ENABLE_LOG = False
ENABLE_RADIO = False


class LEDManager:
	from pyb import LED

	def __init__(self):
		self.red_led = LEDManager.LED(1)
		self.green_led = LEDManager.LED(2)
		self.blue_led = LEDManager.LED(3)
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
	from pyb import UART
	import struct

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

	def __init__(self):
		self._con = CFConnection.UART(4, 115200)
		self.connected = False
		self.connected_event = asyncio.Event()
		self.fps = 0

	async def reset(self):
		self.connected_event.clear()
		readback = False
		while not readback:
			while self._con.any() < 1:
				self._con.write(CFConnection.struct.pack("<B", CFConnection._RESET))
				await asyncio.sleep_ms(50)
			while self._con.any() > 0:
				if CFConnection.struct.unpack("<B", self._con.read(1))[0] == CFConnection._RESET:
					readback = True
					break
		readback = False
		while not readback:
			while self._con.any() > 0:
				if CFConnection.struct.unpack("<B", self._con.read(1))[0] == CFConnection._MAGIC:
					readback = True
					break
			await asyncio.sleep_ms(10)
		self.connected = True
		led_manager.led_connected()
		self.connected_event.set()

	def write_flight_command(self, cmd):
		if not self.connected:
			return False
		self._con.write(CFConnection.struct.pack("<BBB", CFConnection._MAGIC, CFConnection._FLIGHT_CMD, cmd))
		return True

	def track_fps(self, fps):
		self.fps = max(fps, self.fps)

	def log_fps(self):
		try:
			if not self.connected:
				return False
			if self.fps > 255:
				self.fps = 255
			elif self.fps < 0:
				self.fps = 0
			self._con.write(CFConnection.struct.pack("<BBB", CFConnection._MAGIC, CFConnection._LOG_FPS, int(self.fps)))
			self.fps = 0
			return True
		except:
			return False

	def log_error(self, state=0):
		if not self.connected:
			return False
		self._con.write(CFConnection.struct.pack("<BBB", CFConnection._MAGIC, CFConnection._LOG_ERROR, state))
		return True


cf_con = CFConnection()


async def radio_handler(characteristic):
	while True:
		conn, data = await characteristic.written()

async def radio():
	import aioble
	import bluetooth
	from micropython import const

	ble = bluetooth.BLE()
	ble.active(False)

	await asyncio.sleep_ms(200)

	ble.active(True)
	service_uuid = bluetooth.UUID(0x1815)
	service = aioble.Service(service_uuid)
	characteristic = aioble.Characteristic(service, bluetooth.UUID(0x2A56), write=True, capture=True)
	aioble.register_services(service)
	asyncio.create_task(radio_handler(characteristic))

	while True:
		async with await aioble.advertise(
			250_000,
			name="N6",
			services=[service_uuid],
			appearance=const(512),
		) as connection:
			await connection.disconnected()

async def pipeline():
	from ulab import numpy as np
	import csi
	import time
	import sys


	class PostProcess:
		def __init__(self):
			pass

		def __call__(self, model, inputs, outputs):
			return outputs


	try:
		events = np.zeros((2048, 6), dtype=np.uint16)
		pol = np.zeros((2048), dtype=np.int16)

		csi0 = csi.CSI(cid=csi.GENX320)
		csi0.reset()
		csi0.ioctl(csi.IOCTL_GENX320_SET_MODE, csi.GENX320_MODE_EVENT, events.shape[0])
		csi0.ioctl(csi.IOCTL_GENX320_SET_BIASES, csi.GENX320_BIASES_DEFAULT)

		bin_right = 0
		bin_left = 0
		bin_top = 0
		bin_bot = 0
		bin_center = 0

		cur_interval = 0
		BIN_INTERVAL = 200
		THRESH = 100

		cmd = CFConnection.FLIGHT_CMD_IDLE

		clock = time.clock()
		while True:
			clock.tick()

			await asyncio.sleep_ms(0)

			event_count = csi0.ioctl(csi.IOCTL_GENX320_READ_EVENTS, events)
			cur_interval += event_count

			new_events = events[:event_count]
			new_pol = pol[:event_count]
			new_pol[:] = 0	
			event_types = new_events[:, 0]
			y = new_events[:, 4]
			x = new_events[:, 5]

			new_pol[event_types == csi.PIX_OFF_EVENT] = 1
			new_pol[event_types == csi.PIX_ON_EVENT] = -1

			cond_left = x <= 100
			cond_right = x >= 220
			cond_top = ~cond_right & ~cond_left & (y <= 100)
			cond_bot = ~cond_right & ~cond_left & (y >= 220)
			cond_center = ~cond_right & ~cond_left & ~cond_top & ~cond_bot

			bin_right += abs(np.sum(new_pol[cond_right]))
			bin_left += abs(np.sum(new_pol[cond_left]))
			bin_top += abs(np.sum(new_pol[cond_top]))
			bin_bot += abs(np.sum(new_pol[cond_bot]))
			bin_center += abs(np.sum(new_pol[cond_center]))

			if cur_interval >= BIN_INTERVAL:
				block_left = bin_left >= THRESH
				block_right = bin_right >= THRESH
				block_center = bin_center >= THRESH
				block_top = bin_top >= THRESH
				block_bot = bin_bot >= THRESH

				bin_left = bin_right = bin_top = bin_bot = bin_center = cur_interval = 0

				cmd = CFConnection.FLIGHT_CMD_IDLE
				if block_center:
					if not block_right:
						cmd = CFConnection.FLIGHT_CMD_FWD | CFConnection.FLIGHT_CMD_RIGHT | CFConnection.FLIGHT_CMD_YRIGHT
					elif not block_left:
						cmd = CFConnection.FLIGHT_CMD_FWD | CFConnection.FLIGHT_CMD_LEFT | CFConnection.FLIGHT_CMD_YLEFT
					else:
						cmd = CFConnection.FLIGHT_CMD_BCK | CFConnection.FLIGHT_CMD_YLEFT
				elif block_right and not block_left:
					cmd = CFConnection.FLIGHT_CMD_FWD | CFConnection.FLIGHT_CMD_LEFT | CFConnection.FLIGHT_CMD_YLEFT
				elif block_left and not block_right:
					cmd = CFConnection.FLIGHT_CMD_FWD | CFConnection.FLIGHT_CMD_RIGHT | CFConnection.FLIGHT_CMD_YRIGHT
				elif block_left and block_right:
					cmd = CFConnection.FLIGHT_CMD_FWD
				elif block_top or block_bot:
					cmd = CFConnection.FLIGHT_CMD_FWD
				else:
					cmd = CFConnection.FLIGHT_CMD_FWD

			cf_con.write_flight_command(cmd)
			cf_con.track_fps(clock.fps())

	except Exception as e:
		print("Pipeline Crashed")
		sys.print_exception(e)
		cf_con.log_error(1)
		await asyncio.sleep_ms(1000)
		asyncio.create_task(pipeline())

async def log():
	while True:
		if not cf_con.connected:
			await cf_con.connected_event.wait()
		await asyncio.sleep_ms(1000)

async def cleanup():
	import gc
	gc.disable()
	while True:
		gc.collect()
		await asyncio.sleep_ms(1000)

async def heartbeat():
	while True:
		if not cf_con.connected:
			led_manager.led_disconnected()
			await cf_con.reset()

		led_manager.led_heartbeat()
		cf_con.log_fps()
		await asyncio.sleep_ms(500)

async def main():
	asyncio.create_task(cleanup())
	asyncio.create_task(pipeline())

	if ENABLE_LOG:
		asyncio.create_task(log())
	if ENABLE_RADIO:
		asyncio.create_task(radio())

	await heartbeat()

asyncio.run(main())

