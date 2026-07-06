import asyncio

ENABLE_LOG = True
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

    def __init__(self):
        self._con = CFConnection.UART(3, 115200)
        self.connected = False
        self.connected_event = asyncio.Event()

    def reset(self):
        self.connected_event.clear()
        # keep sending reset until we get a readback
        readback = False
        while not readback:
            while self._con.any() < 1:
                self._con.write(CFConnection.struct.pack("<B", CFConnection._RESET))
                await asyncio.sleep_ms(50)

            while self._con.any() > 0:
                if CFConnection.struct.unpack("<B", self._con.read(1))[0] == CFConnection._RESET:
                    readback = True
                    break

        # now wait for magic, indicating ready
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
    import ml
    import csi
    import time

    class PostProcess:
        def __init__(self):
            pass

        def __call__(self, model, inputs, outputs):
            return outputs

    try:
        model = ml.Model("/rom/model.onnx", postprocess=PostProcess())
        print(model)

        events = np.zeros((2048, 6), dtype=np.uint16)

        csi0 = csi.CSI(cid=csi.GENX320)
        csi0.reset()
        csi0.ioctl(csi.IOCTL_GENX320_SET_MODE, csi.GENX320_MODE_EVENT, events.shape[0])

        dummy = np.zeros((1, 16))

        clock = time.clock()

        while True:
            clock.tick()
            event_count = csi0.ioctl(csi.IOCTL_GENX320_READ_EVENTS, events)

            res = model.predict([dummy])

            print(event_count, res, clock.fps())
    except Exception as e:
        print("Pipeline Crashed:", e)
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
            asyncio.create_task(cf_con.reset())
            await cf_con.connected_event.wait()

        led_manager.led_heartbeat()

        await asyncio.sleep_ms(500)


async def main():
    asyncio.create_task(cleanup())
    asyncio.create_task(pipeline())

    if ENABLE_LOG:
        asyncio.create_task(log())

    if ENABLE_RADIO:
        asyncio.create_task(radio())

    await heartbeat()

if __name__ == "__main__":
    asyncio.run(main())
