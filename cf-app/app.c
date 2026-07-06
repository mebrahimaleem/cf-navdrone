#include <stdint.h>
#include <stddef.h>

#include "app.h"
#include "led.h"
#include "log.h"
#include "debug.h"
#include "uart1.h"

#include "FreeRTOS.h"
#include "task.h"

#define CON_RESET			0xB8
#define CON_MAGIC			0x3E

#define CON_DISCONNECTED	0
#define CON_CONNECTED			1

/*
static logVarId_t id_x;
static logVarId_t id_y;
static logVarId_t id_z;

static logVarId_t id_roll;
static logVarId_t id_pitch;
static logVarId_t id_yaw;

static void com_log_pose(void) {
	struct {
		struct com_generic_t header;
		float x;
		float y;
		float z;
		float roll;
		float pitch;
		float yaw;
	} __attribute__((packed)) buffer;

	buffer.header.magic = COM_MAGIC;
	buffer.header.cmd = COM_LOG_POSE;

	buffer.x = logGetFloat(id_x);
	buffer.y = logGetFloat(id_y);
	buffer.z = logGetFloat(id_z);
	buffer.roll = logGetFloat(id_roll);
	buffer.pitch = logGetFloat(id_pitch);
	buffer.yaw = logGetFloat(id_yaw);

	uart1SendData(sizeof(buffer), (void*)&buffer);
}
*/
	/*
	id_x = logGetVarId("stateEstimate", "x");
	id_y = logGetVarId("stateEstimate", "y");
	id_z = logGetVarId("stateEstimate", "z");

	id_roll = logGetVarId("stateEstimate", "roll");
	id_pitch = logGetVarId("stateEstimate", "pitch");
	id_yaw = logGetVarId("stateEstimate", "yaw");
	*/

static uint8_t con_sts;

__attribute__((noreturn)) static void led_task(void* arg) {
	(void)arg;

	uint8_t old_con_sts = ~con_sts;
	uint8_t parity = 0;
	while (1) {
		if (old_con_sts != con_sts) {
			if (con_sts == CON_CONNECTED) {
				ledSet(LED_GREEN_L, 1);
			}
			else {
				ledSet(LED_GREEN_L, 0);
			}

			old_con_sts = con_sts;
		}

		if (parity) {
				ledSet(LED_BLUE_L, 0);
		}
		else {
				ledSet(LED_BLUE_L, 1);
		}

		parity = !parity;

		vTaskDelay(M2T(500));
	}

	__builtin_unreachable();
}

__attribute__((noreturn)) static void con_task(void* arg) {
	(void)arg;

	uart1Init(115200);

	uint8_t byte;
	while (1) {
		uart1Getchar(&byte);

		switch (byte) {
			case CON_RESET:
				do {
					uart1Putchar(CON_RESET);
				} while (uart1GetDataWithTimeout(&byte, 0));
				uart1Putchar(CON_MAGIC);
				con_sts = CON_CONNECTED;
				break;
			default:
				continue;
		}
	}

	__builtin_unreachable();
}

void appInit(void) {
	con_sts = CON_DISCONNECTED;

	xTaskCreate(con_task, "Connection Manager", 2 * configMINIMAL_STACK_SIZE, NULL, 1, NULL);
	xTaskCreate(led_task, "LED Manager", 2 * configMINIMAL_STACK_SIZE, NULL, 1, NULL);
}
