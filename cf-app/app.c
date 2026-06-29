#include <stdint.h>
#include <stddef.h>

#include "app.h"
#include "led.h"
#include "uart1.h"

#include "FreeRTOS.h"
#include "task.h"

__attribute__((noreturn)) static void uartTask(void* arg) {
	uint8_t cntrl;

	uart1Init(115200);

	while (1) {
		if (uart1GetDataWithDefaultTimeout(&cntrl)) {
			switch (cntrl) {
				case 0:
					ledSet(LED_BLUE_L, 1);
					ledSet(LED_GREEN_L, 0);
					break;
				case 1:
					ledSet(LED_BLUE_L, 0);
					ledSet(LED_GREEN_L, 1);
					break;
				default:
					break;
			}
		}

		vTaskDelay(M2T(500));
	}

	__builtin_unreachable();
}

void appInit(void) {
	xTaskCreate(uartTask, "UART Handler", 2 * configMINIMAL_STACK_SIZE, NULL, 1, NULL);
}
