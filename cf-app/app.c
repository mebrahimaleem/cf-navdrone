#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include "FreeRTOS.h"
#include "task.h"

#include "app.h"
#include "commander.h"
#include "crtp_commander_high_level.h"
#include "supervisor.h"
#include "uart1.h"
#include "led.h"
#include "debug.h"
#include "log.h"

#define DEBUG_MODULE "NAVAPP"

#define BAUDRATE        115200
#define CON_RESET     	0xB8u
#define CON_MAGIC     	0x3Eu

#define BIT_FWD         0x01u
#define BIT_BACK        0x02u
#define BIT_RIGHT       0x04u
#define BIT_LEFT        0x08u
#define BIT_UP          0x10u
#define BIT_DOWN        0x20u
#define BIT_YAWR        0x40u
#define BIT_YAWL        0x80u

#define VX_SPEED_MPS    0.65f
#define VY_SPEED_MPS    0.55f
#define YAW_SPEED_DPS   60.0f

#define SMOOTH_ALPHA   0.95f

#define VY_SIGN         (+1.0f)
#define YAW_SIGN        (+1.0f)

#define READ_TIMEOUT_MS     1000

#define LED_PERIOD					250

#define Z_INIT_M         		0.30f
#define Z_DODGE_M           0.60f

#define TAKEOFF_TIME_S			2
#define LANDING_TIME_S			1
#define LANDING_ALT_M				0.03f

#define VBAT_MIN_V         	3.00f
#define VBAT_DEBOUNCE_MS    3000

#define UART_GET(byte)  uart1GetDataWithTimeout((byte), M2T(READ_TIMEOUT_MS))
#define UART_PUT(byte)	uart1Putchar(byte)

#define ARM_GUARD()			do if(!supervisorIsArmed()) return; while(0)
#define FATAL_GUARD()		do if(fs_state == FS_FATL) return; while(0)
#define CHECK_GROUND()	do if(crtpCommanderHighLevelIsStopped() && fs_state != FS_FATL) fs_state = FS_GRND; while(0)

static enum link_state_t {
	LS_COLD,
	LS_ESTB,
} link_state = LS_COLD;

static enum cmd_state_t {
	CS_IDL,
	CS_CMD,
	CS_DIS,
} cmd_state = CS_IDL;

static enum fs_state_t {
	FS_GRND,
	FS_FLYN,
	FS_FATL,
} fs_state = FS_GRND;

enum cmd_t {
	CMD_NONE = 0x00,
	CMD_FLCN = 0x01,
};

static inline void emergency_land(void) {
	ARM_GUARD();	

	commanderRelaxPriority();
	if (crtpCommanderHighLevelLand(LANDING_ALT_M, LANDING_TIME_S)) {
		fs_state = FS_FATL;
		return;
	}

	vTaskDelay(M2T(LANDING_TIME_S));

	CHECK_GROUND();
}

static inline void takeoff(void) {
	ARM_GUARD();
	FATAL_GUARD();

	commanderRelaxPriority();
	if (crtpCommanderHighLevelTakeoff(Z_INIT_M, TAKEOFF_TIME_S)) {
		fs_state = FS_FATL;
		return;
	}

	vTaskDelay(M2T(TAKEOFF_TIME_S * 1000));
	if (crtpCommanderHighLevelIsStopped()) {
		fs_state = FS_FATL;
		return;
	}

	fs_state = FS_FLYN;
}

static inline void set_setpoint(float vx, float vy, float z, float phi) {
	ARM_GUARD();
	FATAL_GUARD();

	if (fs_state != FS_FLYN) {
		takeoff();
	}

	if (fs_state != FS_FLYN) {
		fs_state = FS_FATL;
		return;
	}

	static setpoint_t setpoint;

  memset(&setpoint, 0, sizeof(setpoint));
  setpoint.mode.x   = modeVelocity;
  setpoint.mode.y   = modeVelocity;
  setpoint.mode.z   = modeAbs;
  setpoint.mode.yaw = modeVelocity;

  setpoint.velocity.x       = vx;
  setpoint.velocity.y       = vy;
  setpoint.position.z       = z;
  setpoint.attitudeRate.yaw = phi;
  setpoint.velocity_body    = true;

  commanderSetSetpoint(&setpoint, COMMANDER_PRIORITY_HIGHLEVEL);
}

__attribute__((noreturn)) static void led_task(void* arg) {
	(void)arg;

	static uint8_t parity = 0;

	while (1) {
		if (parity) {
			ledSet(LED_GREEN_L, 0);
			ledSet(LED_BLUE_L, 0);
		}

		switch (fs_state) {
			case FS_GRND:
			case FS_FLYN:
				ledSet(LED_GREEN_L, 1);
				break;
			case FS_FATL:
				ledSet(LED_BLUE_L, 1);
				break;
			default:
				break;
		}

		parity = ~parity;
		vTaskDelay(M2T(LED_PERIOD));
	}

	__builtin_unreachable();
}

__attribute__((noreturn)) static void nav_task(void* arg) {
	(void)arg;

	static uint8_t byte;
	static enum cmd_t cmd = CMD_NONE;
	static uint8_t pend;

	
  static logVarId_t vbat_id = 0;
	static float vbat = 0.0f;
	static float vbat_brownout_time = 0.0f;
	static TickType_t now;

	float vxT, vyT, vyawT, vx = 0, vy = 0, vyaw = 0;
	float pzT = Z_INIT_M, pz = Z_INIT_M;

	while(1) {
		now = xTaskGetTickCount();
		vbat = logGetFloat(vbat_id);

		if (vbat < VBAT_MIN_V && fs_state == FS_FLYN) {
			if (!vbat_brownout_time) {
				vbat_brownout_time = now + M2T(VBAT_DEBOUNCE_MS);
			}
			else if (now >= vbat_brownout_time) {
				DEBUG_PRINT("Battery browning out. Landing");
				fs_state = FS_FATL;
				emergency_land();
			}
		}
		else {
			vbat_brownout_time = 0.0f;
		}

		CHECK_GROUND();

		if (!UART_GET(&byte)) {
			DEBUG_PRINT("Timeout reading from UART");
			emergency_land();
			continue;
		}

		if (byte == CON_RESET) {
			link_state = LS_COLD;
			DEBUG_PRINT("UART link reset");
			UART_PUT(CON_RESET);
			emergency_land();

			while (UART_GET(&byte)) {
				if (byte == CON_RESET) {
					UART_PUT(CON_RESET);
				}
				else {
					DEBUG_PRINT("Unexpected byte %x during reset sequence", byte);
				}
			}

			UART_PUT(CON_MAGIC);

			link_state = LS_ESTB;
			DEBUG_PRINT("UART link established");
			continue;
		}

		if (link_state != LS_ESTB) {
			DEBUG_PRINT("Garbage byte %x during cold connection", byte);
			continue;
		}

		switch (cmd_state) {
			case CS_IDL:
				if (byte == CON_MAGIC) {
					cmd_state = CS_CMD;
				}
				else {
					DEBUG_PRINT("Garbage byte %x after command %x", byte, cmd);
				}
				break;
			case CS_CMD:
				cmd = byte;
				cmd_state = CS_DIS;
				switch (cmd) {
					case CMD_FLCN:
						pend = 1;
						break;
					default:
						DEBUG_PRINT("Unkown command %x", cmd);
						cmd_state = CS_IDL;
						break;
				}
				break;
			case CS_DIS:
				switch (cmd) {
					case CMD_NONE:
						break;
					case CMD_FLCN:
						vxT = vyT = vyawT = 0.0f;
						if (byte & BIT_FWD)   vxT += VX_SPEED_MPS;
						if (byte & BIT_BACK)  vxT -= VX_SPEED_MPS;
						if (byte & BIT_RIGHT) vyT -= VY_SIGN * VY_SPEED_MPS;
						if (byte & BIT_LEFT)  vyT += VY_SIGN * VY_SPEED_MPS;
						if (byte & BIT_YAWR)  vyawT -= YAW_SIGN * YAW_SPEED_DPS;
						if (byte & BIT_YAWL)  vyawT += YAW_SIGN * YAW_SPEED_DPS;
						if (byte & BIT_UP)    pzT = Z_DODGE_M;
						if (byte & BIT_DOWN)  pzT = Z_INIT_M;

						vx  += SMOOTH_ALPHA * (vxT  - vx);
						vy  += SMOOTH_ALPHA * (vyT  - vy);
						vyaw += SMOOTH_ALPHA * (vyawT - vyaw);
						pz += SMOOTH_ALPHA * (pzT - pz);

						set_setpoint(vx, vy, vyaw, pz);
						break;
				}

				pend--;
				if (!pend) {
					cmd_state = CS_IDL;
				}
				break;
		}
	}

	__builtin_unreachable();
}

void appInit(void) {
	xTaskCreate(led_task, "LED Manager", 2 & configMINIMAL_STACK_SIZE, NULL, 1, NULL);
	xTaskCreate(nav_task, "Navigation Manager", 2 & configMINIMAL_STACK_SIZE, NULL, 1, NULL);
}
