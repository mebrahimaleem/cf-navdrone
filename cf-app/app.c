/*
 * app.c — Crazyflie App Layer firmware for OpenMV serial control.
 *
 * The OpenMV N6 does ALL perception + control compute and streams flight
 * setpoints + a COMMAND byte over UART. This app runs a small flight state
 * machine on top of the Crazyflie commander:
 *
 *     IDLE --takeoff--> TAKEOFF --(airborne)--> FLYING --land--> LANDING --> IDLE
 *       ^                                          |                ^
 *       |                                          +---low batt-----+
 *       +-----------------reset--------- SHUTDOWN <--shutdown-- (any state)
 *
 * Wire frame (20 bytes, little-endian, MUST match the OpenMV packer exactly):
 *   [0xAA][0xBB][cmd u8][vx f32][vy f32][yawrate f32][z f32][checksum u8]
 *   checksum = (sum of the 17 payload bytes, cmd + 4 floats) & 0xFF
 *
 * cmd values (MUST match openmv_main.py):
 *   0 CMD_FLY      apply vx/vy/yaw/z normally (only acts while FLYING)
 *   1 CMD_TAKEOFF  arm + climb to DEFAULT_Z, horizontal locked to 0
 *   2 CMD_LAND     controlled descent to the floor, then disarm
 *   3 CMD_SHUTDOWN immediate motor cut (latched until CMD_RESET)
 *   4 CMD_RESET    clear SHUTDOWN/IDLE latches, ready to take off again
 *
 * Autonomous failsafes (no OpenMV involvement):
 *   - battery sags below VBAT_LAND_V for VBAT_DEBOUNCE_MS  -> LANDING
 *   - UART silent > WATCHDOG_MS after the link has come up  -> LANDING
 *
 * Units: vx,vy in m/s (body frame), yawrate in deg/s, z in m (absolute).
 *
 * ---------------------------------------------------------------------------
 * TROUBLESHOOTING — read this if "it's not running":
 *   - Watch the console in cfclient (Console tab). Boot banner + 1 Hz stats.
 *   - GREEN LED blinks ~2 Hz = appMain alive. No blink = app never started
 *     (check APP=1 in Makefile, Kbuild lists app.c, build actually flashed).
 *   - 1 Hz stats show: state, armed, vbat, bytes/s, pkts/s, checksum errors:
 *       bytes/s == 0           -> wiring / wrong uart driver / baud mismatch
 *       bytes/s > 0, pkts == 0 -> framing/checksum mismatch with OpenMV
 *       pkts > 0, stuck IDLE   -> never got CMD_TAKEOFF, or arming blocked
 *
 * All firmware symbols below were verified against crazyflie-firmware
 * @ d96d2abd (cloned in ../crazyflie-firmware):
 *   - uart1Init / uart1GetDataWithTimeout(uint8_t*, ticks) -> bool   [uart1.h]
 *     (NOTE: uart2GetDataWithTimeout has a different, buffer-based signature,
 *      so this code is wired to uart1 — the standard deck UART pins.)
 *   - supervisorRequestArming(bool) / supervisorIsArmed()           [supervisor.h]
 *   - LED_GREEN_R in led_t                                          [led.h]
 *   - log vars stateEstimate.z and pm.vbat (both LOG_FLOAT)
 *   - setpoint_t fields, modeVelocity/modeAbs, COMMANDER_PRIORITY_EXTRX
 * ---------------------------------------------------------------------------
 */
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include "FreeRTOS.h"
#include "task.h"

#include "app.h"
#include "commander.h"
#include "supervisor.h"     // supervisorRequestArming(), supervisorIsArmed()
#include "uart1.h"          // deck UART (TX/RX on the expansion header)
#include "led.h"            // ledSet() heartbeat
#include "debug.h"
#include "log.h"            // logGetVarId/logGetFloat for altitude + battery

#define DEBUG_MODULE "OMVNAV"

// ---- Protocol --------------------------------------------------------------
#define BAUDRATE        115200
#define H0              0xAAu
#define H1              0xBBu
#define PKT_PAYLOAD     17          // 1 cmd byte + 4 x float32

// Commands (MUST match openmv_main.py).
#define CMD_FLY         0u
#define CMD_TAKEOFF     1u
#define CMD_LAND        2u
#define CMD_SHUTDOWN    3u
#define CMD_RESET       4u

// ---- Timing ----------------------------------------------------------------
#define READ_TIMEOUT_MS     20      // per-byte read timeout (drives loop rate)
#define WATCHDOG_MS         300     // no valid packet this long -> failsafe land
#define STATS_PERIOD_MS     1000    // 1 Hz diagnostic print
#define HEARTBEAT_MS        250     // LED toggle period (-> ~2 Hz blink)
#define SETPOINT_PERIOD_MS  20      // setpoint output rate (50 Hz), decoupled
                                    // from the UART byte rate

// ---- Altitude / takeoff / landing ------------------------------------------
#define DEFAULT_Z_M         0.40f   // takeoff target altitude
#define Z_AIRBORNE_M        0.25f   // est. altitude that counts as "off ground"
#define TAKEOFF_SETTLE_MS   2000    // hold above Z_AIRBORNE_M this long -> FLYING
#define RAMP_MS             1000    // ramp vx/vy 0->full over this long at handoff
#define LAND_STEP_M         0.005f  // z drop per setpoint tick (~0.25 m/s @20ms)
#define Z_FLOOR_M           0.05f   // disarm once descended to here

// ---- Battery failsafe (Crazyflie 2.1 is 1S LiPo) ---------------------------
// vbat sags under motor load. 3.0V is the safe floor for a 1S LiPo at rest;
// under full takeoff load a healthy pack can easily read 3.0-3.1V, so the
// threshold must stay well below that. Require 3 continuous seconds so a
// momentary climbout sag never triggers a spurious landing.
#define VBAT_LAND_V         3.00f   // sustained below this -> auto land
#define VBAT_DEBOUNCE_MS    3000    // must stay low this long (ignores load sag)

#define UART_GET(cptr)  uart1GetDataWithTimeout((cptr), M2T(READ_TIMEOUT_MS))

// On-wire float payload (the 16 bytes after the cmd byte).
typedef struct __attribute__((packed)) {
  float vx;
  float vy;
  float yawrate;   // deg/s
  float z;         // m, absolute
} cmd_t;

// Compile-time guard: cmd byte + this struct must equal the on-wire payload.
_Static_assert(sizeof(cmd_t) == PKT_PAYLOAD - 1, "cmd_t size must match wire frame");

// ---- Flight state machine --------------------------------------------------
typedef enum {
  FS_IDLE = 0,   // disarmed on the ground, waiting for CMD_TAKEOFF
  FS_TAKEOFF,    // climbing to DEFAULT_Z, horizontal locked to 0
  FS_FLYING,     // OpenMV in control (with post-handoff velocity ramp)
  FS_LANDING,    // controlled descent, horizontal locked, disarm at floor
  FS_SHUTDOWN    // motors cut, latched until CMD_RESET
} flight_state_t;

static const char *FS_NAMES[] = { "IDLE", "TAKEOFF", "FLYING", "LANDING", "SHUTDOWN" };

static setpoint_t setpoint;

// ---- Diagnostics counters (reset each stats window) ------------------------
static uint32_t g_bytes     = 0;    // bytes read off UART this window
static uint32_t g_validPkts = 0;    // good packets this window
static uint32_t g_csErrors  = 0;    // checksum failures this window
static cmd_t    g_lastCmd   = {0};  // last decoded float setpoint
static uint8_t  g_lastReq   = CMD_FLY;
static float    g_vbat      = 0.0f;  // last battery voltage read

// Print a float as a signed integer in milli-units. The Crazyflie console
// printf does NOT support %f, so we scale by 1000 and print as int.
static int milli(float v) { return (int)(v * 1000.0f); }


static void sendHoverSetpoint(float vx, float vy, float yawrate, float z)
{
  memset(&setpoint, 0, sizeof(setpoint));
  setpoint.mode.x   = modeVelocity;
  setpoint.mode.y   = modeVelocity;
  setpoint.mode.z   = modeAbs;
  setpoint.mode.yaw = modeVelocity;

  setpoint.velocity.x       = vx;
  setpoint.velocity.y       = vy;
  setpoint.position.z       = z;
  setpoint.attitudeRate.yaw = yawrate;
  setpoint.velocity_body    = true;   // vx/vy are body-frame

  // EXTRX priority (3) sits above CRTP (2), so the onboard app wins
  // arbitration over a connected radio. cfclient's emergency-stop (Space)
  // goes through the supervisor, so the kill switch still works.
  commanderSetSetpoint(&setpoint, COMMANDER_PRIORITY_EXTRX);
}

static void armSet(bool arm)
{
  supervisorRequestArming(arm);
}


void appMain(void)
{
  vTaskDelay(M2T(3000));              // let the system + decks boot

  DEBUG_PRINT("=====================================\n");
  DEBUG_PRINT("OpenMV nav app booting\n");
  DEBUG_PRINT("  baud=%u payload=%d watchdog=%dms\n",
              (unsigned)BAUDRATE, (int)PKT_PAYLOAD, (int)WATCHDOG_MS);

  uart1Init(BAUDRATE);
  armSet(false);                      // stay disarmed; arm on CMD_TAKEOFF
  DEBUG_PRINT("  uart up, disarmed. Heartbeat=GREEN LED ~2Hz. Waiting for CMD_TAKEOFF...\n");
  DEBUG_PRINT("=====================================\n");

  // ---- UART frame parser state ----
  enum { WAIT_H0, WAIT_H1, READ_PAYLOAD, READ_CHK } pstate = WAIT_H0;
  uint8_t payload[PKT_PAYLOAD];
  int     idx = 0;
  uint8_t sum = 0;

  // ---- Log var ids (looked up lazily; UINT16_MAX == not found) ----
  logVarId_t zId = 0, vbatId = 0;
  bool zOk = false, vbatOk = false;

  // ---- Timers ----
  const TickType_t now0 = xTaskGetTickCount();
  TickType_t lastRx          = now0;
  TickType_t lastStatsTick   = now0;
  TickType_t lastBeatTick    = now0;
  TickType_t lastSetpointTick = now0;
  TickType_t aboveFloorSince = 0;     // first tick above Z_AIRBORNE_M (0 = none)
  TickType_t handoffTick     = 0;     // tick FLYING began (velocity ramp start)
  TickType_t vbatLowSince    = 0;     // first tick vbat dipped low (0 = none)

  // ---- Flight state ----
  flight_state_t fs = FS_IDLE;
  cmd_t  target = {0};                // last commanded flight setpoint
  float  zHold  = DEFAULT_Z_M;        // current commanded altitude (tracks state)
  bool   linkUp = false;              // true after the FIRST valid packet
  bool   ledOn  = false;

  while (true) {
    uint8_t c;
    if (UART_GET(&c)) {
      g_bytes++;
      switch (pstate) {
        case WAIT_H0:
          if (c == H0) pstate = WAIT_H1;
          break;
        case WAIT_H1:
          if (c == H1)      { pstate = READ_PAYLOAD; idx = 0; sum = 0; }
          else if (c == H0) { pstate = WAIT_H1; }       // resync on fresh H0
          else              { pstate = WAIT_H0; }
          break;
        case READ_PAYLOAD:
          payload[idx++] = c;
          sum += c;                                      // natural mod 256
          if (idx >= PKT_PAYLOAD) pstate = READ_CHK;
          break;
        case READ_CHK:
          if (c == sum) {
            uint8_t req = payload[0];
            memcpy(&target, &payload[1], sizeof(target)); // 16 float bytes
            g_lastCmd = target;
            g_lastReq = req;
            g_validPkts++;
            lastRx = xTaskGetTickCount();
            linkUp = true;

            // ---- Command -> state transitions ----
            switch (req) {
              case CMD_TAKEOFF:
                if (fs == FS_IDLE) {
                  armSet(true);
                  fs = FS_TAKEOFF;
                  aboveFloorSince = 0;
                  DEBUG_PRINT("CMD_TAKEOFF: arming, climbing to %d mm\n",
                              milli(DEFAULT_Z_M));
                }
                break;
              case CMD_LAND:
                if (fs == FS_TAKEOFF || fs == FS_FLYING) {
                  fs = FS_LANDING;
                  DEBUG_PRINT("CMD_LAND: descending from %d mm\n", milli(zHold));
                }
                break;
              case CMD_SHUTDOWN:
                if (fs != FS_SHUTDOWN) {
                  fs = FS_SHUTDOWN;
                  armSet(false);
                  DEBUG_PRINT("CMD_SHUTDOWN: motors cut\n");
                }
                break;
              case CMD_RESET:
                if (fs == FS_SHUTDOWN || fs == FS_IDLE) {
                  fs = FS_IDLE;
                  armSet(false);
                  zHold = DEFAULT_Z_M;
                  DEBUG_PRINT("CMD_RESET: ready for takeoff\n");
                }
                break;
              case CMD_FLY:
              default:
                break;   // setpoint stored; only applied while FLYING
            }
          } else {
            g_csErrors++;
          }
          pstate = WAIT_H0;
          break;
      }
    }

    TickType_t now = xTaskGetTickCount();

    // ---- Read sensors (lazy log id lookup) ----------------------------------
    if (!zOk)    { zId    = logGetVarId("stateEstimate", "z"); zOk    = logVarIdIsValid(zId); }
    if (!vbatOk) { vbatId = logGetVarId("pm", "vbat");         vbatOk = logVarIdIsValid(vbatId); }
    float estZ = zOk    ? logGetFloat(zId)    : 0.0f;
    g_vbat     = vbatOk ? logGetFloat(vbatId) : 0.0f;

    // ---- Battery failsafe: sustained low voltage -> land --------------------
    // Only while airborne, and only if we actually have a battery reading.
    if (vbatOk && (fs == FS_TAKEOFF || fs == FS_FLYING)) {
      if (g_vbat < VBAT_LAND_V) {
        if (vbatLowSince == 0) vbatLowSince = now;
        else if ((now - vbatLowSince) > M2T(VBAT_DEBOUNCE_MS)) {
          fs = FS_LANDING;
          DEBUG_PRINT("BATTERY LOW (%d mV): landing\n", milli(g_vbat));
        }
      } else {
        vbatLowSince = 0;   // recovered (load sag) -> reset debounce
      }
    } else {
      vbatLowSince = 0;
    }

    // ---- Link-loss failsafe: silent UART -> land ---------------------------
    // Armed only after the link has come up once, so a slow OpenMV boot can't
    // disarm the drone before it ever connects.
    if (linkUp && (now - lastRx) > M2T(WATCHDOG_MS) &&
        (fs == FS_TAKEOFF || fs == FS_FLYING)) {
      fs = FS_LANDING;
      DEBUG_PRINT("LINK LOST (>%dms): landing\n", (int)WATCHDOG_MS);
    }

    // ---- Takeoff gate: TAKEOFF -> FLYING once stably airborne ---------------
    if (fs == FS_TAKEOFF) {
      if (estZ >= Z_AIRBORNE_M) {
        if (aboveFloorSince == 0) aboveFloorSince = now;
        if ((now - aboveFloorSince) > M2T(TAKEOFF_SETTLE_MS)) {
          fs = FS_FLYING;
          handoffTick = now;
          DEBUG_PRINT("TAKEOFF complete at z=%d mm -> FLYING\n", milli(estZ));
        }
      } else {
        aboveFloorSince = 0;   // dipped back down (baro/flow noise) -> reset
      }
    }

    // ---- Setpoint output at a fixed rate (decoupled from byte rate) ---------
    if ((now - lastSetpointTick) >= M2T(SETPOINT_PERIOD_MS)) {
      lastSetpointTick = now;
      switch (fs) {
        case FS_IDLE:
        case FS_SHUTDOWN:
          // disarmed; send nothing
          break;

        case FS_TAKEOFF:
          zHold = DEFAULT_Z_M;
          sendHoverSetpoint(0.0f, 0.0f, 0.0f, zHold);
          break;

        case FS_FLYING: {
          // Ramp horizontal velocity in over RAMP_MS so the blind-in-hover
          // vx command doesn't lurch the drone at handoff.
          TickType_t dtHand = now - handoffTick;
          float k = (dtHand >= M2T(RAMP_MS)) ? 1.0f
                    : (float)dtHand / (float)M2T(RAMP_MS);
          zHold = target.z;
          sendHoverSetpoint(target.vx * k, target.vy * k, target.yawrate, zHold);
          break;
        }

        case FS_LANDING:
          zHold -= LAND_STEP_M;
          if (zHold <= Z_FLOOR_M) {
            zHold = Z_FLOOR_M;
            armSet(false);
            fs = FS_IDLE;
            DEBUG_PRINT("LANDED: disarmed\n");
          } else {
            sendHoverSetpoint(0.0f, 0.0f, 0.0f, zHold);
          }
          break;
      }
    }

    // ---- Heartbeat LED: proves appMain is alive even with no console -------
    if ((now - lastBeatTick) > M2T(HEARTBEAT_MS)) {
      lastBeatTick = now;
      ledOn = !ledOn;
      ledSet(LED_GREEN_R, ledOn);
    }

    // ---- 1 Hz diagnostics --------------------------------------------------
    if ((now - lastStatsTick) > M2T(STATS_PERIOD_MS)) {
      lastStatsTick = now;
      DEBUG_PRINT("[stat] %s armed=%d vbat=%dmV bytes/s=%u pkts/s=%u csErr=%u req=%u\n",
                  FS_NAMES[fs],
                  (int)supervisorIsArmed(),
                  milli(g_vbat),
                  (unsigned)g_bytes,
                  (unsigned)g_validPkts,
                  (unsigned)g_csErrors,
                  (unsigned)g_lastReq);
      DEBUG_PRINT("[stat] setpoint (milli): vx=%d vy=%d yaw=%d z=%d zHold=%d\n",
                  milli(g_lastCmd.vx), milli(g_lastCmd.vy),
                  milli(g_lastCmd.yawrate), milli(g_lastCmd.z), milli(zHold));
      g_bytes = 0;
      g_validPkts = 0;
      g_csErrors = 0;
    }
  }
}
