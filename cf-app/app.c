/*
 * app.c — Crazyflie App Layer firmware: handshake + bitfield serial control.
 *
 * PROTOCOL (v2 — replaces the old 0xAA/0xBB 32-byte float frames entirely)
 *
 * Nothing is accepted until the reset handshake completes on uart1:
 *
 *     companion -> CF :  RESET (0xB8)          (usually a BURST of them)
 *     CF -> companion :  RESET (0xB8)          echoed for EVERY reset read
 *         ... CF keeps draining queued RESETs, echoing each one ...
 *     CF -> companion :  MAGIC (0x3E)          sent once the RX queue goes
 *                                              quiet; the N6 does NOT echo it
 *
 * After sending that MAGIC the CF waits for instructions. Every instruction
 * is framed:
 *
 *     [MAGIC 0x3E] [instruction byte] [payload...]
 *
 * The only instruction is FLIGHTCMD (0x01), carrying ONE payload byte whose
 * bits are OR'd motion flags (opposing bits cancel):
 *
 *     0x01 forward   0x02 back      0x04 right     0x08 left
 *     0x10 up        0x20 down      0x40 yaw right 0x80 yaw left
 *
 * Commands are SMOOTHED on this side (see "Command smoothing" below): a changed
 * payload is debounced for CMD_DEBOUNCE_MS, then velocities ease toward the
 * decoded target through a first-order low-pass -- the N6 can be twitchy and
 * the airframe still flies smooth ramps.
 *
 * Flight sequencing: TWO independent gates must BOTH pass before takeoff,
 * in either order:
 *     1. the RESET/MAGIC handshake completes (companion is alive), and
 *     2. the PILOT arms the drone (cfclient Arm button).
 * The app NEVER arms the drone itself (AUTO_ARM_AFTER_HANDSHAKE=false); arming
 * is the human safety gate. Once both gates pass and spin-up verification
 * finishes, it takes off to DEFAULT_Z_M exactly once, then payload bits steer.
 * If the pilot disarms mid-air (emergency), the app stops commanding and
 * returns to waiting for a fresh arm.
 *
 * BRUSHLESS ARMING GOTCHA (the "armed at 10% but never climbs" bug): on the
 * 2.1 brushless, arming enters supervisorStateArming — motors idle-spin while
 * the firmware verifies each motor's RPM. Until that check passes and the
 * state reaches ReadyToFly, supervisorCanFly() is false and stabilizer.c
 * ZEROES every setpoint we send (setpoint = {0}). supervisorIsArmed() goes
 * true immediately on request, so gating takeoff on it starts streaming
 * setpoints that are silently discarded. We therefore gate FS_TAKEOFF on
 * supervisorIsArmed() && supervisorCanFly(), and print a hint if spin-up
 * verification hangs (failing RPM check: damaged prop / ESC telemetry).
 *
 * LEDs:
 *     LED_GREEN_R  ~2 Hz always          = appMain alive (heartbeat)
 *     LED_BLUE_L   ~5 Hz fast blink      = awaiting RESET/MAGIC handshake
 *                  ~1 Hz slow blink      = link up, awaiting pilot ARM
 *                  solid OFF             = armed / flying
 *
 * Failsafes (kept from v1 — they were hard-won):
 *     - payload older than CMD_HOLD_MS while FLYING   -> hover in place
 *     - no valid instruction for WATCHDOG_MS while FLYING -> auto-land
 *       (not during ARMING/TAKEOFF: the companion can't know when takeoff
 *        completes, so it isn't required to stream until we're flying)
 *     - vbat below VBAT_LAND_V for VBAT_DEBOUNCE_MS   -> auto-land
 *     - RESET byte at frame start while airborne      -> land, then re-handshake
 *       (a RESET mid-flight means the companion rebooted; its state is gone)
 *
 * After any landing the app returns to WAIT_LINK: a fresh handshake is
 * required to fly again.
 *
 * Firmware symbols verified against the local clone (../crazyflie-firmware):
 *     uart1Init / uart1GetDataWithTimeout(uint8_t*, ticks) -> bool   [uart1.h]
 *     uart1Putchar(int)                                              [uart1.h:166]
 *     LED_BLUE_L, LED_GREEN_R in led_t                               [led.h:61]
 *     supervisorRequestArming(bool) / supervisorIsArmed()            [supervisor.h]
 *     log vars stateEstimate.z and pm.vbat (both LOG_FLOAT)
 *     setpoint_t modeVelocity/modeAbs, COMMANDER_PRIORITY_EXTRX
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
#include "led.h"            // ledSet() heartbeat + handshake indicator
#include "debug.h"
#include "log.h"            // logGetVarId/logGetFloat for altitude + battery

#define DEBUG_MODULE "OMVNAV"

// ---- Protocol bytes ---------------------------------------------------------
#define BAUDRATE        115200
#define BYTE_RESET      0xB8u
#define BYTE_MAGIC      0x3Eu
#define INSTR_FLIGHTCMD 0x01u

// FLIGHTCMD payload bit flags (OR'd together; opposing bits cancel).
#define BIT_FWD         0x01u
#define BIT_BACK        0x02u
#define BIT_RIGHT       0x04u
#define BIT_LEFT        0x08u
#define BIT_UP          0x10u
#define BIT_DOWN        0x20u
#define BIT_YAWR        0x40u
#define BIT_YAWL        0x80u

// ---- Motion mapping ----------------------------------------------------------
#define VX_SPEED_MPS    0.65f   // forward/back speed while bit held
#define VY_SPEED_MPS    0.55f   // left/right sideways speed while bit held
#define YAW_SPEED_DPS   60.0f   // yaw rate while bit held
#define Z_RATE_MPS      0.30f   // climb/descend rate while up/down held

// ---- Command smoothing ---------------------------------------------------------
// The N6 flips payload bits on every perception twitch; executing each flip
// verbatim makes the flight jittery. Two-stage filter, both on this side:
//   1. DEBOUNCE: a changed payload must persist CMD_DEBOUNCE_MS before it
//      becomes the active target -- single-send flickers are ignored outright.
//   2. LOW-PASS: actual velocities ease toward the target with a first-order
//      filter (VEL_SMOOTH_ALPHA per 20ms tick; 0.12 -> ~150ms time constant),
//      so even legitimate target changes arrive as smooth ramps, and rapid
//      A/B/A alternation averages out instead of shaking the airframe.
#define CMD_DEBOUNCE_MS    0    // new payload must persist this long to apply
#define VEL_SMOOTH_ALPHA   0.95f // per-tick blend toward target (higher = snappier)
// Crazyflie body frame: +x forward, +y LEFT, +z up; positive yaw = CCW (left).
// If right/left or yaw steer backwards on your airframe, flip these two signs.
#define VY_SIGN         (+1.0f)
#define YAW_SIGN        (+1.0f)

// ---- Timing ------------------------------------------------------------------
#define READ_TIMEOUT_MS     20      // per-byte read timeout (drives loop rate)
#define CMD_HOLD_MS         400     // payload validity: older than this -> hover
#define WATCHDOG_MS         2000    // FLYING + silent this long -> auto-land
#define ARM_RETRY_MS        500     // auto-arm mode only: re-request period
#define SPINUP_HINT_MS      5000    // armed but !canFly this long -> start hinting
#define SPINUP_HINT_PERIOD  2000    // repeat the hint at this period

// Arming policy: false (default) = the PILOT arms via cfclient and acts as the
// human safety gate -- the app never arms itself. Set true to restore fully
// autonomous arming after the handshake (field operation without a laptop).
static const bool AUTO_ARM_AFTER_HANDSHAKE = false;
#define STATS_PERIOD_MS     1000    // 1 Hz diagnostic print
#define HEARTBEAT_MS        250     // GREEN_R toggle (~2 Hz) = alive
#define HANDSHAKE_BLINK_MS  100     // BLUE_L toggle (~5 Hz) = awaiting handshake
#define ARMWAIT_BLINK_MS    500     // BLUE_L toggle (~1 Hz) = awaiting pilot ARM
#define SETPOINT_PERIOD_MS  20      // setpoint output rate (50 Hz)

// ---- Altitude / takeoff / landing ---------------------------------------------
#define DEFAULT_Z_M         0.40f   // takeoff target altitude
#define Z_MIN_M             0.15f   // lowest commandable hold altitude
#define Z_MAX_M             1.20f   // highest commandable hold altitude
#define Z_AIRBORNE_M        0.25f   // est. altitude that counts as "off ground"
#define TAKEOFF_SETTLE_MS   2000    // hold above Z_AIRBORNE_M this long -> FLYING
#define LAND_STEP_M         0.005f  // z drop per setpoint tick (~0.25 m/s @20ms)
#define Z_FLOOR_M           0.05f   // disarm once descended to here

// ---- Battery failsafe (1S LiPo; sags hard under load) -------------------------
#define VBAT_LAND_V         3.00f
#define VBAT_DEBOUNCE_MS    3000

#define UART_GET(cptr)  uart1GetDataWithTimeout((cptr), M2T(READ_TIMEOUT_MS))

// ---- Link handshake state ------------------------------------------------------
typedef enum {
  HS_WAIT_RESET = 0,   // cold: nothing accepted until the first RESET arrives
  HS_DRAIN,            // echoing every queued RESET; a quiet gap -> send MAGIC
  HS_ESTABLISHED       // MAGIC sent (not echoed back); instruction stream live
} link_state_t;

static const char *HS_NAMES[] = { "WAIT_RESET", "DRAIN", "ESTABLISHED" };

// ---- Instruction parser state (only meaningful once ESTABLISHED) ---------------
typedef enum {
  RX_IDLE = 0,         // at frame boundary: expect MAGIC (or RESET = re-handshake)
  RX_INSTR,            // got MAGIC: expect instruction byte
  RX_PAYLOAD           // got FLIGHTCMD: expect the payload bits
} rx_state_t;

// ---- Flight state machine -------------------------------------------------------
typedef enum {
  FS_WAIT_LINK = 0,    // disarmed on ground, no handshake yet
  FS_WAIT_ARM,         // handshake done; waiting for the PILOT to arm (cfclient)
  FS_TAKEOFF,          // climbing to DEFAULT_Z, horizontal locked
  FS_FLYING,           // steering by FLIGHTCMD payload bits
  FS_LANDING           // controlled descent, disarm at floor, back to WAIT_LINK
} flight_state_t;

static const char *FS_NAMES[] = { "WAIT_LINK", "WAIT_ARM", "TAKEOFF", "FLYING", "LANDING" };

static setpoint_t setpoint;

// ---- Diagnostics (reset each stats window) --------------------------------------
static uint32_t g_bytes    = 0;   // bytes seen on uart1 this window
static uint32_t g_cmds     = 0;   // valid FLIGHTCMD frames this window
static uint32_t g_badInstr = 0;   // unknown instruction bytes this window
static float    g_vbat     = 0.0f;

// Console printf has no %f — print floats as signed milli-units.
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
  setpoint.velocity_body    = true;

  // EXTRX priority beats CRTP so this app wins over a connected radio, while
  // cfclient's emergency stop (Space) still works through the supervisor.
  commanderSetSetpoint(&setpoint, COMMANDER_PRIORITY_EXTRX);
}


void appMain(void)
{
  vTaskDelay(M2T(3000));              // let the system + decks boot

  DEBUG_PRINT("=====================================\n");
  DEBUG_PRINT("OpenMV nav app v2 (handshake protocol) booting\n");
  DEBUG_PRINT("  baud=%u  RESET=0x%02X MAGIC=0x%02X FLIGHTCMD=0x%02X\n",
              (unsigned)BAUDRATE, BYTE_RESET, BYTE_MAGIC, INSTR_FLIGHTCMD);

  uart1Init(BAUDRATE);
  supervisorRequestArming(false);     // stay disarmed until handshake completes
  DEBUG_PRINT("  uart1 up, disarmed. BLUE_L fast-blink = send me RESET.\n");
  DEBUG_PRINT("=====================================\n");

  // ---- Log var ids (looked up lazily) ----
  logVarId_t zId = 0, vbatId = 0;
  bool zOk = false, vbatOk = false;

  // ---- Timers ----
  const TickType_t now0 = xTaskGetTickCount();
  TickType_t lastInstrTick    = now0;  // last VALID instruction (watchdog feed)
  TickType_t lastPayloadTick  = 0;     // when the current payload arrived
  TickType_t lastStatsTick    = now0;
  TickType_t lastBeatTick     = now0;
  TickType_t lastBlinkTick    = now0;
  TickType_t lastSetpointTick = now0;
  TickType_t lastArmReqTick   = 0;
  TickType_t armedSinceTick   = 0;     // when arming was granted (0 = not yet)
  TickType_t lastSpinupHint   = 0;     // last "spin-up stuck" hint print
  TickType_t aboveFloorSince  = 0;     // first tick above Z_AIRBORNE_M (0 = none)
  TickType_t vbatLowSince     = 0;

  // ---- State ----
  link_state_t   ls = HS_WAIT_RESET;
  rx_state_t     rx = RX_IDLE;
  flight_state_t fs = FS_WAIT_LINK;
  uint8_t  payload   = 0;              // latest FLIGHTCMD bits (raw, as received)
  uint8_t  activePayload = 0;          // debounced bits actually being executed
  uint8_t  pendingPayload = 0;         // candidate bits waiting out the debounce
  TickType_t pendingSince = 0;
  float    vxCmd = 0.0f, vyCmd = 0.0f, yawCmd = 0.0f;   // low-pass filtered
  float    zHold     = DEFAULT_Z_M;
  bool     beatOn = false, blinkOn = false;

  while (true) {
    uint8_t c;
    if (UART_GET(&c)) {
      g_bytes++;

      // ================= Handshake layer =================
      if (ls == HS_WAIT_RESET) {
        if (c == BYTE_RESET) {
          uart1Putchar(BYTE_RESET);               // echo: "I heard you"
          ls = HS_DRAIN;
          DEBUG_PRINT("handshake: RESET echoed, draining queued resets\n");
        }
        // anything else while cold: ignore
      }
      else if (ls == HS_DRAIN) {
        // The companion streamed RESETs while waiting for us, so more are
        // likely queued. Echo EVERY one we process; the quiet-gap branch below
        // (read timeout) decides when the queue is clear and sends MAGIC.
        if (c == BYTE_RESET) {
          uart1Putchar(BYTE_RESET);
        }
        // non-RESET noise while draining: ignore
      }
      // ================= Instruction layer =================
      else { // HS_ESTABLISHED
        switch (rx) {
          case RX_IDLE:
            if (c == BYTE_MAGIC) {
              rx = RX_INSTR;
            } else if (c == BYTE_RESET) {
              // Companion rebooted. Its state is gone: get to the ground, then
              // require a fresh handshake. (If already grounded, re-handshake now.)
              DEBUG_PRINT("RESET at frame start: companion rebooted\n");
              if (fs == FS_TAKEOFF || fs == FS_FLYING) {
                fs = FS_LANDING;
                ls = HS_WAIT_RESET;               // it will re-send RESET later
              } else {
                uart1Putchar(BYTE_RESET);         // re-handshake from the ground
                ls = HS_DRAIN;
                if (fs == FS_WAIT_ARM) fs = FS_WAIT_LINK;
              }
            }
            // any other byte at a frame boundary: desync noise, skip it
            break;

          case RX_INSTR:
            if (c == INSTR_FLIGHTCMD) {
              rx = RX_PAYLOAD;
            } else {
              g_badInstr++;                       // unknown instruction: drop frame
              rx = RX_IDLE;
            }
            break;

          case RX_PAYLOAD:
            payload = c;
            lastPayloadTick = xTaskGetTickCount();
            lastInstrTick   = lastPayloadTick;    // feeds the watchdog
            g_cmds++;
            rx = RX_IDLE;
            break;
        }
      }
    }
    else if (ls == HS_DRAIN) {
      // Read timed out: uart1 was quiet for a full READ_TIMEOUT_MS window
      // (~230 byte-times at 115200) — the companion's reset burst is drained.
      // Send ONE MAGIC to signal the link is up. The N6 does NOT echo it.
      uart1Putchar(BYTE_MAGIC);
      ls = HS_ESTABLISHED;
      rx = RX_IDLE;
      lastInstrTick = xTaskGetTickCount();
      DEBUG_PRINT("handshake: queue clear, MAGIC sent -- awaiting instructions\n");
      if (fs == FS_WAIT_LINK) {
        fs = FS_WAIT_ARM;
        lastArmReqTick = 0;
        if (!AUTO_ARM_AFTER_HANDSHAKE) {
          DEBUG_PRINT("link up -- ARM via cfclient to take off\n");
        }
      }
    }

    TickType_t now = xTaskGetTickCount();

    // ---- Sensors (lazy log id lookup) ---------------------------------------
    if (!zOk)    { zId    = logGetVarId("stateEstimate", "z"); zOk    = logVarIdIsValid(zId); }
    if (!vbatOk) { vbatId = logGetVarId("pm", "vbat");         vbatOk = logVarIdIsValid(vbatId); }
    float estZ = zOk    ? logGetFloat(zId)    : 0.0f;
    g_vbat     = vbatOk ? logGetFloat(vbatId) : 0.0f;

    // ---- Battery failsafe: sustained low voltage while airborne -> land -----
    if (vbatOk && (fs == FS_TAKEOFF || fs == FS_FLYING)) {
      if (g_vbat < VBAT_LAND_V) {
        if (vbatLowSince == 0) vbatLowSince = now;
        else if ((now - vbatLowSince) > M2T(VBAT_DEBOUNCE_MS)) {
          fs = FS_LANDING;
          DEBUG_PRINT("BATTERY LOW (%d mV): landing\n", milli(g_vbat));
        }
      } else {
        vbatLowSince = 0;
      }
    } else {
      vbatLowSince = 0;
    }

    // ---- Link watchdog: FLYING + no valid instruction too long -> land ------
    // Scoped to FLYING only: the companion cannot know when takeoff completes
    // (the CF transmits nothing after the handshake MAGIC), so it must not be
    // required to stream during ARMING/TAKEOFF.
    if (fs == FS_FLYING && (now - lastInstrTick) > M2T(WATCHDOG_MS)) {
      fs = FS_LANDING;
      DEBUG_PRINT("LINK SILENT (>%dms): landing\n", (int)WATCHDOG_MS);
    }

    // ---- Await arming, then WAIT OUT spin-up verification --------------------
    // Arming normally comes from the PILOT (cfclient Arm button) -- the human
    // safety gate. supervisorIsArmed() goes true the instant arming is granted,
    // but on the brushless the supervisor then sits in supervisorStateArming
    // while it verifies motor RPM at idle. Until supervisorCanFly()
    // (ReadyToFly), the stabilizer ZEROES all setpoints -- taking off early
    // would silently idle at ~10% throttle forever. So takeoff waits for BOTH.
    if (fs == FS_WAIT_ARM) {
      if (supervisorIsArmed() && supervisorCanFly()) {
        fs = FS_TAKEOFF;
        zHold = DEFAULT_Z_M;
        aboveFloorSince = 0;
        armedSinceTick = 0;
        vxCmd = vyCmd = yawCmd = 0.0f;      // smoothing starts from rest, so the
        activePayload = pendingPayload = 0; // handoff to FLYING ramps in gently
        DEBUG_PRINT("pilot armed + spin-up verified -> TAKEOFF to %d mm\n",
                    milli(DEFAULT_Z_M));
      } else if (!supervisorIsArmed()) {
        // Waiting for the pilot's ARM click. (Auto-arm only if configured.)
        if (AUTO_ARM_AFTER_HANDSHAKE &&
            (lastArmReqTick == 0 || (now - lastArmReqTick) > M2T(ARM_RETRY_MS))) {
          lastArmReqTick = now;
          supervisorRequestArming(true);
        }
      } else {
        // Armed but canFly not granted yet: spin-up verification in progress.
        if (armedSinceTick == 0) armedSinceTick = now;
        if ((now - armedSinceTick) > M2T(SPINUP_HINT_MS) &&
            (now - lastSpinupHint) > M2T(SPINUP_HINT_PERIOD)) {
          lastSpinupHint = now;
          DEBUG_PRINT("armed %ds but spin-up check hasn't passed -- motor RPM\n",
                      (int)((now - armedSinceTick) / 1000));
          DEBUG_PRINT("  out of arming window? (bent prop / ESC RPM telemetry)\n");
        }
      }
    }

    // ---- External disarm while airborne (pilot emergency): stop commanding ---
    if ((fs == FS_TAKEOFF || fs == FS_FLYING) && !supervisorIsArmed()) {
      fs = FS_WAIT_ARM;
      armedSinceTick = 0;
      DEBUG_PRINT("externally disarmed -> WAIT_ARM\n");
    }

    // ---- Takeoff gate: stably airborne -> FLYING -----------------------------
    if (fs == FS_TAKEOFF) {
      if (estZ >= Z_AIRBORNE_M) {
        if (aboveFloorSince == 0) aboveFloorSince = now;
        if ((now - aboveFloorSince) > M2T(TAKEOFF_SETTLE_MS)) {
          fs = FS_FLYING;
          lastInstrTick = now;   // watchdog grace starts at handoff, not handshake
          DEBUG_PRINT("TAKEOFF complete at z=%d mm -> FLYING\n", milli(estZ));
        }
      } else {
        aboveFloorSince = 0;
      }
    }

    // ---- Setpoint output at fixed 50 Hz --------------------------------------
    if ((now - lastSetpointTick) >= M2T(SETPOINT_PERIOD_MS)) {
      lastSetpointTick = now;
      switch (fs) {
        case FS_WAIT_LINK:
        case FS_WAIT_ARM:
          break;                       // grounded, not ours to command: send nothing

        case FS_TAKEOFF:
          sendHoverSetpoint(0.0f, 0.0f, 0.0f, zHold);
          break;

        case FS_FLYING: {
          // Stage 1 -- debounce: only adopt a changed payload after it has
          // held steady for CMD_DEBOUNCE_MS. One-off flickers never steer.
          if (payload != activePayload) {
            if (payload != pendingPayload) {
              pendingPayload = payload;
              pendingSince = now;
            } else if ((now - pendingSince) >= M2T(CMD_DEBOUNCE_MS)) {
              activePayload = payload;
            }
          } else {
            pendingPayload = activePayload;
          }

          // Stale link -> target zero; the low-pass eases us into the hover.
          uint8_t p = activePayload;
          if (lastPayloadTick == 0 || (now - lastPayloadTick) > M2T(CMD_HOLD_MS)) {
            p = 0;
          }

          float vxT = 0.0f, vyT = 0.0f, yawT = 0.0f;
          if (p & BIT_FWD)   vxT += VX_SPEED_MPS;
          if (p & BIT_BACK)  vxT -= VX_SPEED_MPS;
          if (p & BIT_RIGHT) vyT -= VY_SIGN * VY_SPEED_MPS;    // body +y = LEFT
          if (p & BIT_LEFT)  vyT += VY_SIGN * VY_SPEED_MPS;
          if (p & BIT_YAWR)  yawT -= YAW_SIGN * YAW_SPEED_DPS; // +yaw = CCW
          if (p & BIT_YAWL)  yawT += YAW_SIGN * YAW_SPEED_DPS;
          const float dz = Z_RATE_MPS * (SETPOINT_PERIOD_MS / 1000.0f);
          if (p & BIT_UP)    zHold += dz;
          if (p & BIT_DOWN)  zHold -= dz;
          if (zHold > Z_MAX_M) zHold = Z_MAX_M;
          if (zHold < Z_MIN_M) zHold = Z_MIN_M;

          // Stage 2 -- low-pass each axis toward its target.
          vxCmd  += VEL_SMOOTH_ALPHA * (vxT  - vxCmd);
          vyCmd  += VEL_SMOOTH_ALPHA * (vyT  - vyCmd);
          yawCmd += VEL_SMOOTH_ALPHA * (yawT - yawCmd);

          sendHoverSetpoint(vxCmd, vyCmd, yawCmd, zHold);
          break;
        }

        case FS_LANDING:
          zHold -= LAND_STEP_M;
          if (zHold <= Z_FLOOR_M) {
            zHold = Z_FLOOR_M;
            supervisorRequestArming(false);
            fs = FS_WAIT_LINK;         // fresh handshake required to fly again
            if (ls == HS_ESTABLISHED) ls = HS_WAIT_RESET;
            // Tell the companion the session ended (it watches for a bare
            // RESET while connected) so it re-handshakes instead of deadlocking.
            uart1Putchar(BYTE_RESET);
            DEBUG_PRINT("LANDED: disarmed, awaiting new handshake\n");
          } else {
            sendHoverSetpoint(0.0f, 0.0f, 0.0f, zHold);
          }
          break;
      }
    }

    // ---- LEDs -----------------------------------------------------------------
    // GREEN_R ~2 Hz: appMain alive, always.
    if ((now - lastBeatTick) > M2T(HEARTBEAT_MS)) {
      lastBeatTick = now;
      beatOn = !beatOn;
      ledSet(LED_GREEN_R, beatOn);
    }
    // BLUE_L: ~5 Hz = awaiting handshake, ~1 Hz = link up + awaiting pilot ARM,
    // solid off = armed/flying.
    if (ls != HS_ESTABLISHED || fs == FS_WAIT_ARM) {
      const uint32_t period = (ls != HS_ESTABLISHED) ? HANDSHAKE_BLINK_MS
                                                     : ARMWAIT_BLINK_MS;
      if ((now - lastBlinkTick) > M2T(period)) {
        lastBlinkTick = now;
        blinkOn = !blinkOn;
        ledSet(LED_BLUE_L, blinkOn);
      }
    } else if (blinkOn) {
      blinkOn = false;
      ledSet(LED_BLUE_L, false);
    }

    // ---- 1 Hz diagnostics ------------------------------------------------------
    if ((now - lastStatsTick) > M2T(STATS_PERIOD_MS)) {
      lastStatsTick = now;
      DEBUG_PRINT("[stat] link=%s fs=%s armed=%d canFly=%d vbat=%dmV bytes/s=%u cmds/s=%u bad=%u\n",
                  HS_NAMES[ls], FS_NAMES[fs],
                  (int)supervisorIsArmed(), (int)supervisorCanFly(), milli(g_vbat),
                  (unsigned)g_bytes, (unsigned)g_cmds, (unsigned)g_badInstr);
      if (ls == HS_ESTABLISHED) {
        DEBUG_PRINT("[cmd]  raw=0x%02X active=0x%02X %s%s%s%s%s%s%s%s zHold=%dmm\n",
                    payload, activePayload,
                    (activePayload & BIT_FWD)   ? "FWD "  : "",
                    (activePayload & BIT_BACK)  ? "BACK " : "",
                    (activePayload & BIT_RIGHT) ? "RGT "  : "",
                    (activePayload & BIT_LEFT)  ? "LFT "  : "",
                    (activePayload & BIT_UP)    ? "UP "   : "",
                    (activePayload & BIT_DOWN)  ? "DWN "  : "",
                    (activePayload & BIT_YAWR)  ? "YAWR " : "",
                    (activePayload & BIT_YAWL)  ? "YAWL " : "",
                    milli(zHold));
        DEBUG_PRINT("[vel]  vx=%d vy=%d yaw=%d (smoothed, milli-units)\n",
                    milli(vxCmd), milli(vyCmd), milli(yawCmd));
      }
      g_bytes = 0;
      g_cmds = 0;
      g_badInstr = 0;
    }
  }
}
