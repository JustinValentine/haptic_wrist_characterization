# p4wrist

A small real-time driver for the three-joint P4 wrist: pronation–supination (PS) on puck 6,
flexion–extension (FE) and ulnar–radial deviation (UR) on a cable differential driven by pucks 7
and 8. C++17, no dependencies beyond the standard library and pthreads; Linux SocketCAN for
hardware, a callback bus for simulation on any OS. A C API (`include/p4wrist/p4wrist.h`) serves
C, C++ and Python (`python/p4wrist.py`, ctypes, no compiled extension).

It exists so the wrist can be run on the bench without the WAM, its passive J5, or its safety
board, while keeping libbarrett-canopen's bus contract: the same pucks work on the WAM and here
without reconfiguring.

## Build and test

```bash
cmake -S p4wrist -B p4wrist/build && cmake --build p4wrist/build
p4wrist/build/p4wrist_tests                      # C++ unit tests (transmission, framing, config)
python -m unittest tests.test_p4wrist            # the full library against emulated pucks
```

## Use

```python
from wrist_backend import p4wrist          # or put p4wrist/python on sys.path

config = p4wrist.default_config()          # nodes 6-8, wam8g transmission, 500 Hz, limits...
config.realtime_priority = 80              # SCHED_FIFO loop thread (needs CAP_SYS_NICE)
with p4wrist.Wrist.socketcan('can0', config) as w:
    kept, report = w.connect(previous_record)        # keeps the zero if no puck was reset
    if not kept:
        w.zero(p4wrist.zero_cal(home), exact=True)   # wrist placed at home
    w.start()                                         # 500 Hz SYNC loop (its own thread)
    w.enable()                                        # CiA-402 walk to CST, zero torque
    w.set_command(p4wrist.hold(w.sample().q, kp=(20, 10, 10), kd=(0.1, 0.1, 0.1)))
    ...                                               # command at least every 100 ms (watchdog)
    log, dropped = w.read_log()                       # every cycle, numpy structured array
    record = w.zero_record()                          # save it for the next run
```

Commands are per joint: PD about a setpoint (a NaN position integrates the setpoint from the
velocity, as moteus does) plus feedforward torque, or a stay-within wall (feedforward inside
`[lower, upper]`, PD toward a violated bound), each with its own torque cap; `brake = 1` puts the
drives in IDLE with grounded phases. The loop computes joint torques at 500 Hz and maps them to
motor torques through j2mp^−T, scaling motors that share a joint together when one saturates.

## Bus layout

As libbarrett-canopen's WAM contract (`doc/wam8g.md`):

| Frame | COB-ID | Content |
|---|---|---|
| SYNC | 0x080 | every 2 ms |
| Torque quad B | 0x205 | slot 0 unused (the WAM's passive J5), slots 1–3: motors 6, 7, 8, per mille of rated torque, latched at the next SYNC |
| Controlword + mode | 0x28A | the WAM safety board's broadcast COB (this library stands in for the board); applied on receipt; refreshed every 25 cycles |
| Position (+ q current) | 0x180 + node | TPDO1 every SYNC: 0x6064 (+ 0x6078 when the drive maps it) |
| Statusword + mode display | 0x280 + node | TPDO2; a P4 sends it every 100 ms whatever its type |
| Telemetry | 0x380 + node | TPDO3 every 50 SYNCs: bus voltage, amplifier and motor temperature (each optional) |
| Host heartbeat | 0x77F | every 250 ms |
| Puck heartbeats | 0x700 + node | 250 ms, supervised |

## Startup

`connect()` with a zero record first checks whether every puck has kept counting since it was
made: heartbeat period still the one this library sets (a reset restores the saved one), position
counter still in step with the absolute single-turn encoder (either direction). If so it only
puts the pucks in pre-operational and reconfigures them, keeping the zero. Otherwise it resets
each puck individually (never a broadcast reset) and waits for its boot-up (only state 0x00
counts; a P4 sends one more heartbeat after a reset).

Each puck is then configured with libbarrett-canopen's sequence: identity check, encoder
resolution, rated torque (refused if unset), Kt and peak current; the persisted torque target
cleared; every PDO remapped by the CiA 301 procedure (invalidate, clear, map, count, type,
re-link); unused PDOs disabled; heartbeat producer; SYNC and interpolation periods (both needed
for CST on fw 4.3.2).

`zero()` sets per-motor offsets so the joint angles read the home pose, either with the wrist
exactly at home or, given a single-turn survey (`zero_calibration.py`), from anywhere within half
a motor turn of home.

`start()` puts the pucks in operational and starts the loop. Once SYNC runs, the loop writes the
drives' SYNC-loss timeout (0x2406, 20 ms) through a non-blocking SDO queue: written earlier, a
drive could fault before the first SYNC. `enable()` walks the CiA-402 state machine through the
broadcast controlword (fault reset if needed, shutdown, enable operation) with the CST mode byte
in every frame, because fw 4.3.2 accepts a mode only while Operation Enabled, and waits for every
drive to report Operation Enabled in CST.

## The cycle

1. Handle frames that arrived since the last cycle (statuswords, telemetry, EMCY, heartbeats, SDO
   replies); late positions from an earlier SYNC are dropped.
2. Send SYNC (and the host heartbeat every 125 cycles).
3. Collect this SYNC's positions (up to `tpdo_wait_s`).
4. Joint angles from the zeroed motor angles; velocity from differenced positions through a
   first-order filter.
5. Advance the runtime SDO queue.
6. Take the caller's latest command (try-lock: the loop never blocks on the caller).
7. Supervise, run the state machine, compute torques.
8. Send the torque quad (latched at the next SYNC) and the controlword when due.
9. Publish the sample (try-lock) and append it to a lock-free log.

A torque computed from the positions of SYNC k is latched at SYNC k+1 and ramped to over the
following period (CST), so it is reached 2 periods (4 ms) after the positions it came from.

## Safety

No safety board watches the bench, so the loop does, every cycle: joint position and velocity
limits (brake with grounded phases for `brake_s`, then disable); per-joint and per-motor torque
caps; the host watchdog (`command_timeout_s`); missing positions; missing heartbeats; puck resets
(which also void the zero); drive faults from EMCY or the statusword; drives leaving Operation
Enabled; CAN errors; repeated loop overruns. Faults latch until `clear_fault()`, and `enable()`
refuses outside the joint limits or without a zero. If the process dies, SYNC stops and the
drives fault after the SYNC-loss timeout; a clean `stop()` disables the drives and disarms that
timeout first. Nothing here replaces a hardware E-stop on the motor supply.

## Simulation

`p4w_open_callback()` takes send/receive callbacks instead of a CAN socket; the loop is then
stepped by the caller (`step`, `wait_until`), so simulated time is deterministic. The
characterization repo's `p4_sim.py` supplies emulated pucks (barrett-puck's MockP4, extended with
the P4 behaviour above) and a plant, through `Wrist.callback(bus, config)`.

## To verify on hardware

The library follows libbarrett-canopen's bench-proven sequences where they exist. These parts
are new and were checked only against the emulator:
* mapping q-axis current (0x6078) into the position TPDO (6 bytes) and the slow telemetry TPDO3
  (both fall back gracefully if the drive refuses the mapping);
* writing 0x2406 while operational, and whether a drive with 0x2406 set faults before the first
  SYNC;
* enabling through the broadcast controlword without a safety board (the board does exactly this);
* IDLE as a brake (the manual: "motor phase leads tied to ground");
* the current-to-torque convention (Kt per A rms, current as phase peak);
* the positive directions of FE and UR (`wrist_check.py nudge`).

Sources: libbarrett-canopen (`include/barrett/bus/canopen.h`, `src/bus/canopen.cpp`,
`doc/wam8g.md`, `doc/linux-sync-timing.md`, the 2026-09 bench notes), the Barrett P4 CANopen
Manual Rev BA, and the P4 EDS (`puck4.eds`).
