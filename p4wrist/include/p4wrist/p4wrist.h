/*
 * p4wrist: a small real-time driver for the three-joint P4 wrist
 * (PS on puck 6, FE/UR differential on pucks 7 and 8) on a bare CAN bus.
 *
 * C API. Every struct here is plain data with only 4- and 8-byte fields, so
 * Python's ctypes (python/p4wrist.py) can mirror it field for field.
 *
 * Units: joint angles in rad, joint torques in N m, time in s. Joint order is
 * PS, FE, UR; motor order follows config.node (default pucks 6, 7, 8).
 *
 * Functions returning int give 0 on success and -1 on error; the message is
 * then in p4w_last_error() (per thread).
 */
#ifndef P4WRIST_H
#define P4WRIST_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define P4W_JOINTS 3
#define P4W_ABI_VERSION 1

enum { P4W_PS = 0, P4W_FE = 1, P4W_UR = 2 };

/* Wrist states (p4w_sample.wrist_state). */
enum {
	P4W_STATE_STOPPED = 0,    /* no SYNC loop running */
	P4W_STATE_DISABLED = 1,   /* loop running, drives disabled */
	P4W_STATE_ENABLING = 2,   /* walking the CiA-402 state machine */
	P4W_STATE_ACTIVE = 3,     /* torque control (CST) */
	P4W_STATE_BRAKE = 4,      /* drives enabled in IDLE mode: phases grounded */
	P4W_STATE_BRAKING = 5,    /* fault reaction: brake, then disable */
	P4W_STATE_DISABLING = 6,
	P4W_STATE_FAULT = 7       /* drives disabled, fault latched until cleared */
};

/* Per-joint command modes. Numbering follows moteus so logs read alike. */
enum {
	P4W_MODE_POSITION = 10,     /* PD about a setpoint plus feedforward */
	P4W_MODE_STAY_WITHIN = 13   /* feedforward inside [lower, upper], PD to a violated bound */
};

/* Fault codes (p4w_sample.fault). */
enum {
	P4W_FAULT_NONE = 0,
	P4W_FAULT_POSITION_LIMIT = 1,
	P4W_FAULT_VELOCITY_LIMIT = 2,
	P4W_FAULT_HOST_TIMEOUT = 3,     /* no new command within command_timeout_s */
	P4W_FAULT_FEEDBACK_LOST = 4,    /* position TPDO missing for feedback_timeout_cycles */
	P4W_FAULT_DRIVE = 5,            /* drive fault (statusword or EMCY); see emcy_code */
	P4W_FAULT_PUCK_RESET = 6,       /* a puck sent boot-up: its counter and zero are gone */
	P4W_FAULT_HEARTBEAT_LOST = 7,
	P4W_FAULT_DRIVE_DROPPED = 8,    /* a drive left Operation Enabled on its own */
	P4W_FAULT_ENABLE_TIMEOUT = 9,
	P4W_FAULT_BUS = 10,             /* CAN send failed */
	P4W_FAULT_SDO = 11,             /* a runtime SDO write failed */
	P4W_FAULT_OVERRUN = 12          /* the loop missed overrun_limit_cycles in a row */
};

typedef struct {
	int32_t node[P4W_JOINTS];           /* puck node ids of the three motors (6, 7, 8) */
	int32_t controlword_refresh_cycles; /* controlword/mode broadcast period (25 = 50 ms) */
	double j2mp[P4W_JOINTS][P4W_JOINTS];/* motor angle = j2mp * joint angle; rows motors, columns PS FE UR */
	double position_sign[P4W_JOINTS];   /* per motor, applied to 0x6064 counts */
	double torque_sign[P4W_JOINTS];     /* per motor, applied to torque targets and currents */
	double period_s;                    /* SYNC period (0.002) */
	double joint_min[P4W_JOINTS];       /* position limits (rad), absolute after zeroing */
	double joint_max[P4W_JOINTS];
	double max_joint_velocity[P4W_JOINTS];  /* rad/s */
	double max_joint_torque[P4W_JOINTS];    /* N m; caps every command */
	double velocity_filter_hz;          /* first-order filter on differenced position */
	double command_timeout_s;           /* host watchdog while ACTIVE or BRAKE */
	double brake_s;                     /* fault reaction: brake this long, then disable */
	double heartbeat_timeout_s;         /* puck heartbeat supervision */
	double tpdo_wait_s;                 /* how long after SYNC to wait for positions */
	double current_torque_factor;       /* motor N m per (A x Kt): P4 currents are phase peak,
	                                       Kt is per A rms, so 1/sqrt(2) (unverified) */
	int32_t max_motor_permille;         /* torque target cap, per mille of rated torque */
	int32_t sync_loss_timeout_ms;       /* written to 0x2406 once SYNC runs; 0 keeps the drive's */
	int32_t puck_heartbeat_ms;          /* written to 0x1017; also marks "configured since reset" */
	int32_t feedback_timeout_cycles;
	int32_t overrun_limit_cycles;       /* consecutive late cycles before a fault (0 = off) */
	int32_t telemetry_divider;          /* TPDO3 (bus voltage, temperatures) every n SYNCs */
	int32_t realtime_priority;          /* SCHED_FIFO priority of the loop thread; 0 = none */
	int32_t map_current;                /* 1: put q-axis current (0x6078) in the position TPDO */
} p4w_config;

typedef struct {
	double position;     /* rad; NaN keeps integrating the setpoint from velocity */
	double velocity;     /* rad/s, setpoint velocity (also the D-term reference) */
	double torque;       /* N m feedforward */
	double kp;           /* N m/rad */
	double kd;           /* N m s/rad */
	double max_torque;   /* N m, cap on this joint's output (<= config cap); NaN = config cap */
	double lower;        /* stay-within bounds (rad); NaN = open side */
	double upper;
	int32_t mode;        /* P4W_MODE_POSITION or P4W_MODE_STAY_WITHIN */
	int32_t reserved;
} p4w_joint_command;

typedef struct {
	p4w_joint_command joint[P4W_JOINTS];
	double t_host;       /* host timestamp, echoed in samples */
	uint32_t seq;        /* echoed in samples; set by p4w_set_command if 0 */
	int32_t brake;       /* 1: drives in IDLE (phases grounded) instead of torque control */
} p4w_command;

/* One control cycle. */
typedef struct {
	uint64_t cycle;
	double time;                        /* s, loop clock (p4w_now) at the SYNC */
	double q[P4W_JOINTS];               /* joint angle (rad) */
	double qd[P4W_JOINTS];              /* filtered joint velocity (rad/s) */
	double setpoint[P4W_JOINTS];        /* PD setpoint used this cycle (NaN in stay-within) */
	double setpoint_velocity[P4W_JOINTS];
	double tau_cmd[P4W_JOINTS];         /* joint torque commanded (after limits and saturation) */
	double tau_meas[P4W_JOINTS];        /* joint torque from measured q-axis currents (NaN if unmapped) */
	double motor_angle[P4W_JOINTS];     /* zeroed motor angle (rad) */
	double current[P4W_JOINTS];         /* q-axis current (A, drive sign) */
	double bus_voltage[P4W_JOINTS];     /* V, NaN until telemetry arrives */
	double amp_temp[P4W_JOINTS];        /* C */
	double motor_temp[P4W_JOINTS];      /* C (raw thermistor reading, if mapped) */
	double command_t_host;              /* t_host of the command used */
	double late_s;                      /* how late this cycle started */
	int32_t counts[P4W_JOINTS];         /* raw 0x6064 */
	int32_t permille[P4W_JOINTS];       /* torque targets sent */
	int32_t statusword[P4W_JOINTS];
	int32_t mode_display[P4W_JOINTS];
	int32_t joint_mode[P4W_JOINTS];
	int32_t wrist_state;
	int32_t fault;
	int32_t fault_node;
	int32_t emcy_code;                  /* last EMCY code seen (warnings included) */
	uint32_t command_seq;
	uint32_t missed_feedback;           /* bit i: motor i missed its position this cycle */
	int32_t zeroed;
	int32_t reserved;
} p4w_sample;

/* Zero calibration: the home pose, and optionally each motor's single-turn
   encoder (0x3012:01) reading at home with its direction relative to 0x6064,
   for "rough placement" zeroing within half a motor turn of home. */
typedef struct {
	double home[P4W_JOINTS];
	int32_t single_turn_at_home[P4W_JOINTS];
	int32_t single_turn_dir[P4W_JOINTS];   /* +1, -1, or 0 = unknown */
	int32_t has_single_turn;
	int32_t reserved;
} p4w_zero_cal;

/* What a zeroed wrist needs to keep its zero across program runs. */
typedef struct {
	double motor_offset[P4W_JOINTS];  /* rad, subtracted from sign*counts*2pi/cpr */
	int32_t counts[P4W_JOINTS];       /* 0x6064 when the record was made */
	int32_t single_turn[P4W_JOINTS];  /* 0x3012:01 when the record was made */
	int32_t valid;
	int32_t reserved;
} p4w_zero_record;

/* A bus supplied by the caller, e.g. emulated pucks in Python. send returns 0
   on success; recv returns 1 and fills the frame if one is waiting, else 0.
   The bus is synchronous: replies to a frame are available as soon as send
   returns, and the loop is stepped with p4w_step() instead of a thread, so
   simulated time is deterministic. */
typedef int32_t (*p4w_send_fn)(void* user, uint32_t id, const uint8_t* data, int32_t len);
typedef int32_t (*p4w_recv_fn)(void* user, uint32_t* id, uint8_t* data, int32_t* len);

typedef struct p4w_wrist p4w_wrist;

const char* p4w_last_error(void);
int p4w_abi_version(void);
/* sizeof the structs above, for bindings to check their mirrors. */
void p4w_struct_sizes(int32_t* config, int32_t* command, int32_t* sample,
		int32_t* zero_cal, int32_t* zero_record);

void p4w_default_config(p4w_config* config);

/* Open a SocketCAN interface (Linux). NULL on error. */
p4w_wrist* p4w_open_socketcan(const p4w_config* config, const char* interface);
/* Open a caller-supplied synchronous bus (any OS); see p4w_send_fn. */
p4w_wrist* p4w_open_callback(const p4w_config* config, p4w_send_fn send, p4w_recv_fn recv,
		void* user);
/* Disables the drives, stops the loop, closes the bus. */
void p4w_close(p4w_wrist* w);

/* Bring the pucks up (configured, pre-operational). With a valid record, first
   try to keep its zero: no NMT resets, and every puck must still be counting
   from the same zero (heartbeat period still ours, counter in step with the
   single-turn encoder). Otherwise reset and reconfigure; zeroing is then
   needed. *kept_zero says which; report (if given) says why. */
int p4w_connect(p4w_wrist* w, const p4w_zero_record* record, int32_t* kept_zero,
		char* report, int32_t report_len);
/* Read 0x6064 and 0x3012:01 of each motor (loop must be stopped). */
int p4w_read_encoders(p4w_wrist* w, int32_t* counts, int32_t* single_turn);
/* Zero with the wrist exactly at cal->home (exact != 0 or no single-turn data),
   or anywhere within half a motor turn of home using the single-turn encoders.
   from_home (optional) gets the joint offsets from home at zeroing. */
int p4w_zero(p4w_wrist* w, const p4w_zero_cal* cal, int32_t exact, double* from_home);
int p4w_get_zero_record(p4w_wrist* w, p4w_zero_record* record);
/* +1, -1 or 0 (ambiguous): the single-turn encoder's direction relative to
   0x6064, from one motion's counter and single-turn deltas. */
int32_t p4w_single_turn_direction(int32_t position_delta, int32_t single_turn_delta,
		int32_t counts_per_rev);

/* Start the SYNC loop: a real-time thread on SocketCAN, manual stepping on a callback bus. */
int p4w_start(p4w_wrist* w);
int p4w_stop(p4w_wrist* w);
/* Callback bus: run n control cycles. */
int p4w_step(p4w_wrist* w, int32_t cycles);
/* Loop clock (s): cycles x period on a callback bus, monotonic time on SocketCAN. */
double p4w_now(p4w_wrist* w);
/* SocketCAN: sleep until loop time t. Callback bus: step until loop time >= t. */
int p4w_wait_until(p4w_wrist* w, double t);

/* Enable torque control, holding the current pose (setpoints = q, zero
   torque) until the first command. Waits (or steps) up to timeout_s. */
int p4w_enable(p4w_wrist* w, double timeout_s);
int p4w_disable(p4w_wrist* w, double timeout_s);
/* Clear a latched fault (drives stay disabled). */
int p4w_clear_fault(p4w_wrist* w);

int p4w_set_command(p4w_wrist* w, const p4w_command* command);
int p4w_get_sample(p4w_wrist* w, p4w_sample* sample);
/* Copy up to max logged samples (oldest first); returns the count, or -1.
   *dropped gets how many were lost to a full log since the last read. */
int32_t p4w_read_log(p4w_wrist* w, p4w_sample* out, int32_t max, uint32_t* dropped);

/* Describe the latched fault (empty if none). */
int p4w_fault_text(p4w_wrist* w, char* text, int32_t text_len);
/* Drive parameters read at connect, per motor. */
int p4w_drive_info(p4w_wrist* w, double* rated_torque_mnm, double* kt_mnm_per_a,
		double* peak_current_ma, int32_t* counts_per_rev);

#ifdef __cplusplus
}
#endif

#endif
