// The wrist: three P4 pucks, the PS/FE/UR transmission, and a SYNC-driven
// control loop with its own safety supervision (there is no safety board on
// the bench: this loop is the only thing watching limits and liveness).
//
// Threading: on SocketCAN the loop runs in its own (optionally SCHED_FIFO)
// thread and never blocks on the caller: commands and samples go through a
// try-locked mailbox, logged samples through a lock-free ring. On a callback
// bus the caller steps the loop itself and everything is single-threaded.
#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "p4wrist/can.h"
#include "p4wrist/canopen.h"
#include "p4wrist/p4wrist.h"
#include "p4wrist/transmission.h"

namespace p4wrist {

// Fold a count difference into one turn: [-cpr/2, cpr/2).
int withinTurn(int counts, int cpr);
// +1, -1 or 0 (ambiguous) from a motion's counter and single-turn deltas.
int singleTurnDirection(int position_delta, int single_turn_delta, int cpr);

// What a puck reports about itself, plus where the position TPDO carries what.
struct DriveInfo : p4w_drive_info {
	DriveInfo() : p4w_drive_info{} {}
	int current_offset = -1;   // byte offsets in TPDO1 (-1: not mapped)
	int demand_offset = -1;
};

class Wrist {
public:
	Wrist(const p4w_config& config, std::unique_ptr<CanBus> bus, bool manual_clock);
	~Wrist();
	Wrist(const Wrist&) = delete;
	Wrist& operator=(const Wrist&) = delete;

	// --- setup (loop stopped) ---------------------------------------------
	// Returns true if the record's zero was kept (no resets); see p4w_connect.
	bool connect(const p4w_zero_record* record, std::string* report);
	void readEncoders(int32_t counts[3], int32_t single_turn[3]);
	Vec3 zero(const p4w_zero_cal& cal, bool exact);  // returns offsets from home
	p4w_zero_record zeroRecord() const { return record_; }
	const DriveInfo& driveInfo(int motor) const { return drive_[motor]; }
	void setDriveGains(int motor, const p4w_drive_gains& gains, double timeout_s);

	// --- loop -------------------------------------------------------------------
	void start();
	void stop();
	void step(int cycles);            // manual clock only
	double now() const;
	void waitUntil(double t);
	void enable(double timeout_s);
	void disable(double timeout_s);
	void clearFault();

	void setCommand(const p4w_command& command);
	p4w_sample sample() const;
	int readLog(p4w_sample* out, int max, uint32_t* dropped);
	std::string faultText() const;

private:
	struct Published {
		p4w_sample sample{};
		char fault_text[256] = {};
	};
	enum SdoKind { SDO_PLAIN, SDO_ARM_SYNC_LOSS };
	struct RuntimeSdo {
		int node;
		uint16_t index;
		uint8_t sub;
		uint32_t value;
		int nbytes;
		int kind = SDO_PLAIN;
		bool caller = false;   // requested through setDriveGains (counted in sdo_completed_)
	};
	void restoreGains();
	static std::vector<RuntimeSdo> gainWrites(int node, const p4w_drive_gains& g);
	void writeGainsDirect(int node, const p4w_drive_gains& g);
	void updateDriveLoop(double t);
	double advanceSetpoint(int j, const p4w_joint_command& c);
	void sendDriveTargets();
	enum Request { REQ_NONE = 0, REQ_ENABLE, REQ_DISABLE, REQ_CLEAR, REQ_QUIESCE };
	enum EnablePhase { EN_FAULT_RESET, EN_SHUTDOWN, EN_OPERATION };

	void configurePuck(Master& master, int motor);
	void mapPdo(Master& master, int node, uint16_t comm, uint16_t map, uint32_t cob, uint8_t type,
			const std::vector<std::pair<uint32_t, bool>>& entries, std::vector<bool>* mapped);
	bool encodersStill(Master& master, int node, int32_t* counts, int32_t* single_turn);

	void threadMain();
	void cycle();
	void dispatch(const CanFrame& f, bool collecting, uint32_t* got, double t);
	void supervise(double t);
	void runStateMachine(double t);
	void control(double t, Vec3* tau_joint);
	void sendTorques(const Vec3& tau_joint);
	void sendControl(uint16_t cw, int8_t mode);
	void raise(int code, int motor, const std::string& text, bool brake, double t);
	bool drivesAll(bool (*pred)(uint16_t), double since) const;
	void publish(const p4w_sample& s);
	bool waitFor(const std::function<bool()>& done, double timeout_s);
	void send(const CanFrame& f);
	bool receive(CanFrame* f, double timeout_s);
	int motorOfNode(int node) const;

	p4w_config cfg_;
	std::unique_ptr<CanBus> bus_;
	const bool manual_;
	Transmission tx_;
	DriveInfo drive_[3];
	p4w_zero_record record_{};
	std::atomic<bool> zeroed_{false};

	// Loop-owned state.
	std::thread thread_;
	std::atomic<bool> running_{false};
	std::atomic<bool> quit_{false};
	std::atomic<int> request_{REQ_NONE};
	double t0_ = 0.0;
	uint64_t cycle_ = 0;
	double t_sync_prev_ = 0.0;
	int state_ = P4W_STATE_STOPPED;
	int fault_ = P4W_FAULT_NONE;
	int fault_motor_ = -1;
	std::string fault_text_;
	int emcy_code_ = 0;
	double state_since_ = 0.0;
	double brake_until_ = 0.0;
	double enable_deadline_ = 0.0;
	int enable_phase_ = EN_FAULT_RESET;
	double phase_since_ = 0.0;
	uint16_t cw_ = co::CW_DISABLE_VOLTAGE;
	int8_t mode_byte_ = co::MODE_IDLE;
	uint64_t last_control_cycle_ = 0;
	bool control_dirty_ = true;
	int late_cycles_ = 0;
	double late_s_ = 0.0;

	int32_t counts_[3] = {};
	bool have_demand_[3] = {};
	double current_[3] = {};
	bool have_feedback_[3] = {};
	int missed_[3] = {};
	uint32_t missed_mask_ = 0;
	uint16_t statusword_[3] = {};
	int8_t mode_display_[3] = {};
	double statusword_time_[3] = {};
	double last_heartbeat_[3] = {};
	double bus_voltage_[3], amp_temp_[3], motor_temp_[3];
	Vec3 q_{}, qd_{}, q_prev_{}, motor_angle_{};
	bool have_q_prev_ = false;
	Vec3 setpoint_{}, setpoint_velocity_{}, held_setpoint_{};
	bool have_setpoint_[3] = {};
	int joint_mode_[3] = {};
	Vec3 tau_cmd_{};
	int32_t permille_[3] = {};

	std::vector<RuntimeSdo> sdo_queue_;
	size_t sdo_next_ = 0;
	bool sdo_waiting_ = false;
	double sdo_sent_at_ = 0.0;
	bool sync_loss_armed_ = false;
	int sync_loss_acks_ = 0;
	std::atomic<bool> sdo_idle_{true};
	// SDO writes requested by the caller while the loop runs.
	std::mutex sdo_mutex_;
	std::vector<RuntimeSdo> sdo_requests_;
	std::atomic<uint64_t> sdo_requested_{0}, sdo_completed_{0};
	bool gains_changed_[3] = {};
	bool gains_known_[3] = {};

	// Drive loop (CSP): 0 off, 1 arming (targets = current position, still CST), 2 on.
	int csp_phase_ = 0;
	int csp_arm_cycles_ = 0;
	double csp_since_ = 0.0;
	int32_t motor_target_[3] = {};
	Vec3 offset_nm_{};               // drive torque offsets (motor N m, drive frame)
	double torque_demand_[3] = {};   // drive's torque demand (motor N m, drive frame)

	// Caller <-> loop.
	mutable std::mutex command_mutex_;
	p4w_command command_{};
	bool command_valid_ = false;
	p4w_command active_command_{};
	bool active_valid_ = false;
	uint32_t active_seq_ = 0;
	uint32_t next_seq_ = 1;
	bool got_command_since_enable_ = false;
	double last_command_time_ = 0.0;

	mutable std::mutex publish_mutex_;
	Published published_;

	std::vector<p4w_sample> log_;
	std::atomic<size_t> log_head_{0}, log_tail_{0};
	std::atomic<uint32_t> log_dropped_{0};
};

}  // namespace p4wrist
