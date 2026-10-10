#include "p4wrist/wrist.h"

#include <algorithm>
#include <cstdarg>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#endif

namespace p4wrist {

namespace {
constexpr double kTwoPi = 6.283185307179586;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr int kZeroSlipCounts = 16;       // counter vs single-turn agreement, 0.4% of a turn
constexpr size_t kLogCapacity = 1 << 15;  // 65 s at 500 Hz
constexpr int kEnableRefreshCycles = 5;
constexpr double kEnableLimit_s = 3.0;    // the loop's own cap on one enable attempt
constexpr double kDisableWait_s = 0.5;
constexpr double kSdoTimeout_s = 0.2;

std::string format(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
std::string format(const char* fmt, ...)
{
	char text[256];
	va_list args;
	va_start(args, fmt);
	std::vsnprintf(text, sizeof(text), fmt, args);
	va_end(args);
	return text;
}

int16_t le16(const uint8_t* p) { return int16_t(uint16_t(p[0]) | (uint16_t(p[1]) << 8)); }
int32_t le32(const uint8_t* p)
{
	return int32_t(uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24));
}

Mat3 toMat3(const double a[3][3])
{
	Mat3 m;
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j) m.a[i][j] = a[i][j];
	return m;
}

// TPDO3 telemetry entries, in mapping order.
const uint32_t kTelemetryEntries[3] = {
	co::mapEntry(co::OD_AMPLIFIER, 1, 16),       // bus voltage, 0.1 V
	co::mapEntry(co::OD_AMPLIFIER, 2, 16),       // amplifier temperature, C
	co::mapEntry(co::OD_MOTOR_SENSORS, 3, 16),   // motor thermistor
};
}  // namespace

int withinTurn(int counts, int cpr)
{
	int r = counts % cpr;
	if (r >= cpr / 2) r -= cpr;
	if (r < -cpr / 2) r += cpr;
	return r;
}

int singleTurnDirection(int position_delta, int single_turn_delta, int cpr)
{
	const int with = std::abs(withinTurn(single_turn_delta - position_delta, cpr));
	const int against = std::abs(withinTurn(single_turn_delta + position_delta, cpr));
	if (with <= kZeroSlipCounts && against > 4 * kZeroSlipCounts) return 1;
	if (against <= kZeroSlipCounts && with > 4 * kZeroSlipCounts) return -1;
	return 0;
}

Wrist::Wrist(const p4w_config& config, std::unique_ptr<CanBus> bus, bool manual_clock)
	: cfg_(config), bus_(std::move(bus)), manual_(manual_clock), tx_(toMat3(config.j2mp)),
	  log_(kLogCapacity)
{
	if (!(cfg_.period_s >= 0.0005 && cfg_.period_s <= 0.05))
		throw std::invalid_argument("period_s must be within 0.5-50 ms");
	const double period_ms = cfg_.period_s * 1000.0;
	if (std::fabs(period_ms - std::round(period_ms)) > 1e-9)
		throw std::invalid_argument("period_s must be a whole number of ms (0x60C2 holds ms)");
	for (int i = 0; i < 3; ++i) {
		if (cfg_.node[i] < 1 || cfg_.node[i] > 127) throw std::invalid_argument("node ids must be 1-127");
		for (int k = 0; k < i; ++k)
			if (cfg_.node[i] == cfg_.node[k]) throw std::invalid_argument("node ids must differ");
		if (std::fabs(cfg_.position_sign[i]) != 1.0 || std::fabs(cfg_.torque_sign[i]) != 1.0)
			throw std::invalid_argument("position_sign and torque_sign entries must be +1 or -1");
		if (!(cfg_.joint_min[i] < cfg_.joint_max[i])) throw std::invalid_argument("joint_min must be < joint_max");
		if (!(cfg_.max_joint_velocity[i] > 0.0) || !(cfg_.max_joint_torque[i] >= 0.0))
			throw std::invalid_argument("velocity and torque limits must be positive");
		bus_voltage_[i] = amp_temp_[i] = motor_temp_[i] = kNaN;
	}
	if (cfg_.controlword_refresh_cycles < 1 || cfg_.controlword_refresh_cycles > 50)
		throw std::invalid_argument("controlword_refresh_cycles must be 1-50 (the P4 drops out after ~55 SYNCs)");
	if (cfg_.max_motor_permille <= 0 || cfg_.max_motor_permille > 32767)
		throw std::invalid_argument("max_motor_permille must be 1-32767");
	if (cfg_.telemetry_divider < 0 || cfg_.telemetry_divider > 240)
		throw std::invalid_argument("telemetry_divider must be 0-240");
	if (!(cfg_.velocity_filter_hz > 0.0)) throw std::invalid_argument("velocity_filter_hz must be positive");
	if (cfg_.tpdo_wait_s <= 0.0) cfg_.tpdo_wait_s = 0.6 * cfg_.period_s;
	if (cfg_.tpdo_wait_s >= cfg_.period_s) throw std::invalid_argument("tpdo_wait_s must be shorter than period_s");
}

Wrist::~Wrist()
{
	try {
		stop();
	} catch (...) {
	}
}

int Wrist::motorOfNode(int node) const
{
	for (int i = 0; i < 3; ++i)
		if (cfg_.node[i] == node) return i;
	return -1;
}

// --- setup ------------------------------------------------------------------------------

bool Wrist::encodersStill(Master& m, int node, int32_t* counts, int32_t* single_turn)
{
	const int32_t before = m.upload(node, co::OD_POSITION, 0);
	const int32_t st = m.upload(node, co::OD_RAW_POSITION, 1);
	const int32_t after = m.upload(node, co::OD_POSITION, 0);
	if (std::abs(after - before) > kZeroSlipCounts) return false;
	*counts = before + (after - before) / 2;
	*single_turn = st;
	return true;
}

void Wrist::mapPdo(Master& m, int node, uint16_t comm, uint16_t map, uint32_t cob, uint8_t type,
		const std::vector<std::pair<uint32_t, bool>>& entries, std::vector<bool>* mapped)
{
	// CiA 301 procedure: invalidate the COB, clear the map, write entries,
	// set the count, set the type, re-link the COB.
	m.download(node, comm, 1, co::PDO_COB_INVALID | cob, 4);
	m.download(node, map, 0, 0, 1);
	int n = 0;
	for (const auto& entry : entries) {
		try {
			m.download(node, map, uint8_t(n + 1), entry.first, 4);
			++n;
			if (mapped) mapped->push_back(true);
		} catch (const SdoError& e) {
			// An optional entry the drive will not map is left out; anything
			// else (a timeout, a required entry) stops the setup.
			if (entry.second || e.abort_code == 0) throw;
			if (mapped) mapped->push_back(false);
		}
	}
	if (n == 0) return;  // nothing mappable: leave the PDO invalid
	m.download(node, map, 0, uint32_t(n), 1);
	m.download(node, comm, 2, type, 1);
	m.download(node, comm, 1, cob, 4);
}

void Wrist::configurePuck(Master& m, int motor)
{
	const int node = cfg_.node[motor];
	DriveInfo& d = drive_[motor];
	d = DriveInfo();
	m.upload(node, co::OD_DEVICE_TYPE, 0);  // is it there at all?
	d.counts_per_rev = m.upload(node, co::OD_ENCODER, 1);
	d.rated_torque_mnm = m.upload(node, co::OD_RATED_TORQUE, 0);
	d.kt_mnm_per_a = m.upload(node, co::OD_MOTOR, 4);
	d.peak_current_ma = m.upload(node, co::OD_MOTOR, 9);
	// Motor model and loop gains, for the record (and the simulator). Optional:
	// an older firmware without them must not stop the setup.
	const auto optional = [&](uint16_t index, uint8_t sub, double scale) {
		try {
			return m.upload(node, index, sub) * scale;
		} catch (const SdoError& e) {
			if (e.abort_code == 0) throw;
			return kNaN;
		}
	};
	d.cont_current_ma = optional(co::OD_MOTOR, 8, 1.0);
	d.resistance_ohm = optional(co::OD_MOTOR, 5, 0.01);
	d.inductance_mh = optional(co::OD_MOTOR, 6, 0.01);
	// 0x3011:07 is g cm^2 x 2^15; 1 g cm^2 = 1e-7 kg m^2.
	d.rotor_inertia_kgm2 = optional(co::OD_MOTOR, 7, 1e-7 / 32768.0);
	const auto optionalBits = [&](uint16_t index, uint8_t sub, bool* ok) -> int32_t {
		try {
			*ok = true;
			return m.upload(node, index, sub);
		} catch (const SdoError& e) {
			if (e.abort_code == 0) throw;
			*ok = false;
			return 0;
		}
	};
	const auto floatOf = [](int32_t bits) {
		float f;
		std::memcpy(&f, &bits, sizeof(f));
		return double(f);
	};
	bool ok[6];
	d.gains.kp = optionalBits(co::OD_POSITION_GAINS, 1, &ok[0]);
	d.gains.kvff = optionalBits(co::OD_POSITION_GAINS, 2, &ok[1]);
	d.gains.kaff = optionalBits(co::OD_POSITION_GAINS, 3, &ok[2]);
	d.gains.kout = optionalBits(co::OD_POSITION_GAINS, 4, &ok[3]);
	d.gains.kv = floatOf(optionalBits(co::OD_VELOCITY_GAINS, 1, &ok[4]));
	d.gains.ki = floatOf(optionalBits(co::OD_VELOCITY_GAINS, 2, &ok[5]));
	gains_known_[motor] = ok[0] && ok[1] && ok[2] && ok[3] && ok[4] && ok[5];
	gains_changed_[motor] = false;
	if (d.counts_per_rev <= 0)
		throw std::runtime_error(format("puck %d reports %d encoder counts per turn", node, d.counts_per_rev));
	if (d.rated_torque_mnm <= 0.0)
		throw std::runtime_error(format("puck %d reports rated torque %.0f mNm; refusing to scale torque "
				"commands for an unconfigured drive", node, d.rated_torque_mnm));

	// Clear any persisted torque target before torque PDOs are linked.
	m.download(node, co::OD_TARGET_TORQUE, 0, 0, 2);

	// RPDO1: this motor's slot of torque quad B, applied on SYNC.
	std::vector<std::pair<uint32_t, bool>> quad;
	for (int slot = 0; slot < 4; ++slot)
		quad.push_back({slot == motor + 1 ? co::mapEntry(co::OD_TARGET_TORQUE, 0, 16) : co::PDO_MAP_DUMMY16, true});
	mapPdo(m, node, 0x1400, 0x1600, co::COB_QUAD_B, 0, quad, nullptr);

	// RPDO2: controlword + mode, applied on receipt. The P4 (fw 4.3.2) only
	// accepts a mode while Operation Enabled and disables itself after ~55
	// SYNCs without a controlword, so the loop rebroadcasts both every
	// controlword_refresh_cycles.
	mapPdo(m, node, 0x1401, 0x1601, co::COB_CONTROL, 255,
			{{co::mapEntry(co::OD_CONTROLWORD, 0, 16), true}, {co::mapEntry(co::OD_MODE, 0, 8), true}}, nullptr);
	// RPDO3: position target (and torque offset) for the drive's own position
	// loop (CSP), latched on SYNC. Only sent while the drive loop runs.
	std::vector<bool> mapped;
	if (cfg_.map_drive_targets) {
		mapPdo(m, node, 0x1402, 0x1602, co::COB_RPDO3 + node, 0,
				{{co::mapEntry(co::OD_TARGET_POSITION, 0, 32), false},
				 {co::mapEntry(co::OD_TORQUE_OFFSET, 0, 16), false}}, &mapped);
		d.drive_targets_mapped = mapped.size() > 0 && mapped[0];
		d.torque_offset_mapped = d.drive_targets_mapped && mapped.size() > 1 && mapped[1];
		if (!d.drive_targets_mapped)
			m.download(node, 0x1402, 1, co::PDO_COB_INVALID | (co::COB_RPDO3 + node), 4);
	} else {
		m.download(node, 0x1402, 1, co::PDO_COB_INVALID | (co::COB_RPDO3 + node), 4);
	}
	m.download(node, 0x1403, 1, co::PDO_COB_INVALID | (co::COB_RPDO4 + node), 4);

	// TPDO1 on every SYNC: position, and when the drive maps them, q-axis
	// current (what the motor delivers) and the drive's own torque demand.
	std::vector<std::pair<uint32_t, bool>> feedback = {{co::mapEntry(co::OD_POSITION, 0, 32), true}};
	if (cfg_.map_current) feedback.push_back({co::mapEntry(co::OD_CURRENT, 0, 16), false});
	if (cfg_.map_torque_demand) feedback.push_back({co::mapEntry(co::OD_TORQUE_DEMAND, 0, 16), false});
	mapped.clear();
	mapPdo(m, node, 0x1800, 0x1A00, co::COB_TPDO1 + node, 0, feedback, &mapped);
	int offset = 4;
	for (size_t k = 1; k < feedback.size(); ++k) {
		if (!mapped[k]) continue;
		if (feedback[k].first == co::mapEntry(co::OD_CURRENT, 0, 16)) d.current_offset = offset;
		else d.demand_offset = offset;
		offset += 2;
	}
	d.current_mapped = d.current_offset >= 0;
	d.torque_demand_mapped = d.demand_offset >= 0;

	// TPDO2: statusword + mode display (the P4 sends it every 100 ms whatever
	// the type; type 5 is libbarrett-canopen's setting).
	mapPdo(m, node, 0x1801, 0x1A01, co::COB_TPDO2 + node, 5,
			{{co::mapEntry(co::OD_STATUSWORD, 0, 16), true}, {co::mapEntry(co::OD_MODE_DISPLAY, 0, 8), true}},
			nullptr);

	// TPDO3: slow telemetry.
	d.telemetry_mask = 0;
	if (cfg_.telemetry_divider > 0) {
		mapped.clear();
		mapPdo(m, node, 0x1802, 0x1A02, co::COB_TPDO3 + node, uint8_t(cfg_.telemetry_divider),
				{{kTelemetryEntries[0], false}, {kTelemetryEntries[1], false}, {kTelemetryEntries[2], false}},
				&mapped);
		for (size_t k = 0; k < mapped.size(); ++k)
			if (mapped[k]) d.telemetry_mask |= 1u << k;
	} else {
		m.download(node, 0x1802, 1, co::PDO_COB_INVALID | (co::COB_TPDO3 + node), 4);
	}
	m.download(node, 0x1803, 1, co::PDO_COB_INVALID | (co::COB_TPDO4 + node), 4);

	// Heartbeat producer. The period also marks "configured since the last
	// reset" for keeping the zero: a reset restores the saved period.
	m.download(node, co::OD_HB_PRODUCER, 0, uint32_t(cfg_.puck_heartbeat_ms), 2);

	// CST needs both cyclic timing objects (fw 4.3.2).
	const int period_us = int(std::lround(cfg_.period_s * 1e6));
	m.download(node, co::OD_SYNC_PERIOD, 0, uint32_t(period_us), 4);
	m.download(node, co::OD_INTERP_PERIOD, 1, uint32_t(period_us / 1000), 1);
	m.download(node, co::OD_INTERP_PERIOD, 2, 0xFD, 1);  // x 10^-3 s
}

bool Wrist::connect(const p4w_zero_record* record, std::string* report)
{
	if (running_) throw std::logic_error("stop the loop before connecting");
	Master m(*bus_);
	m.drain();

	bool keep = false;
	std::string why = "no zero record";
	if (record && record->valid) {
		keep = true;
		try {
			for (int i = 0; i < 3 && keep; ++i) {
				const int node = cfg_.node[i];
				const int hb = m.upload(node, co::OD_HB_PRODUCER, 0);
				if (hb != cfg_.puck_heartbeat_ms) {
					why = format("puck %d was reset (heartbeat period %d ms)", node, hb);
					keep = false;
					break;
				}
				const int cpr = m.upload(node, co::OD_ENCODER, 1);
				int32_t counts, st;
				if (cpr <= 0 || !encodersStill(m, node, &counts, &st)) {
					why = format("puck %d is moving", node);
					keep = false;
					break;
				}
				// The encoder's direction relative to 0x6064 is not assumed.
				const int counted = counts - record->counts[i];
				const int turned = withinTurn(st - record->single_turn[i], cpr);
				if (std::min(std::abs(withinTurn(counted - turned, cpr)), std::abs(withinTurn(counted + turned, cpr)))
						> kZeroSlipCounts) {
					why = format("puck %d's counter (moved %d counts) disagrees with its single-turn "
							"encoder (moved %d)", node, counted, turned);
					keep = false;
				}
			}
		} catch (const std::exception& e) {
			why = e.what();
			keep = false;
		}
	}

	if (keep) {
		for (int i = 0; i < 3; ++i) m.nmt(co::NMT_PREOP, cfg_.node[i]);
	} else {
		// Reset one puck at a time and wait for its boot-up. Draining first
		// drops stale heartbeats; the boot-up itself is never drained.
		for (int i = 0; i < 3; ++i) {
			m.drain();
			m.nmt(co::NMT_RESET_NODE, cfg_.node[i]);
			if (!m.waitBootUp(cfg_.node[i], 3.0)) {
				throw std::runtime_error(format("puck %d did not boot after a reset: is it powered and "
						"on the bus at 1 Mbit/s?", cfg_.node[i]));
			}
		}
	}
	for (int i = 0; i < 3; ++i) configurePuck(m, i);
	m.drain();

	if (keep) {
		record_ = *record;
		zeroed_ = true;
	} else {
		record_ = p4w_zero_record{};
		zeroed_ = false;
	}
	if (report) *report = keep ? "kept the zero from the previous run (no puck reset since)"
	                           : "pucks reset (" + why + "); the wrist needs zeroing";
	return keep;
}

void Wrist::readEncoders(int32_t counts[3], int32_t single_turn[3])
{
	if (running_) throw std::logic_error("stop the loop before reading encoders over SDO");
	Master m(*bus_);
	for (int i = 0; i < 3; ++i) {
		if (!encodersStill(m, cfg_.node[i], &counts[i], &single_turn[i]))
			throw std::runtime_error(format("puck %d is moving; hold the wrist still", cfg_.node[i]));
	}
}

Vec3 Wrist::zero(const p4w_zero_cal& cal, bool exact)
{
	if (running_) throw std::logic_error("stop the loop before zeroing");
	if (drive_[0].counts_per_rev <= 0) throw std::logic_error("connect before zeroing");
	int32_t counts[3], st[3];
	readEncoders(counts, st);
	const Vec3 home{cal.home[0], cal.home[1], cal.home[2]};
	const Vec3 m_home = tx_.motorAngles(home);
	p4w_zero_record r{};
	Vec3 m_now{};
	for (int i = 0; i < 3; ++i) {
		const int cpr = drive_[i].counts_per_rev;
		const double raw = cfg_.position_sign[i] * counts[i] * kTwoPi / cpr;
		if (exact || !cal.has_single_turn) {
			r.motor_offset[i] = raw - m_home[i];
		} else {
			if (cal.single_turn_dir[i] == 0)
				throw std::runtime_error(format("motor %d has no single-turn direction in the calibration; "
						"zero exactly at home instead", cfg_.node[i]));
			// Motor turn from home, in 0x6064 counts, from the absolute
			// single-turn encoder: right only within half a motor turn of home.
			const int delta = withinTurn(cal.single_turn_dir[i] * (st[i] - cal.single_turn_at_home[i]), cpr);
			r.motor_offset[i] = cfg_.position_sign[i] * double(counts[i] - delta) * kTwoPi / cpr - m_home[i];
		}
		m_now[i] = raw - r.motor_offset[i];
		r.counts[i] = counts[i];
		r.single_turn[i] = st[i];
	}
	r.valid = 1;
	record_ = r;
	zeroed_ = true;
	const Vec3 q = tx_.jointAngles(m_now);
	return {q[0] - home[0], q[1] - home[1], q[2] - home[2]};
}

// --- loop control -------------------------------------------------------------------------

double Wrist::now() const
{
	if (manual_) return double(cycle_) * cfg_.period_s;
	return running_ ? monotonicNow() - t0_ : 0.0;
}

void Wrist::start()
{
	if (running_) throw std::logic_error("loop already running");
	if (drive_[0].counts_per_rev <= 0) throw std::logic_error("connect before starting the loop");
	Master m(*bus_);
	m.drain();
	for (int i = 0; i < 3; ++i) m.nmt(co::NMT_START, cfg_.node[i]);

	cycle_ = 0;
	state_ = P4W_STATE_DISABLED;
	fault_ = P4W_FAULT_NONE;
	fault_motor_ = -1;
	fault_text_.clear();
	emcy_code_ = 0;
	cw_ = co::CW_DISABLE_VOLTAGE;
	mode_byte_ = co::MODE_IDLE;
	control_dirty_ = true;
	have_q_prev_ = false;
	late_cycles_ = 0;
	for (int i = 0; i < 3; ++i) {
		have_feedback_[i] = false;
		missed_[i] = 0;
		statusword_time_[i] = -1.0;
		last_heartbeat_[i] = 0.0;
		have_setpoint_[i] = false;
		current_[i] = kNaN;
	}
	sdo_queue_.clear();
	sdo_next_ = 0;
	sdo_waiting_ = false;
	sync_loss_armed_ = cfg_.sync_loss_timeout_ms <= 0;
	sync_loss_acks_ = 0;
	if (cfg_.sync_loss_timeout_ms > 0) {
		// Written once SYNC runs, from the loop: a drive whose SYNC-loss
		// timer is set before the first SYNC may fault at once.
		for (int i = 0; i < 3; ++i)
			sdo_queue_.push_back({cfg_.node[i], co::OD_SYNC_LOSS_TIMEOUT, 0, uint32_t(cfg_.sync_loss_timeout_ms), 2,
					SDO_ARM_SYNC_LOSS});
	}
	{
		std::lock_guard<std::mutex> lock(sdo_mutex_);
		sdo_requests_.clear();
	}
	sdo_requested_ = sdo_completed_ = 0;
	csp_phase_ = 0;
	for (int i = 0; i < 3; ++i) {
		have_demand_[i] = false;
		torque_demand_[i] = kNaN;
	}
	sdo_idle_ = sdo_queue_.empty();
	request_ = REQ_NONE;
	quit_ = false;
	log_head_ = log_tail_ = 0;
	log_dropped_ = 0;
	{
		std::lock_guard<std::mutex> lock(command_mutex_);
		command_valid_ = false;
	}
	active_valid_ = false;
	running_ = true;
	if (manual_) {
		t0_ = 0.0;
		return;
	}

#ifdef __linux__
	if (cfg_.realtime_priority > 0 && mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
		running_ = false;
		throw std::runtime_error("mlockall failed (raise the memlock limit, or set realtime_priority 0)");
	}
#endif
	t0_ = monotonicNow();
	thread_ = std::thread(&Wrist::threadMain, this);
#ifdef __linux__
	if (cfg_.realtime_priority > 0) {
		sched_param sp{};
		sp.sched_priority = cfg_.realtime_priority;
		const int err = pthread_setschedparam(thread_.native_handle(), SCHED_FIFO, &sp);
		if (err != 0) {
			quit_ = true;
			thread_.join();
			running_ = false;
			throw std::runtime_error(format("could not give the loop SCHED_FIFO priority %d (error %d): run with "
					"CAP_SYS_NICE (e.g. sudo setcap cap_sys_nice+ep on the interpreter) or set "
					"realtime_priority 0", cfg_.realtime_priority, err));
		}
	}
#endif
}

void Wrist::threadMain()
{
	double scheduled = 0.0;
	while (!quit_) {
		sleepUntil(t0_ + scheduled);
		const double t = monotonicNow() - t0_;
		late_s_ = t - scheduled;
		try {
			cycle();
		} catch (const std::exception& e) {
			// Nothing may escape the loop thread; a fault stops the torques.
			raise(P4W_FAULT_BUS, -1, std::string("loop error: ") + e.what(), false, t);
			++cycle_;
		}
		scheduled += cfg_.period_s;
		const double after = monotonicNow() - t0_;
		if (after > scheduled + cfg_.period_s) {
			// Fell more than a cycle behind: skip ahead instead of bursting SYNCs.
			scheduled = std::ceil(after / cfg_.period_s) * cfg_.period_s;
		}
	}
}

void Wrist::step(int cycles)
{
	if (!manual_) throw std::logic_error("step() is for callback buses; the SocketCAN loop runs in a thread");
	if (!running_) throw std::logic_error("start the loop before stepping");
	for (int k = 0; k < cycles; ++k) {
		late_s_ = 0.0;
		cycle();
	}
}

void Wrist::waitUntil(double t)
{
	if (!running_) throw std::logic_error("loop not running");
	if (manual_) {
		const uint64_t target = uint64_t(std::max(0.0, std::ceil(t / cfg_.period_s - 1e-9)));
		while (cycle_ < target) cycle();
	} else {
		sleepUntil(t0_ + t);
	}
}

bool Wrist::waitFor(const std::function<bool()>& done, double timeout_s)
{
	if (manual_) {
		const uint64_t limit = cycle_ + uint64_t(std::ceil(timeout_s / cfg_.period_s));
		while (!done()) {
			if (cycle_ >= limit) return false;
			step(1);
		}
		return true;
	}
	const double deadline = monotonicNow() + timeout_s;
	while (!done()) {
		if (monotonicNow() > deadline) return false;
		sleepUntil(monotonicNow() + 0.001);
	}
	return true;
}

void Wrist::enable(double timeout_s)
{
	if (!running_) throw std::logic_error("start the loop before enabling");
	if (!zeroed_) throw std::runtime_error("zero the wrist before enabling");
	p4w_sample s = sample();
	if (s.fault != P4W_FAULT_NONE) throw std::runtime_error("clear the fault before enabling: " + faultText());
	if (s.wrist_state == P4W_STATE_ACTIVE || s.wrist_state == P4W_STATE_BRAKE) return;
	request_ = REQ_ENABLE;
	const bool finished = waitFor([this] {
		const p4w_sample x = sample();
		return request_ == REQ_NONE && (x.wrist_state == P4W_STATE_ACTIVE || x.fault != P4W_FAULT_NONE);
	}, timeout_s);
	s = sample();
	if (s.wrist_state == P4W_STATE_ACTIVE) return;
	if (!finished) {
		request_ = REQ_DISABLE;
		waitFor([this] { return request_ == REQ_NONE; }, kDisableWait_s);
		throw std::runtime_error("enable timed out");
	}
	throw std::runtime_error("enable failed: " + faultText());
}

void Wrist::disable(double timeout_s)
{
	if (!running_) return;
	request_ = REQ_DISABLE;
	const bool ok = waitFor([this] {
		const int st = sample().wrist_state;
		return request_ == REQ_NONE && (st == P4W_STATE_DISABLED || st == P4W_STATE_FAULT);
	}, timeout_s);
	if (!ok) throw std::runtime_error("drives did not report disabled in time");
}

void Wrist::clearFault()
{
	if (!running_) {
		fault_ = P4W_FAULT_NONE;
		fault_text_.clear();
		return;
	}
	request_ = REQ_CLEAR;
	if (!waitFor([this] { return request_ == REQ_NONE; }, 1.0))
		throw std::runtime_error("the loop did not take the clear request");
	if (sample().fault != P4W_FAULT_NONE) throw std::runtime_error("fault still latched: " + faultText());
}

void Wrist::stop()
{
	if (!running_) {
		restoreGains();
		return;
	}
	try {
		const int st = sample().wrist_state;
		if (st != P4W_STATE_DISABLED && st != P4W_STATE_FAULT) disable(1.0);
		restoreGains();
		if (cfg_.sync_loss_timeout_ms > 0) {
			// Disarm SYNC-loss detection before SYNC stops, so the drives
			// do not fault (and EMCY) as soon as the loop ends.
			request_ = REQ_QUIESCE;
			waitFor([this] { return request_ == REQ_NONE && sdo_idle_; }, 1.0);
		}
	} catch (...) {
	}
	if (manual_) {
		running_ = false;
	} else {
		quit_ = true;
		if (thread_.joinable()) thread_.join();
		running_ = false;
	}
	state_ = P4W_STATE_STOPPED;
	try {
		Master m(*bus_);
		for (int i = 0; i < 3; ++i) m.nmt(co::NMT_PREOP, cfg_.node[i]);
		m.drain();
	} catch (...) {
	}
	p4w_sample s = sample();
	s.wrist_state = P4W_STATE_STOPPED;
	publish(s);
}

// --- drive gains ---------------------------------------------------------------------------------

std::vector<Wrist::RuntimeSdo> Wrist::gainWrites(int node, const p4w_drive_gains& g)
{
	const auto bits = [](double v) {
		const float f = float(v);
		uint32_t u;
		std::memcpy(&u, &f, sizeof(u));
		return u;
	};
	return {{node, co::OD_POSITION_GAINS, 1, uint32_t(g.kp), 2},
	        {node, co::OD_POSITION_GAINS, 2, uint32_t(g.kvff), 2},
	        {node, co::OD_POSITION_GAINS, 3, uint32_t(g.kaff), 2},
	        {node, co::OD_POSITION_GAINS, 4, uint32_t(g.kout), 2},
	        {node, co::OD_VELOCITY_GAINS, 1, bits(g.kv), 4},
	        {node, co::OD_VELOCITY_GAINS, 2, bits(g.ki), 4}};
}

void Wrist::writeGainsDirect(int node, const p4w_drive_gains& g)
{
	Master m(*bus_);
	for (const RuntimeSdo& w : gainWrites(node, g)) m.download(w.node, w.index, w.sub, w.value, w.nbytes);
}

void Wrist::setDriveGains(int motor, const p4w_drive_gains& g, double timeout_s)
{
	if (motor < 0 || motor > 2) throw std::invalid_argument("motor must be 0, 1 or 2");
	for (int32_t v : {g.kp, g.kvff, g.kaff, g.kout})
		if (v < 0 || v > 65535) throw std::invalid_argument("kp, kvff, kaff and kout are 16-bit unsigned");
	if (!std::isfinite(g.kv) || !std::isfinite(g.ki) || g.kv < 0 || g.ki < 0)
		throw std::invalid_argument("kv and ki must be finite and non-negative");
	if (drive_[motor].counts_per_rev <= 0) throw std::logic_error("connect before setting drive gains");
	const int node = cfg_.node[motor];
	gains_changed_[motor] = true;
	if (!running_) {
		writeGainsDirect(node, g);
		return;
	}
	std::vector<RuntimeSdo> writes = gainWrites(node, g);
	for (RuntimeSdo& w : writes) w.caller = true;
	uint64_t target;
	{
		std::lock_guard<std::mutex> lock(sdo_mutex_);
		sdo_requests_.insert(sdo_requests_.end(), writes.begin(), writes.end());
		target = (sdo_requested_ += writes.size());
	}
	if (!waitFor([&] { return sdo_completed_ >= target; }, timeout_s))
		throw std::runtime_error(format("puck %d did not take its gains in time", node));
	if (sample().fault == P4W_FAULT_SDO) throw std::runtime_error(faultText());
}

void Wrist::restoreGains()
{
	for (int i = 0; i < 3; ++i) {
		if (!gains_changed_[i] || !gains_known_[i]) continue;
		try {
			setDriveGains(i, drive_[i].gains, 1.0);
			gains_changed_[i] = false;
		} catch (...) {
		}
	}
}

// --- caller <-> loop -------------------------------------------------------------------------

void Wrist::setCommand(const p4w_command& command)
{
	std::lock_guard<std::mutex> lock(command_mutex_);
	command_ = command;
	if (command_.seq == 0) {
		command_.seq = next_seq_++;
		if (next_seq_ == 0) next_seq_ = 1;
	}
	command_valid_ = true;
}

p4w_sample Wrist::sample() const
{
	std::lock_guard<std::mutex> lock(publish_mutex_);
	return published_.sample;
}

std::string Wrist::faultText() const
{
	std::lock_guard<std::mutex> lock(publish_mutex_);
	return published_.fault_text;
}

void Wrist::publish(const p4w_sample& s)
{
	// The loop never blocks on the caller: if a reader holds the lock, this
	// cycle's snapshot is skipped (the log still has it).
	std::unique_lock<std::mutex> lock(publish_mutex_, std::defer_lock);
	if (manual_) lock.lock();
	else if (!lock.try_lock()) return;
	published_.sample = s;
	std::snprintf(published_.fault_text, sizeof(published_.fault_text), "%s", fault_text_.c_str());
}

int Wrist::readLog(p4w_sample* out, int max, uint32_t* dropped)
{
	int n = 0;
	size_t tail = log_tail_.load(std::memory_order_relaxed);
	const size_t head = log_head_.load(std::memory_order_acquire);
	while (tail != head && n < max) {
		out[n++] = log_[tail];
		tail = (tail + 1) % log_.size();
	}
	log_tail_.store(tail, std::memory_order_release);
	if (dropped) *dropped = log_dropped_.exchange(0);
	return n;
}

// --- the cycle ---------------------------------------------------------------------------------

void Wrist::send(const CanFrame& f)
{
	try {
		bus_->send(f);
	} catch (const std::exception& e) {
		raise(P4W_FAULT_BUS, -1, std::string("CAN send failed: ") + e.what(), false, now());
	}
}

bool Wrist::receive(CanFrame* f, double timeout_s)
{
	try {
		return bus_->receive(f, timeout_s);
	} catch (const std::exception& e) {
		raise(P4W_FAULT_BUS, -1, std::string("CAN receive failed: ") + e.what(), false, now());
		return false;
	}
}

void Wrist::sendControl(uint16_t cw, int8_t mode)
{
	CanFrame f;
	f.id = co::COB_CONTROL;
	f.len = 3;
	f.data[0] = cw & 0xFF;
	f.data[1] = cw >> 8;
	f.data[2] = uint8_t(mode);
	send(f);
	last_control_cycle_ = cycle_;
	control_dirty_ = false;
}

void Wrist::raise(int code, int motor, const std::string& text, bool brake, double t)
{
	if (fault_ != P4W_FAULT_NONE) return;  // the first fault is the one reported
	fault_ = code;
	fault_motor_ = motor;
	fault_text_ = text;
	if (brake && (state_ == P4W_STATE_ACTIVE || state_ == P4W_STATE_BRAKE)) {
		state_ = P4W_STATE_BRAKING;
		brake_until_ = t + cfg_.brake_s;
	} else if (state_ != P4W_STATE_STOPPED) {
		state_ = P4W_STATE_FAULT;
	}
	state_since_ = t;
	control_dirty_ = true;
}

void Wrist::dispatch(const CanFrame& f, bool collecting, uint32_t* got)
{
	const double t = now();
	const uint32_t base = f.id & 0x780;
	const int motor = motorOfNode(int(f.id & 0x7F));
	if (motor < 0) return;
	const int node = cfg_.node[motor];
	switch (base) {
	case co::COB_TPDO1:
		// A position left over from an earlier SYNC would put this motor a
		// cycle behind the others; only positions answering this SYNC count.
		if (!collecting || f.len < 4) return;
		counts_[motor] = le32(f.data);
		if (drive_[motor].current_offset >= 0 && f.len >= drive_[motor].current_offset + 2)
			current_[motor] = le16(f.data + drive_[motor].current_offset) / 1000.0
					* drive_[motor].peak_current_ma / 1000.0;
		if (drive_[motor].demand_offset >= 0 && f.len >= drive_[motor].demand_offset + 2) {
			torque_demand_[motor] = le16(f.data + drive_[motor].demand_offset) / 1000.0
					* drive_[motor].rated_torque_mnm / 1000.0;
			have_demand_[motor] = true;
		}
		*got |= 1u << motor;
		return;
	case co::COB_TPDO2:
		if (f.len < 3) return;
		statusword_[motor] = uint16_t(le16(f.data));
		mode_display_[motor] = int8_t(f.data[2]);
		statusword_time_[motor] = t;
		return;
	case co::COB_TPDO3: {
		int offset = 0;
		const uint32_t mask = drive_[motor].telemetry_mask;
		for (int k = 0; k < 3; ++k) {
			if (!(mask & (1u << k))) continue;
			if (f.len < offset + 2) return;
			const int16_t v = le16(f.data + offset);
			offset += 2;
			if (k == 0) bus_voltage_[motor] = uint16_t(v) / 10.0;
			else if (k == 1) amp_temp_[motor] = v;
			else motor_temp_[motor] = v;
		}
		return;
	}
	case 0x080:  // EMCY (0x080 + node)
		if (f.len >= 2) {
			const uint16_t code = uint16_t(le16(f.data));
			if (code == 0) return;
			emcy_code_ = code;
			if (!co::emcyIsWarning(code))
				raise(P4W_FAULT_DRIVE, motor, format("puck %d: EMCY 0x%04X %s", node, code, co::emcyText(code)),
						false, t);
		}
		return;
	case co::COB_HEARTBEAT:
		if (f.len < 1) return;
		if (f.data[0] == co::HB_BOOT_UP) {
			zeroed_ = false;
			record_.valid = 0;
			raise(P4W_FAULT_PUCK_RESET, motor, format("puck %d reset (boot-up seen): its position counter "
					"restarted, so the zero is lost", node), false, t);
		} else {
			last_heartbeat_[motor] = t;
		}
		return;
	case co::COB_SDO_TX:
		if (sdo_waiting_ && sdo_next_ < sdo_queue_.size()) {
			const RuntimeSdo& r = sdo_queue_[sdo_next_];
			if (r.node != node || !co::sdoMatches(f, r.index, r.sub)) return;
			if (f.data[0] == 0x60) {
				if (r.kind == SDO_ARM_SYNC_LOSS && ++sync_loss_acks_ >= 3) sync_loss_armed_ = true;
			} else {
				raise(P4W_FAULT_SDO, motor, format("puck %d refused 0x%04X:%02X (SDO reply 0x%02X)", node, r.index,
						r.sub, f.data[0]), false, t);
			}
			if (r.caller) ++sdo_completed_;
			sdo_waiting_ = false;
			++sdo_next_;
		}
		return;
	default:
		return;
	}
}

bool Wrist::drivesAll(bool (*pred)(uint16_t), double since) const
{
	for (int i = 0; i < 3; ++i)
		if (statusword_time_[i] <= since || !pred(statusword_[i])) return false;
	return true;
}

void Wrist::supervise(double t)
{
	for (int i = 0; i < 3; ++i) {
		if (missed_[i] > cfg_.feedback_timeout_cycles)
			raise(P4W_FAULT_FEEDBACK_LOST, i, format("no position from puck %d for %d cycles", cfg_.node[i],
					missed_[i]), false, t);
		if (t - last_heartbeat_[i] > cfg_.heartbeat_timeout_s)
			raise(P4W_FAULT_HEARTBEAT_LOST, i, format("no heartbeat from puck %d for %.2f s", cfg_.node[i],
					t - last_heartbeat_[i]), false, t);
	}
	if (cfg_.overrun_limit_cycles > 0 && late_cycles_ >= cfg_.overrun_limit_cycles)
		raise(P4W_FAULT_OVERRUN, -1, format("the loop was late %d cycles in a row (last %.2f ms)", late_cycles_,
				late_s_ * 1000.0), true, t);

	const bool torque_on = state_ == P4W_STATE_ACTIVE || state_ == P4W_STATE_BRAKE;
	if (!torque_on) return;
	for (int i = 0; i < 3; ++i) {
		// Statuswords arrive every ~100 ms; only ones sent after this state began count.
		if (statusword_time_[i] > state_since_ && !co::swOperationEnabled(statusword_[i])) {
			if (co::swFault(statusword_[i]))
				raise(P4W_FAULT_DRIVE, i, format("puck %d faulted (statusword 0x%04X, last EMCY 0x%04X %s)",
						cfg_.node[i], statusword_[i], emcy_code_, co::emcyText(uint16_t(emcy_code_))), false, t);
			else
				raise(P4W_FAULT_DRIVE_DROPPED, i, format("puck %d left Operation Enabled (statusword 0x%04X)",
						cfg_.node[i], statusword_[i]), false, t);
		}
	}
	if (csp_phase_ == 2) {
		for (int i = 0; i < 3; ++i) {
			if (statusword_time_[i] > csp_since_ && co::swOperationEnabled(statusword_[i])
					&& mode_display_[i] != co::MODE_CSP)
				raise(P4W_FAULT_DRIVE_LOOP, i, format("puck %d did not enter its position loop (mode display %d)",
						cfg_.node[i], mode_display_[i]), true, t);
		}
	}
	if (!zeroed_) return;
	static const char* names[3] = {"PS", "FE", "UR"};
	for (int j = 0; j < 3; ++j) {
		if (q_[j] < cfg_.joint_min[j] || q_[j] > cfg_.joint_max[j])
			raise(P4W_FAULT_POSITION_LIMIT, j, format("%s at %.1f deg, outside [%.1f, %.1f] deg", names[j],
					q_[j] * 360.0 / kTwoPi, cfg_.joint_min[j] * 360.0 / kTwoPi, cfg_.joint_max[j] * 360.0 / kTwoPi),
					true, t);
		if (std::fabs(qd_[j]) > cfg_.max_joint_velocity[j])
			raise(P4W_FAULT_VELOCITY_LIMIT, j, format("%s at %.0f deg/s, above %.0f deg/s", names[j],
					qd_[j] * 360.0 / kTwoPi, cfg_.max_joint_velocity[j] * 360.0 / kTwoPi), true, t);
	}
	if (t - last_command_time_ > cfg_.command_timeout_s)
		raise(P4W_FAULT_HOST_TIMEOUT, -1, format("no new command for %.0f ms", (t - last_command_time_) * 1000.0),
				true, t);
}

void Wrist::runStateMachine(double t)
{
	const int request = request_.load();
	auto enter = [&](int s) {
		state_ = s;
		state_since_ = t;
		control_dirty_ = true;
	};

	if (request == REQ_CLEAR) {
		if (state_ == P4W_STATE_FAULT || state_ == P4W_STATE_DISABLED) {
			fault_ = P4W_FAULT_NONE;
			fault_motor_ = -1;
			fault_text_.clear();
			if (state_ == P4W_STATE_FAULT) enter(P4W_STATE_DISABLED);
			request_ = REQ_NONE;
		} else if (state_ != P4W_STATE_BRAKING) {
			request_ = REQ_NONE;  // nothing to clear while running normally
		}
	} else if (request == REQ_DISABLE) {
		if (state_ == P4W_STATE_ACTIVE || state_ == P4W_STATE_BRAKE || state_ == P4W_STATE_ENABLING)
			enter(P4W_STATE_DISABLING);
		if (state_ != P4W_STATE_BRAKING) request_ = REQ_NONE;
	} else if (request == REQ_QUIESCE) {
		if (cfg_.sync_loss_timeout_ms > 0) {
			for (int i = 0; i < 3; ++i) sdo_queue_.push_back({cfg_.node[i], co::OD_SYNC_LOSS_TIMEOUT, 0, 0, 2});
			sdo_idle_ = false;
		}
		request_ = REQ_NONE;
	} else if (request == REQ_ENABLE) {
		if (state_ == P4W_STATE_DISABLED && fault_ == P4W_FAULT_NONE) {
			bool inside = true;
			for (int j = 0; j < 3; ++j)
				if (q_[j] < cfg_.joint_min[j] || q_[j] > cfg_.joint_max[j]) inside = false;
			if (!have_q_prev_) {
				raise(P4W_FAULT_FEEDBACK_LOST, -1, "no position feedback yet", false, t);
			} else if (!inside) {
				raise(P4W_FAULT_POSITION_LIMIT, -1, format("refusing to enable outside the joint limits "
						"(PS %.1f, FE %.1f, UR %.1f deg); move the wrist back by hand", q_[0] * 360.0 / kTwoPi,
						q_[1] * 360.0 / kTwoPi, q_[2] * 360.0 / kTwoPi), false, t);
			} else {
				enter(P4W_STATE_ENABLING);
				bool any_fault = false;
				for (int i = 0; i < 3; ++i) any_fault |= co::swFault(statusword_[i]);
				enable_phase_ = any_fault ? EN_FAULT_RESET : EN_SHUTDOWN;
				phase_since_ = t;
				enable_deadline_ = t + kEnableLimit_s;
			}
		}
		request_ = REQ_NONE;
	}

	switch (state_) {
	case P4W_STATE_ENABLING: {
		if (t > enable_deadline_) {
			std::string detail;
			for (int i = 0; i < 3; ++i)
				detail += format(" %d:0x%04X", cfg_.node[i], statusword_[i]);
			raise(P4W_FAULT_ENABLE_TIMEOUT, -1, format("drives did not enable (phase %d, statuswords%s, last EMCY "
					"0x%04X %s). Is the motor supply on?", enable_phase_, detail.c_str(), emcy_code_,
					co::emcyText(uint16_t(emcy_code_))), false, t);
			break;
		}
		if (enable_phase_ == EN_FAULT_RESET) {
			if (drivesAll([](uint16_t sw) { return !co::swFault(sw); }, phase_since_)) {
				enable_phase_ = EN_SHUTDOWN;
				phase_since_ = t;
				control_dirty_ = true;
			}
		} else if (enable_phase_ == EN_SHUTDOWN) {
			if (drivesAll(co::swReadyToSwitchOn, phase_since_) && sync_loss_armed_) {
				enable_phase_ = EN_OPERATION;
				phase_since_ = t;
				control_dirty_ = true;
			}
		} else {
			bool cst = true;
			for (int i = 0; i < 3; ++i) cst &= mode_display_[i] == co::MODE_CST;
			if (drivesAll(co::swOperationEnabled, phase_since_) && cst) {
				enter(P4W_STATE_ACTIVE);
				last_command_time_ = t;
				got_command_since_enable_ = false;
				for (int j = 0; j < 3; ++j) have_setpoint_[j] = false;
			}
		}
		break;
	}
	case P4W_STATE_ACTIVE:
		if (got_command_since_enable_ && active_command_.brake) enter(P4W_STATE_BRAKE);
		break;
	case P4W_STATE_BRAKE:
		if (!active_command_.brake) {
			enter(P4W_STATE_ACTIVE);
			for (int j = 0; j < 3; ++j) have_setpoint_[j] = false;
		}
		break;
	case P4W_STATE_BRAKING:
		if (t >= brake_until_) enter(P4W_STATE_FAULT);
		break;
	case P4W_STATE_DISABLING:
		if (drivesAll(co::swSwitchOnDisabled, state_since_) || t - state_since_ > kDisableWait_s)
			enter(fault_ != P4W_FAULT_NONE ? P4W_STATE_FAULT : P4W_STATE_DISABLED);
		break;
	default:
		break;
	}

	updateDriveLoop(t);

	// What the drives should be doing.
	uint16_t cw = co::CW_DISABLE_VOLTAGE;
	int8_t mode = co::MODE_IDLE;
	switch (state_) {
	case P4W_STATE_ENABLING:
		cw = enable_phase_ == EN_FAULT_RESET ? co::CW_FAULT_RESET
		   : enable_phase_ == EN_SHUTDOWN ? co::CW_SHUTDOWN : co::CW_ENABLE_OPERATION;
		mode = enable_phase_ == EN_OPERATION ? co::MODE_CST : co::MODE_IDLE;
		break;
	case P4W_STATE_ACTIVE:
		cw = co::CW_ENABLE_OPERATION;
		mode = csp_phase_ == 2 ? co::MODE_CSP : co::MODE_CST;
		break;
	case P4W_STATE_BRAKE:
	case P4W_STATE_BRAKING:
		cw = co::CW_ENABLE_OPERATION;  // enabled, phases grounded
		mode = co::MODE_IDLE;
		break;
	default:
		break;
	}
	if (cw != cw_ || mode != mode_byte_) control_dirty_ = true;
	cw_ = cw;
	mode_byte_ = mode;
}

void Wrist::updateDriveLoop(double t)
{
	const bool want = state_ == P4W_STATE_ACTIVE && got_command_since_enable_ && active_command_.drive_loop;
	if (!want) {
		if (csp_phase_ != 0) {
			csp_phase_ = 0;
			control_dirty_ = true;
			for (int j = 0; j < 3; ++j) have_setpoint_[j] = false;   // the host PD takes over from here
		}
		return;
	}
	for (int i = 0; i < 3; ++i) {
		if (!drive_[i].drive_targets_mapped) {
			raise(P4W_FAULT_DRIVE_LOOP, i, format("drive loop requested, but puck %d did not accept a "
					"position target mapping (0x607A)", cfg_.node[i]), true, t);
			return;
		}
	}
	if (csp_phase_ == 0) {
		// Arm: stream targets equal to the current positions while still in CST,
		// so the drive's buffered target is where the motor is when CSP engages.
		csp_phase_ = 1;
		csp_arm_cycles_ = 0;
		for (int j = 0; j < 3; ++j) have_setpoint_[j] = false;
	} else if (csp_phase_ == 1 && ++csp_arm_cycles_ >= 2) {
		csp_phase_ = 2;
		csp_since_ = t;
		control_dirty_ = true;
	}
}

double Wrist::advanceSetpoint(int j, const p4w_joint_command& c)
{
	if (!have_setpoint_[j]) {
		setpoint_[j] = q_[j];
		have_setpoint_[j] = true;
	} else {
		setpoint_[j] = held_setpoint_[j];
	}
	if (std::isfinite(c.position)) setpoint_[j] = c.position;
	setpoint_velocity_[j] = c.velocity;
	// Advance after use, so the error compares setpoint and position at the
	// same instant (moteus semantics for a NaN position).
	held_setpoint_[j] = setpoint_[j] + (std::isfinite(c.position) ? 0.0 : c.velocity * cfg_.period_s);
	return setpoint_[j];
}

void Wrist::control(double t, Vec3* tau_joint)
{
	(void)t;
	*tau_joint = Vec3{};
	for (int j = 0; j < 3; ++j) {
		setpoint_[j] = kNaN;
		setpoint_velocity_[j] = 0.0;
		joint_mode_[j] = 0;
	}
	offset_nm_ = Vec3{};
	if (state_ != P4W_STATE_ACTIVE) {
		for (int j = 0; j < 3; ++j) have_setpoint_[j] = false;
		return;
	}
	if (csp_phase_ >= 1) {
		// The drives render the joints: send them motor targets from the joint
		// setpoints, and each joint's feedforward as torque offsets.
		Vec3 ff{};
		for (int j = 0; j < 3; ++j) {
			const p4w_joint_command& c = active_command_.joint[j];
			advanceSetpoint(j, c);
			ff[j] = std::isfinite(c.torque) ? c.torque : 0.0;
			joint_mode_[j] = P4W_MODE_DRIVE;
		}
		Vec3 cap{};
		for (int i = 0; i < 3; ++i) cap[i] = cfg_.max_motor_permille * drive_[i].rated_torque_mnm / 1e6;
		offset_nm_ = tx_.saturate(tx_.motorTorques(ff), cap);
		const Vec3 m = tx_.motorAngles(setpoint_);
		for (int i = 0; i < 3; ++i) {
			if (csp_phase_ == 1) {
				motor_target_[i] = counts_[i];
			} else {
				const double raw = m[i] + record_.motor_offset[i];
				motor_target_[i] = int32_t(std::lround(cfg_.position_sign[i] * raw * drive_[i].counts_per_rev / kTwoPi));
			}
		}
		return;
	}
	for (int j = 0; j < 3; ++j) {
		p4w_joint_command c{};
		c.position = kNaN;
		c.mode = P4W_MODE_POSITION;
		c.max_torque = kNaN;
		if (got_command_since_enable_) c = active_command_.joint[j];
		// Zero torque until the first command after enabling.
		double tau = 0.0;
		if (c.mode == P4W_MODE_STAY_WITHIN) {
			have_setpoint_[j] = false;
			if (std::isfinite(c.upper) && q_[j] > c.upper)
				tau = c.kp * (c.upper - q_[j]) - c.kd * qd_[j] + c.torque;
			else if (std::isfinite(c.lower) && q_[j] < c.lower)
				tau = c.kp * (c.lower - q_[j]) - c.kd * qd_[j] + c.torque;
			else
				tau = c.torque;
		} else {
			advanceSetpoint(j, c);
			tau = c.kp * (setpoint_[j] - q_[j]) + c.kd * (c.velocity - qd_[j]) + c.torque;
			c.mode = P4W_MODE_POSITION;
		}
		joint_mode_[j] = c.mode;
		double cap = cfg_.max_joint_torque[j];
		if (std::isfinite(c.max_torque)) cap = std::min(cap, std::fabs(c.max_torque));
		if (!std::isfinite(tau)) tau = 0.0;
		(*tau_joint)[j] = std::max(-cap, std::min(cap, tau));
	}
}

void Wrist::sendTorques(const Vec3& tau_joint)
{
	Vec3 cap{};
	for (int i = 0; i < 3; ++i) cap[i] = cfg_.max_motor_permille * drive_[i].rated_torque_mnm / 1e6;
	const Vec3 tau_m = tx_.saturate(tx_.motorTorques(tau_joint), cap);
	tau_cmd_ = tx_.jointTorques(tau_m);
	CanFrame f;
	f.id = co::COB_QUAD_B;
	f.len = 8;
	for (int i = 0; i < 3; ++i) {
		const double pm = cfg_.torque_sign[i] * tau_m[i] * 1e6 / drive_[i].rated_torque_mnm;
		permille_[i] = int32_t(std::lround(std::max(-double(cfg_.max_motor_permille),
				std::min(double(cfg_.max_motor_permille), pm))));
		f.data[2 * (i + 1)] = uint8_t(permille_[i] & 0xFF);
		f.data[2 * (i + 1) + 1] = uint8_t((permille_[i] >> 8) & 0xFF);
	}
	send(f);
}

void Wrist::sendDriveTargets()
{
	if (csp_phase_ < 1) return;
	for (int i = 0; i < 3; ++i) {
		CanFrame f;
		f.id = co::COB_RPDO3 + cfg_.node[i];
		f.len = drive_[i].torque_offset_mapped ? 6 : 4;
		const uint32_t target = uint32_t(motor_target_[i]);
		for (int k = 0; k < 4; ++k) f.data[k] = uint8_t(target >> (8 * k));
		if (drive_[i].torque_offset_mapped) {
			const double pm = cfg_.torque_sign[i] * offset_nm_[i] * 1e6 / drive_[i].rated_torque_mnm;
			const int32_t v = int32_t(std::lround(std::max(-double(cfg_.max_motor_permille),
					std::min(double(cfg_.max_motor_permille), pm))));
			f.data[4] = uint8_t(v & 0xFF);
			f.data[5] = uint8_t((v >> 8) & 0xFF);
		}
		send(f);
	}
}

void Wrist::cycle()
{
	const double t = now();
	const double dt = cycle_ == 0 ? cfg_.period_s : std::max(0.5 * cfg_.period_s, std::min(3.0 * cfg_.period_s,
			t - t_sync_prev_));
	late_cycles_ = late_s_ > cfg_.period_s * 0.5 ? late_cycles_ + 1 : 0;

	// 1. Whatever arrived since the last cycle: statuswords, telemetry, EMCY,
	//    heartbeats, SDO replies, and late positions (dropped).
	CanFrame f;
	uint32_t got = 0;
	while (receive(&f, 0.0)) dispatch(f, false, &got);

	// 2. SYNC: the pucks apply the torques sent last cycle and send positions.
	CanFrame sync_frame;
	sync_frame.id = co::COB_SYNC;
	sync_frame.len = 0;
	send(sync_frame);
	const int hb_cycles = std::max(1, int(std::lround(0.25 / cfg_.period_s)));
	if (cycle_ % uint64_t(hb_cycles) == 0) {
		CanFrame hb;
		hb.id = co::COB_HOST_HEARTBEAT;
		hb.len = 1;
		hb.data[0] = co::HB_OPERATIONAL;  // the host is the NMT master
		send(hb);
	}

	// 3. Collect this SYNC's positions.
	const double deadline = monotonicNow() + cfg_.tpdo_wait_s;
	while (got != 7u) {
		const double remaining = manual_ ? 0.0 : deadline - monotonicNow();
		if (!manual_ && remaining <= 0.0) break;
		if (!receive(&f, remaining)) {
			if (manual_ || fault_ == P4W_FAULT_BUS) break;
			continue;
		}
		dispatch(f, true, &got);
	}
	missed_mask_ = 7u & ~got;
	for (int i = 0; i < 3; ++i) {
		if (got & (1u << i)) {
			missed_[i] = 0;
			have_feedback_[i] = true;
		} else {
			++missed_[i];
		}
	}

	// 4. Joint state.
	if (have_feedback_[0] && have_feedback_[1] && have_feedback_[2]) {
		for (int i = 0; i < 3; ++i)
			motor_angle_[i] = cfg_.position_sign[i] * counts_[i] * kTwoPi / drive_[i].counts_per_rev
					- record_.motor_offset[i];
		q_ = tx_.jointAngles(motor_angle_);
		const double alpha = 1.0 - std::exp(-kTwoPi * cfg_.velocity_filter_hz * dt);
		for (int j = 0; j < 3; ++j) {
			const double raw = have_q_prev_ ? (q_[j] - q_prev_[j]) / dt : 0.0;
			qd_[j] += alpha * (raw - qd_[j]);
		}
		q_prev_ = q_;
		have_q_prev_ = true;
	}
	t_sync_prev_ = t;

	// 5. Runtime SDO writes (one at a time, never waited on).
	{
		std::unique_lock<std::mutex> lock(sdo_mutex_, std::defer_lock);
		if (manual_) lock.lock();
		if ((lock.owns_lock() || lock.try_lock()) && !sdo_requests_.empty()) {
			sdo_queue_.insert(sdo_queue_.end(), sdo_requests_.begin(), sdo_requests_.end());
			sdo_requests_.clear();
			sdo_idle_ = false;
		}
	}
	if (sdo_waiting_ && t - sdo_sent_at_ > kSdoTimeout_s) {
		const RuntimeSdo& r = sdo_queue_[sdo_next_];
		raise(P4W_FAULT_SDO, motorOfNode(r.node), format("puck %d did not answer writing 0x%04X:%02X", r.node,
				r.index, r.sub), false, t);
		if (r.caller) ++sdo_completed_;
		sdo_waiting_ = false;
		++sdo_next_;
	}
	if (!sdo_waiting_ && sdo_next_ < sdo_queue_.size() && cycle_ >= 3) {
		const RuntimeSdo& r = sdo_queue_[sdo_next_];
		send(co::sdoDownloadRequest(r.node, r.index, r.sub, r.value, r.nbytes));
		sdo_waiting_ = true;
		sdo_sent_at_ = t;
	}
	if (!sdo_waiting_ && sdo_next_ >= sdo_queue_.size()) sdo_idle_ = true;

	// 6. The caller's latest command.
	{
		std::unique_lock<std::mutex> lock(command_mutex_, std::defer_lock);
		if (manual_) lock.lock();
		if (lock.owns_lock() || lock.try_lock()) {
			if (command_valid_ && (!active_valid_ || command_.seq != active_seq_)) {
				active_command_ = command_;
				active_seq_ = command_.seq;
				active_valid_ = true;
				last_command_time_ = t;
				if (state_ == P4W_STATE_ACTIVE || state_ == P4W_STATE_BRAKE) got_command_since_enable_ = true;
			}
		}
	}

	// 7. Supervision, state machine, control.
	if (state_ != P4W_STATE_STOPPED) supervise(t);
	runStateMachine(t);
	Vec3 tau_joint{};
	control(t, &tau_joint);

	// 8. Commands out: torques and drive targets (latched at the next SYNC), the controlword.
	sendTorques(tau_joint);
	sendDriveTargets();
	const int refresh = state_ == P4W_STATE_ENABLING ? kEnableRefreshCycles : cfg_.controlword_refresh_cycles;
	if (control_dirty_ || cycle_ - last_control_cycle_ >= uint64_t(refresh)) sendControl(cw_, mode_byte_);

	// 9. Publish and log.
	p4w_sample s{};
	s.cycle = cycle_;
	s.time = t;
	const Vec3 tau_m_meas = [&] {
		Vec3 v{};
		for (int i = 0; i < 3; ++i)
			v[i] = cfg_.torque_sign[i] * current_[i] * drive_[i].kt_mnm_per_a / 1000.0 * cfg_.current_torque_factor;
		return v;
	}();
	const Vec3 tau_meas = tx_.jointTorques(tau_m_meas);
	for (int k = 0; k < 3; ++k) {
		s.q[k] = q_[k];
		s.qd[k] = qd_[k];
		s.setpoint[k] = setpoint_[k];
		s.setpoint_velocity[k] = setpoint_velocity_[k];
		s.tau_cmd[k] = state_ != P4W_STATE_ACTIVE ? 0.0 : (csp_phase_ >= 1 ? kNaN : tau_cmd_[k]);
		s.motor_target[k] = motor_target_[k];
		s.tau_meas[k] = tau_meas[k];
		s.motor_angle[k] = motor_angle_[k];
		s.current[k] = current_[k];
		s.bus_voltage[k] = bus_voltage_[k];
		s.amp_temp[k] = amp_temp_[k];
		s.motor_temp[k] = motor_temp_[k];
		s.counts[k] = counts_[k];
		s.permille[k] = permille_[k];
		s.statusword[k] = statusword_[k];
		s.mode_display[k] = mode_display_[k];
		s.joint_mode[k] = joint_mode_[k];
	}
	s.command_t_host = active_valid_ ? active_command_.t_host : kNaN;
	s.command_seq = active_valid_ ? active_seq_ : 0;
	s.late_s = late_s_;
	s.wrist_state = state_;
	s.fault = fault_;
	s.fault_node = fault_motor_ >= 0 && fault_ != P4W_FAULT_POSITION_LIMIT && fault_ != P4W_FAULT_VELOCITY_LIMIT
			? cfg_.node[fault_motor_] : 0;
	s.emcy_code = emcy_code_;
	s.missed_feedback = missed_mask_;
	s.zeroed = zeroed_ ? 1 : 0;
	s.drive_loop = csp_phase_ == 2 ? 1 : 0;
	{
		bool all = true;
		Vec3 demand{};
		for (int i = 0; i < 3; ++i) {
			all = all && have_demand_[i];
			demand[i] = cfg_.torque_sign[i] * torque_demand_[i];
		}
		const Vec3 tau_demand = tx_.jointTorques(demand);
		for (int k = 0; k < 3; ++k) s.tau_demand[k] = all ? tau_demand[k] : kNaN;
	}
	publish(s);
	const size_t head = log_head_.load(std::memory_order_relaxed);
	const size_t next = (head + 1) % log_.size();
	if (next == log_tail_.load(std::memory_order_acquire)) {
		log_dropped_.fetch_add(1);
	} else {
		log_[head] = s;
		log_head_.store(next, std::memory_order_release);
	}
	++cycle_;
}

}  // namespace p4wrist
