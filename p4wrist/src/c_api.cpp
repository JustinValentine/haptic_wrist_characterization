// C API over p4wrist::Wrist (see include/p4wrist/p4wrist.h).
#include <cmath>
#include <cstring>
#include <string>

#include "p4wrist/p4wrist.h"
#include "p4wrist/wrist.h"

using namespace p4wrist;

struct p4w_wrist {
	std::unique_ptr<Wrist> wrist;
};

namespace {
thread_local std::string last_error;

template <typename F>
int guard(F&& f)
{
	try {
		f();
		return 0;
	} catch (const std::exception& e) {
		last_error = e.what();
	} catch (...) {
		last_error = "unknown error";
	}
	return -1;
}

void copyText(const std::string& text, char* out, int32_t len)
{
	if (!out || len <= 0) return;
	std::snprintf(out, size_t(len), "%s", text.c_str());
}

class CallbackBus : public CanBus {
public:
	CallbackBus(p4w_send_fn send, p4w_recv_fn recv, void* user) : send_(send), recv_(recv), user_(user) {}
	void send(const CanFrame& f) override
	{
		if (send_(user_, f.id, f.data, f.len) != 0) throw BusError("callback bus refused a frame");
	}
	bool receive(CanFrame* f, double) override
	{
		uint32_t id = 0;
		int32_t len = 0;
		uint8_t data[8] = {};
		if (recv_(user_, &id, data, &len) != 1) return false;
		f->id = id;
		f->len = uint8_t(len < 0 ? 0 : (len > 8 ? 8 : len));
		std::memcpy(f->data, data, f->len);
		return true;
	}
	bool synchronous() const override { return true; }

private:
	p4w_send_fn send_;
	p4w_recv_fn recv_;
	void* user_;
};

constexpr double kPi = 3.141592653589793;
}  // namespace

extern "C" {

const char* p4w_last_error(void) { return last_error.c_str(); }

int p4w_abi_version(void) { return P4W_ABI_VERSION; }

void p4w_struct_sizes(int32_t* config, int32_t* command, int32_t* sample, int32_t* zero_cal,
		int32_t* zero_record, int32_t* drive_gains, int32_t* drive_info)
{
	*drive_gains = int32_t(sizeof(p4w_drive_gains));
	*drive_info = int32_t(sizeof(p4w_drive_info));
	*config = int32_t(sizeof(p4w_config));
	*command = int32_t(sizeof(p4w_command));
	*sample = int32_t(sizeof(p4w_sample));
	*zero_cal = int32_t(sizeof(p4w_zero_cal));
	*zero_record = int32_t(sizeof(p4w_zero_record));
}

void p4w_default_config(p4w_config* c)
{
	std::memset(c, 0, sizeof(*c));
	c->node[0] = 6;
	c->node[1] = 7;
	c->node[2] = 8;
	c->controlword_refresh_cycles = 25;  // 50 ms, as the WAM safety board
	// libbarrett-canopen config/wam8g.conf (2026-09-22): rows motors 6-8.
	const double n2 = 8.7456, n1 = 7.5077;
	const double j2mp[3][3] = {{-n2, 0, 0}, {0, n1, n1}, {0, -n1, n1}};
	std::memcpy(c->j2mp, j2mp, sizeof(j2mp));
	for (int i = 0; i < 3; ++i) {
		c->position_sign[i] = 1.0;
		c->torque_sign[i] = 1.0;
		// Wide placeholders: set real limits for your wrist and home pose.
		c->joint_min[i] = -kPi;
		c->joint_max[i] = kPi;
		c->max_joint_velocity[i] = 2.0 * kPi;
		c->max_joint_torque[i] = 2.0;
	}
	c->period_s = 0.002;
	c->velocity_filter_hz = 30.0;
	c->command_timeout_s = 0.1;
	c->brake_s = 0.5;
	c->heartbeat_timeout_s = 1.0;
	c->tpdo_wait_s = 0.0;   // 0: 60% of the period
	c->current_torque_factor = 1.0 / std::sqrt(2.0);
	c->max_motor_permille = 1000;
	c->sync_loss_timeout_ms = 20;
	c->puck_heartbeat_ms = 250;
	c->feedback_timeout_cycles = 5;
	c->overrun_limit_cycles = 10;
	c->telemetry_divider = 50;
	c->realtime_priority = 0;
	c->map_current = 1;
	c->map_torque_demand = 1;
	c->map_drive_targets = 1;
}

p4w_wrist* p4w_open_socketcan(const p4w_config* config, const char* interface)
{
	p4w_wrist* w = nullptr;
	guard([&] {
		std::unique_ptr<p4w_wrist> h(new p4w_wrist);
		h->wrist.reset(new Wrist(*config, openSocketCan(interface ? interface : "can0"), false));
		w = h.release();
	});
	return w;
}

p4w_wrist* p4w_open_callback(const p4w_config* config, p4w_send_fn send, p4w_recv_fn recv, void* user)
{
	p4w_wrist* w = nullptr;
	guard([&] {
		if (!send || !recv) throw std::invalid_argument("send and recv callbacks are required");
		std::unique_ptr<p4w_wrist> h(new p4w_wrist);
		h->wrist.reset(new Wrist(*config, std::unique_ptr<CanBus>(new CallbackBus(send, recv, user)), true));
		w = h.release();
	});
	return w;
}

void p4w_close(p4w_wrist* w)
{
	if (!w) return;
	guard([&] { w->wrist->stop(); });
	delete w;
}

int p4w_connect(p4w_wrist* w, const p4w_zero_record* record, int32_t* kept_zero, char* report,
		int32_t report_len)
{
	return guard([&] {
		std::string text;
		const bool kept = w->wrist->connect(record, &text);
		if (kept_zero) *kept_zero = kept ? 1 : 0;
		copyText(text, report, report_len);
	});
}

int p4w_read_encoders(p4w_wrist* w, int32_t* counts, int32_t* single_turn)
{
	return guard([&] { w->wrist->readEncoders(counts, single_turn); });
}

int p4w_zero(p4w_wrist* w, const p4w_zero_cal* cal, int32_t exact, double* from_home)
{
	return guard([&] {
		const Vec3 d = w->wrist->zero(*cal, exact != 0);
		if (from_home)
			for (int i = 0; i < 3; ++i) from_home[i] = d[i];
	});
}

int p4w_get_zero_record(p4w_wrist* w, p4w_zero_record* record)
{
	return guard([&] { *record = w->wrist->zeroRecord(); });
}

int32_t p4w_single_turn_direction(int32_t position_delta, int32_t single_turn_delta, int32_t counts_per_rev)
{
	return counts_per_rev > 0 ? singleTurnDirection(position_delta, single_turn_delta, counts_per_rev) : 0;
}

int p4w_start(p4w_wrist* w) { return guard([&] { w->wrist->start(); }); }
int p4w_stop(p4w_wrist* w) { return guard([&] { w->wrist->stop(); }); }
int p4w_step(p4w_wrist* w, int32_t cycles) { return guard([&] { w->wrist->step(cycles); }); }
double p4w_now(p4w_wrist* w) { return w->wrist->now(); }
int p4w_wait_until(p4w_wrist* w, double t) { return guard([&] { w->wrist->waitUntil(t); }); }
int p4w_enable(p4w_wrist* w, double timeout_s) { return guard([&] { w->wrist->enable(timeout_s); }); }
int p4w_disable(p4w_wrist* w, double timeout_s) { return guard([&] { w->wrist->disable(timeout_s); }); }
int p4w_clear_fault(p4w_wrist* w) { return guard([&] { w->wrist->clearFault(); }); }

int p4w_set_command(p4w_wrist* w, const p4w_command* command)
{
	return guard([&] { w->wrist->setCommand(*command); });
}

int p4w_get_sample(p4w_wrist* w, p4w_sample* sample)
{
	return guard([&] { *sample = w->wrist->sample(); });
}

int32_t p4w_read_log(p4w_wrist* w, p4w_sample* out, int32_t max, uint32_t* dropped)
{
	int32_t n = -1;
	guard([&] { n = w->wrist->readLog(out, max, dropped); });
	return n;
}

int p4w_fault_text(p4w_wrist* w, char* text, int32_t text_len)
{
	return guard([&] { copyText(w->wrist->faultText(), text, text_len); });
}

int p4w_get_drive_info(p4w_wrist* w, int32_t motor, p4w_drive_info* info)
{
	return guard([&] {
		if (motor < 0 || motor > 2) throw std::invalid_argument("motor must be 0, 1 or 2");
		*info = w->wrist->driveInfo(motor);
	});
}

int p4w_set_drive_gains(p4w_wrist* w, int32_t motor, const p4w_drive_gains* gains, double timeout_s)
{
	return guard([&] { w->wrist->setDriveGains(motor, *gains, timeout_s); });
}

}  // extern "C"
