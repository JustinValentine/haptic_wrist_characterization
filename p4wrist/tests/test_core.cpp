// Unit tests for the parts that need no bus. The loop, setup and supervision
// are tested end to end against emulated pucks from Python (tests/test_p4wrist.py
// in the characterization repo).
#include <cmath>
#include <cstdio>
#include <cstring>

#include "p4wrist/canopen.h"
#include "p4wrist/p4wrist.h"
#include "p4wrist/transmission.h"
#include "p4wrist/wrist.h"

using namespace p4wrist;

static int failures = 0;
#define CHECK(cond)                                                          \
	do {                                                                     \
		if (!(cond)) {                                                       \
			std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
			++failures;                                                      \
		}                                                                    \
	} while (0)
#define CHECK_NEAR(a, b, tol) CHECK(std::fabs((a) - (b)) <= (tol))

static Transmission wristTransmission()
{
	p4w_config c;
	p4w_default_config(&c);
	Mat3 m;
	std::memcpy(m.a, c.j2mp, sizeof(m.a));
	return Transmission(m);
}

static void testTransmission()
{
	const Transmission tx = wristTransmission();
	const double n1 = 7.5077, n2 = 8.7456;
	const Vec3 q{0.3, -0.2, 0.1};
	const Vec3 m = tx.motorAngles(q);
	CHECK_NEAR(m[0], -n2 * 0.3, 1e-12);
	CHECK_NEAR(m[1], n1 * (-0.2 + 0.1), 1e-12);
	CHECK_NEAR(m[2], n1 * (0.2 + 0.1), 1e-12);
	const Vec3 back = tx.jointAngles(m);
	for (int i = 0; i < 3; ++i) CHECK_NEAR(back[i], q[i], 1e-12);
	// FE is the difference of the differential motors, UR the sum.
	CHECK_NEAR(back[1], (m[1] - m[2]) / (2 * n1), 1e-12);
	CHECK_NEAR(back[2], (m[1] + m[2]) / (2 * n1), 1e-12);

	// Virtual work: the motor torques for a joint torque do the same work.
	const Vec3 tau_q{0.5, -1.0, 2.0};
	const Vec3 tau_m = tx.motorTorques(tau_q);
	CHECK_NEAR(tau_m[0], -0.5 / n2, 1e-12);
	CHECK_NEAR(tau_m[1], (-1.0 + 2.0) / (2 * n1), 1e-12);
	CHECK_NEAR(tau_m[2], (1.0 + 2.0) / (2 * n1), 1e-12);
	const Vec3 dq{0.01, 0.02, -0.03};
	const Vec3 dm = tx.motorAngles(dq);
	double wq = 0, wm = 0;
	for (int i = 0; i < 3; ++i) {
		wq += tau_q[i] * dq[i];
		wm += tau_m[i] * dm[i];
	}
	CHECK_NEAR(wq, wm, 1e-12);
	const Vec3 tq = tx.jointTorques(tau_m);
	for (int i = 0; i < 3; ++i) CHECK_NEAR(tq[i], tau_q[i], 1e-12);

	// Equal motor-space stiffness gives a diagonal joint stiffness.
	// K_q = j2mp^T k j2mp: FE/UR cross terms cancel.
	const Mat3 k_q = tx.j2mp().transpose() * tx.j2mp();
	CHECK_NEAR(k_q.a[1][2], 0.0, 1e-12);
	CHECK_NEAR(k_q.a[0][1], 0.0, 1e-12);
	CHECK_NEAR(k_q.a[1][1], 2 * n1 * n1, 1e-9);
}

static void testSaturation()
{
	const Transmission tx = wristTransmission();
	const Vec3 cap{0.1, 0.1, 0.1};
	// Pure FE torque large enough to clip both differential motors.
	const Vec3 tau_m = tx.motorTorques(Vec3{0.0, 3.0, 0.0});
	const Vec3 sat = tx.saturate(tau_m, cap);
	for (int i = 0; i < 3; ++i) CHECK(std::fabs(sat[i]) <= cap[i] + 1e-12);
	const Vec3 tq = tx.jointTorques(sat);
	CHECK_NEAR(tq[2], 0.0, 1e-12);   // no UR torque leaks out of an FE command
	CHECK(tq[1] > 0.0);
	// FE + UR where only motor 7 clips: both differential motors scale together.
	const Vec3 mixed = tx.motorTorques(Vec3{0.0, 1.0, 1.0});  // all on motor 7
	const Vec3 s2 = tx.saturate(mixed, cap);
	const Vec3 t2 = tx.jointTorques(s2);
	CHECK_NEAR(t2[1], t2[2], 1e-12);  // direction kept
	// PS is independent of the differential.
	const Vec3 ps = tx.saturate(tx.motorTorques(Vec3{5.0, 0.1, 0.0}), cap);
	const Vec3 t3 = tx.jointTorques(ps);
	CHECK_NEAR(t3[1], 0.1, 1e-12);   // FE untouched by PS clipping
}

static void testTurns()
{
	CHECK(withinTurn(0, 4096) == 0);
	CHECK(withinTurn(4095, 4096) == -1);
	CHECK(withinTurn(2047, 4096) == 2047);
	CHECK(withinTurn(2048, 4096) == -2048);
	CHECK(withinTurn(-2049, 4096) == 2047);
	CHECK(withinTurn(4096 * 5 + 10, 4096) == 10);
	CHECK(singleTurnDirection(500, 500, 4096) == 1);
	CHECK(singleTurnDirection(500, 4096 - 500, 4096) == -1);
	CHECK(singleTurnDirection(5000, 5000 - 4096, 4096) == 1);   // more than a turn
	CHECK(singleTurnDirection(10, 10, 4096) == 0);              // too small to tell
	CHECK(singleTurnDirection(2048, 2048, 4096) == 0);          // half a turn: ambiguous
}

static void testSdoFrames()
{
	const CanFrame d = co::sdoDownloadRequest(7, 0x1017, 0, 250, 2);
	CHECK(d.id == 0x607);
	CHECK(d.len == 8);
	CHECK(d.data[0] == 0x2B);
	CHECK(d.data[1] == 0x17 && d.data[2] == 0x10 && d.data[3] == 0);
	CHECK(d.data[4] == 250 && d.data[5] == 0 && d.data[6] == 0);
	const CanFrame d1 = co::sdoDownloadRequest(6, 0x6060, 0, 10, 1);
	CHECK(d1.data[0] == 0x2F);
	const CanFrame d4 = co::sdoDownloadRequest(6, 0x1400, 1, 0x80000205u, 4);
	CHECK(d4.data[0] == 0x23 && d4.data[7] == 0x80 && d4.data[4] == 0x05);
	const CanFrame u = co::sdoUploadRequest(8, 0x6064, 0);
	CHECK(u.id == 0x608 && u.data[0] == 0x40 && u.data[1] == 0x64 && u.data[2] == 0x60);
	CHECK(co::mapEntry(0x6071, 0, 16) == 0x60710010u);
	CHECK(co::emcyIsWarning(0x2310));
	CHECK(!co::emcyIsWarning(0x8180));
}

static void testConfigValidation()
{
	p4w_config c;
	p4w_default_config(&c);
	c.node[2] = 7;
	CHECK(p4w_open_callback(&c, [](void*, uint32_t, const uint8_t*, int32_t) { return 0; },
			[](void*, uint32_t*, uint8_t*, int32_t*) { return 0; }, nullptr) == nullptr);
	CHECK(std::strstr(p4w_last_error(), "differ") != nullptr);
	p4w_default_config(&c);
	c.period_s = 0.0025;
	CHECK(p4w_open_callback(&c, [](void*, uint32_t, const uint8_t*, int32_t) { return 0; },
			[](void*, uint32_t*, uint8_t*, int32_t*) { return 0; }, nullptr) == nullptr);
	p4w_default_config(&c);
	p4w_wrist* w = p4w_open_callback(&c, [](void*, uint32_t, const uint8_t*, int32_t) { return 0; },
			[](void*, uint32_t*, uint8_t*, int32_t*) { return 0; }, nullptr);
	CHECK(w != nullptr);
	// No pucks answer: connect fails with a clear message instead of hanging.
	CHECK(p4w_connect(w, nullptr, nullptr, nullptr, 0) == -1);
	CHECK(std::strstr(p4w_last_error(), "did not boot") != nullptr);
	CHECK(p4w_enable(w, 0.1) == -1);  // not started
	p4w_close(w);
}

int main()
{
	testTransmission();
	testSaturation();
	testTurns();
	testSdoFrames();
	testConfigValidation();
	if (failures) {
		std::printf("%d check(s) failed\n", failures);
		return 1;
	}
	std::printf("all p4wrist core tests passed\n");
	return 0;
}
