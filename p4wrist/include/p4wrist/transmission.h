// The wrist's joint <-> motor maps.
//
// Motor angles are linear in the joint angles, m = j2mp q (rows motors 6, 7, 8;
// columns PS, FE, UR). For the wrist as commissioned on the WAM (libbarrett-
// canopen config/wam8g.conf, 2026-09-22):
//
//   m6 = -N2 q_PS,  m7 = N1 (q_FE + q_UR),  m8 = N1 (-q_FE + q_UR)
//   N2 = 8.7456,    N1 = 7.5077
//
// so FE is the difference of the differential motors and UR their sum. By
// virtual work, joint torques are tau_q = j2mp^T tau_m, and the motor torques
// that produce a joint torque are tau_m = j2mp^-T tau_q.
#pragma once

#include <array>
#include <cmath>
#include <stdexcept>

namespace p4wrist {

using Vec3 = std::array<double, 3>;

struct Mat3 {
	double a[3][3] = {};

	static Mat3 identity() {
		Mat3 m;
		for (int i = 0; i < 3; ++i) m.a[i][i] = 1.0;
		return m;
	}
	Vec3 operator*(const Vec3& v) const {
		Vec3 r{};
		for (int i = 0; i < 3; ++i)
			for (int j = 0; j < 3; ++j) r[i] += a[i][j] * v[j];
		return r;
	}
	Mat3 operator*(const Mat3& o) const {
		Mat3 r;
		for (int i = 0; i < 3; ++i)
			for (int j = 0; j < 3; ++j)
				for (int k = 0; k < 3; ++k) r.a[i][j] += a[i][k] * o.a[k][j];
		return r;
	}
	Mat3 transpose() const {
		Mat3 r;
		for (int i = 0; i < 3; ++i)
			for (int j = 0; j < 3; ++j) r.a[i][j] = a[j][i];
		return r;
	}
	double determinant() const {
		return a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
				- a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
				+ a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
	}
	Mat3 inverse() const {
		const double det = determinant();
		if (!(std::fabs(det) > 1e-12)) throw std::invalid_argument("singular transmission matrix");
		Mat3 r;
		r.a[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) / det;
		r.a[0][1] = (a[0][2] * a[2][1] - a[0][1] * a[2][2]) / det;
		r.a[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) / det;
		r.a[1][0] = (a[1][2] * a[2][0] - a[1][0] * a[2][2]) / det;
		r.a[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) / det;
		r.a[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) / det;
		r.a[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) / det;
		r.a[2][1] = (a[0][1] * a[2][0] - a[0][0] * a[2][1]) / det;
		r.a[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) / det;
		return r;
	}
};

class Transmission {
public:
	explicit Transmission(const Mat3& j2mp)
		: j2mp_(j2mp), mp2j_(j2mp.inverse()), mt2j_(j2mp.transpose()),
		  j2mt_(j2mp.transpose().inverse()) {
		// Motors that share a joint saturate together (see saturate()).
		for (int i = 0; i < 3; ++i) {
			for (int k = 0; k < 3; ++k) {
				coupled_[i][k] = false;
				for (int j = 0; j < 3; ++j)
					if (j2mp.a[i][j] != 0.0 && j2mp.a[k][j] != 0.0) coupled_[i][k] = true;
			}
		}
	}

	Vec3 motorAngles(const Vec3& q) const { return j2mp_ * q; }
	Vec3 jointAngles(const Vec3& m) const { return mp2j_ * m; }
	Vec3 motorTorques(const Vec3& tau_q) const { return j2mt_ * tau_q; }
	Vec3 jointTorques(const Vec3& tau_m) const { return mt2j_ * tau_m; }
	const Mat3& j2mp() const { return j2mp_; }

	// Scale motor torques down to |tau_m[i]| <= cap[i]. Motors coupled through
	// a joint are scaled by a common factor, which keeps the joint torque's
	// direction instead of letting one clipped motor leak torque into the other
	// joint of the differential.
	Vec3 saturate(const Vec3& tau_m, const Vec3& cap) const {
		Vec3 need{};
		for (int i = 0; i < 3; ++i)
			need[i] = (std::fabs(tau_m[i]) > cap[i]) ? cap[i] / std::fabs(tau_m[i]) : 1.0;
		Vec3 out{};
		for (int i = 0; i < 3; ++i) {
			double s = 1.0;
			for (int k = 0; k < 3; ++k)
				if (coupled_[i][k] && need[k] < s) s = need[k];
			out[i] = tau_m[i] * s;
		}
		return out;
	}

private:
	Mat3 j2mp_, mp2j_, mt2j_, j2mt_;
	bool coupled_[3][3];
};

}  // namespace p4wrist
