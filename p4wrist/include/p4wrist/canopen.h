// CANopen (CiA 301/402) pieces of the Barrett P4 that the wrist uses.
//
// The bus layout matches libbarrett-canopen's WAM contract, so the same pucks
// work on the WAM and on the bench without reconfiguring:
//   SYNC 0x080 every 2 ms; torque quad B on 0x205 (slot 0 unused - the WAM's
//   passive J5 - then motors 6, 7, 8); position TPDO1 on 0x180+node every SYNC;
//   statusword TPDO2 on 0x280+node; controlword + mode broadcast on 0x28A (the
//   WAM safety board's COB - this library stands in for the board); host
//   heartbeat 0x77F.
#pragma once

#include <cstdint>
#include <deque>
#include <functional>
#include <stdexcept>
#include <string>

#include "p4wrist/can.h"

namespace p4wrist {
namespace co {

constexpr uint32_t COB_NMT = 0x000;
constexpr uint32_t COB_SYNC = 0x080;
constexpr uint32_t COB_EMCY = 0x080;        // + node
constexpr uint32_t COB_TPDO1 = 0x180;       // + node: position (+ current)
constexpr uint32_t COB_TPDO2 = 0x280;       // + node: statusword + mode display
constexpr uint32_t COB_TPDO3 = 0x380;       // + node: bus voltage, temperatures
constexpr uint32_t COB_TPDO4 = 0x480;
constexpr uint32_t COB_RPDO3 = 0x400;
constexpr uint32_t COB_RPDO4 = 0x500;
constexpr uint32_t COB_SDO_TX = 0x580;      // + node: server -> host
constexpr uint32_t COB_SDO_RX = 0x600;      // + node: host -> server
constexpr uint32_t COB_HEARTBEAT = 0x700;   // + node
constexpr uint32_t COB_QUAD_B = 0x205;
constexpr uint32_t COB_CONTROL = 0x28A;
constexpr uint32_t COB_HOST_HEARTBEAT = 0x77F;

constexpr uint8_t NMT_START = 0x01;
constexpr uint8_t NMT_STOP = 0x02;
constexpr uint8_t NMT_PREOP = 0x80;
constexpr uint8_t NMT_RESET_NODE = 0x81;
constexpr uint8_t NMT_RESET_COMM = 0x82;

constexpr uint8_t HB_BOOT_UP = 0x00;
constexpr uint8_t HB_STOPPED = 0x04;
constexpr uint8_t HB_OPERATIONAL = 0x05;
constexpr uint8_t HB_PREOP = 0x7F;

constexpr uint16_t OD_DEVICE_TYPE = 0x1000;
constexpr uint16_t OD_SYNC_PERIOD = 0x1006;
constexpr uint16_t OD_HB_PRODUCER = 0x1017;
constexpr uint16_t OD_SYNC_LOSS_TIMEOUT = 0x2406;
constexpr uint16_t OD_AMPLIFIER = 0x3000;      // :1 bus voltage (0.1 V), :2 temperature (C)
constexpr uint16_t OD_MOTOR_SENSORS = 0x3010;  // :3 motor thermistor
constexpr uint16_t OD_MOTOR = 0x3011;          // :4 Kt (mNm/A), :9 peak current (mA)
constexpr uint16_t OD_RAW_POSITION = 0x3012;   // :1 single-turn absolute encoder
constexpr uint16_t OD_ENCODER = 0x3013;        // :1 counts per motor turn
constexpr uint16_t OD_ERROR_CODE = 0x603F;
constexpr uint16_t OD_CONTROLWORD = 0x6040;
constexpr uint16_t OD_STATUSWORD = 0x6041;
constexpr uint16_t OD_MODE = 0x6060;
constexpr uint16_t OD_MODE_DISPLAY = 0x6061;
constexpr uint16_t OD_POSITION = 0x6064;
constexpr uint16_t OD_TARGET_TORQUE = 0x6071;
constexpr uint16_t OD_RATED_TORQUE = 0x6076;
constexpr uint16_t OD_CURRENT = 0x6078;        // q-axis, per mille of peak current
constexpr uint16_t OD_INTERP_PERIOD = 0x60C2;

constexpr int8_t MODE_IDLE = 0;   // current loop off, phases grounded
constexpr int8_t MODE_CST = 10;

constexpr uint16_t CW_DISABLE_VOLTAGE = 0x0000;
constexpr uint16_t CW_SHUTDOWN = 0x0006;
constexpr uint16_t CW_ENABLE_OPERATION = 0x000F;
constexpr uint16_t CW_FAULT_RESET = 0x0080;

constexpr uint32_t PDO_COB_INVALID = 0x80000000u;
// CiA 301 dummy entry (INTEGER16) padding a torque quad; the pattern Barrett's
// production configs use.
constexpr uint32_t PDO_MAP_DUMMY16 = 0x00030010u;

constexpr uint32_t mapEntry(uint16_t index, uint8_t sub, uint8_t bits) {
	return (uint32_t(index) << 16) | (uint32_t(sub) << 8) | bits;
}

inline bool swFault(uint16_t sw) { return (sw & 0x0008) != 0; }
inline bool swOperationEnabled(uint16_t sw) { return (sw & 0x006F) == 0x0027; }
inline bool swReadyToSwitchOn(uint16_t sw) { return (sw & 0x006F) == 0x0021; }
inline bool swSwitchOnDisabled(uint16_t sw) { return (sw & 0x004F) == 0x0040; }

// EMCY codes the P4 sends as warnings (auto-reset); every other nonzero code
// is a latched fault (P4 CANopen manual Rev BA, table 19).
bool emcyIsWarning(uint16_t code);
const char* emcyText(uint16_t code);

// Expedited SDO frames.
CanFrame sdoDownloadRequest(int node, uint16_t index, uint8_t sub, uint32_t value, int nbytes);
CanFrame sdoUploadRequest(int node, uint16_t index, uint8_t sub);
bool sdoMatches(const CanFrame& reply, uint16_t index, uint8_t sub);

}  // namespace co

struct SdoError : std::runtime_error {
	SdoError(const std::string& what, uint32_t abort) : std::runtime_error(what), abort_code(abort) {}
	uint32_t abort_code;  // 0 for timeouts and malformed replies
};

// Setup-time CANopen master: NMT and blocking SDO transfers. Not for use while
// the SYNC loop runs (the loop owns the bus then). Frames that arrive while it
// waits for something else are kept, in order, in stash().
class Master {
public:
	explicit Master(CanBus& bus, double sdo_timeout_s = 0.5);

	void nmt(uint8_t command, int node);
	void download(int node, uint16_t index, uint8_t sub, uint32_t value, int nbytes);
	int32_t upload(int node, uint16_t index, uint8_t sub);
	// Wait for this node's boot-up message (heartbeat state 0x00). A P4 also
	// sends one last 0x05 heartbeat after a reset; only 0x00 counts.
	bool waitBootUp(int node, double timeout_s);
	// Discard everything received so far.
	void drain();
	std::deque<CanFrame>& stash() { return stash_; }

private:
	bool waitFor(const std::function<bool(const CanFrame&)>& match, double timeout_s, CanFrame* out);

	CanBus& bus_;
	double timeout_;
	std::deque<CanFrame> stash_;
};

}  // namespace p4wrist
