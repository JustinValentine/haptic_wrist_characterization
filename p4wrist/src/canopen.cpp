#include "p4wrist/canopen.h"

#include <cstdio>

namespace p4wrist {
namespace co {

bool emcyIsWarning(uint16_t code)
{
	switch (code) {
	case 0x2310:  // current limit (i2t) active
	case 0x6010:  // watchdog reset (reported after boot-up)
	case 0x7383: case 0x7384:  // soft stops
	case 0x8411:  // velocity tracking warning
	case 0x8418:  // velocity limit active
	case 0x8480:  // acceleration limit active
	case 0x8611:  // position tracking warning
		return true;
	default:
		return false;
	}
}

const char* emcyText(uint16_t code)
{
	switch (code) {
	case 0x0000: return "error cleared";
	case 0x2310: return "current limit active (i2t)";
	case 0x2320: return "short circuit";
	case 0x3210: return "bus over-voltage";
	case 0x3220: return "bus under-voltage (motor supply off?)";
	case 0x4210: return "amplifier over-temperature";
	case 0x4310: return "motor over-temperature";
	case 0x6010: return "watchdog reset";
	case 0x6320: return "parameter error";
	case 0x7122: return "motor phasing error";
	case 0x7320: return "encoder feedback fault";
	case 0x7321: return "encoder magnet distance fault";
	case 0x73A0: return "position wrapped";
	case 0x8110: return "CAN overrun (frames lost)";
	case 0x8140: return "recovered from bus-off";
	case 0x8180: return "SYNC loss";
	case 0x8411: return "velocity tracking warning";
	case 0x8414: return "velocity tracking fault";
	case 0x8418: return "velocity limit active";
	case 0x8480: return "acceleration limit active";
	case 0x8611: return "position tracking warning";
	case 0x8614: return "position tracking fault";
	default: return "unknown EMCY code";
	}
}

CanFrame sdoDownloadRequest(int node, uint16_t index, uint8_t sub, uint32_t value, int nbytes)
{
	CanFrame f;
	f.id = COB_SDO_RX + node;
	f.len = 8;
	f.data[0] = uint8_t(0x23 | ((4 - nbytes) << 2));  // expedited, size indicated
	f.data[1] = index & 0xFF;
	f.data[2] = index >> 8;
	f.data[3] = sub;
	for (int i = 0; i < 4; ++i) f.data[4 + i] = i < nbytes ? uint8_t(value >> (8 * i)) : 0;
	return f;
}

CanFrame sdoUploadRequest(int node, uint16_t index, uint8_t sub)
{
	CanFrame f;
	f.id = COB_SDO_RX + node;
	f.len = 8;
	f.data[0] = 0x40;
	f.data[1] = index & 0xFF;
	f.data[2] = index >> 8;
	f.data[3] = sub;
	return f;
}

bool sdoMatches(const CanFrame& reply, uint16_t index, uint8_t sub)
{
	return reply.len == 8 && reply.data[1] == (index & 0xFF) && reply.data[2] == (index >> 8)
			&& reply.data[3] == sub;
}

}  // namespace co

namespace {
std::string where(int node, uint16_t index, uint8_t sub)
{
	char text[64];
	std::snprintf(text, sizeof(text), "node %d 0x%04X:%02X", node, index, sub);
	return text;
}

uint32_t abortCode(const CanFrame& f)
{
	return uint32_t(f.data[4]) | (uint32_t(f.data[5]) << 8) | (uint32_t(f.data[6]) << 16)
			| (uint32_t(f.data[7]) << 24);
}

std::string hex32(uint32_t v)
{
	char text[16];
	std::snprintf(text, sizeof(text), "0x%08X", v);
	return text;
}
}  // namespace

Master::Master(CanBus& bus, double sdo_timeout_s) : bus_(bus), timeout_(sdo_timeout_s) {}

void Master::nmt(uint8_t command, int node)
{
	if (node == 0 && command == co::NMT_RESET_NODE) {
		// Broadcast resets restart every node on the bus, including ones this
		// library does not own (libbarrett-canopen's rule too).
		throw std::logic_error("refusing to broadcast NMT reset-node");
	}
	CanFrame f;
	f.id = co::COB_NMT;
	f.len = 2;
	f.data[0] = command;
	f.data[1] = uint8_t(node);
	bus_.send(f);
}

bool Master::waitFor(const std::function<bool(const CanFrame&)>& match, double timeout_s, CanFrame* out)
{
	const double deadline = monotonicNow() + timeout_s;
	CanFrame f;
	while (true) {
		const double remaining = deadline - monotonicNow();
		if (remaining <= 0.0) return false;
		if (!bus_.receive(&f, remaining)) {
			if (bus_.synchronous()) return false;
			continue;
		}
		if (match(f)) {
			*out = f;
			return true;
		}
		stash_.push_back(f);
		if (stash_.size() > 4096) stash_.pop_front();
	}
}

void Master::download(int node, uint16_t index, uint8_t sub, uint32_t value, int nbytes)
{
	if (nbytes != 1 && nbytes != 2 && nbytes != 4) throw std::logic_error("SDO size must be 1, 2 or 4");
	// A reply still buffered belongs to an earlier, timed-out transfer.
	for (auto it = stash_.begin(); it != stash_.end();)
		it = (it->id == co::COB_SDO_TX + uint32_t(node)) ? stash_.erase(it) : it + 1;
	bus_.send(co::sdoDownloadRequest(node, index, sub, value, nbytes));
	CanFrame reply;
	const auto match = [&](const CanFrame& f) {
		return f.id == co::COB_SDO_TX + uint32_t(node) && co::sdoMatches(f, index, sub);
	};
	if (!waitFor(match, timeout_, &reply)) {
		throw SdoError("no SDO reply writing " + where(node, index, sub), 0);
	}
	if (reply.data[0] == 0x80) {
		throw SdoError("SDO abort " + hex32(abortCode(reply)) + " writing " + where(node, index, sub),
				abortCode(reply));
	}
	if (reply.data[0] != 0x60) throw SdoError("unexpected SDO reply writing " + where(node, index, sub), 0);
}

int32_t Master::upload(int node, uint16_t index, uint8_t sub)
{
	for (auto it = stash_.begin(); it != stash_.end();)
		it = (it->id == co::COB_SDO_TX + uint32_t(node)) ? stash_.erase(it) : it + 1;
	bus_.send(co::sdoUploadRequest(node, index, sub));
	CanFrame reply;
	const auto match = [&](const CanFrame& f) {
		return f.id == co::COB_SDO_TX + uint32_t(node) && co::sdoMatches(f, index, sub);
	};
	if (!waitFor(match, timeout_, &reply)) {
		throw SdoError("no SDO reply reading " + where(node, index, sub), 0);
	}
	const uint8_t cs = reply.data[0];
	if (cs == 0x80) {
		throw SdoError("SDO abort " + hex32(abortCode(reply)) + " reading " + where(node, index, sub),
				abortCode(reply));
	}
	// Expedited upload response: 0x4n with e=1 (bit 1).
	if ((cs & 0xE0) != 0x40 || !(cs & 0x02)) {
		throw SdoError("non-expedited SDO reply reading " + where(node, index, sub), 0);
	}
	const int nbytes = (cs & 0x01) ? 4 - ((cs >> 2) & 3) : 4;
	// Sign-extend from the transferred size (I16 objects such as 0x6078 and
	// 0x3012:01 come back as two bytes).
	uint32_t v = (reply.data[3 + nbytes] & 0x80) ? 0xFFFFFFFFu : 0u;
	for (int i = 3 + nbytes; i >= 4; --i) v = (v << 8) | reply.data[i];
	return int32_t(v);
}

bool Master::waitBootUp(int node, double timeout_s)
{
	const auto isBootUp = [node](const CanFrame& f) {
		return f.id == co::COB_HEARTBEAT + uint32_t(node) && f.len == 1 && f.data[0] == co::HB_BOOT_UP;
	};
	for (auto it = stash_.begin(); it != stash_.end(); ++it) {
		if (isBootUp(*it)) {
			stash_.erase(it);
			return true;
		}
	}
	CanFrame f;
	return waitFor(isBootUp, timeout_s, &f);
}

void Master::drain()
{
	stash_.clear();
	CanFrame f;
	while (bus_.receive(&f, 0.0)) {
	}
}

}  // namespace p4wrist
