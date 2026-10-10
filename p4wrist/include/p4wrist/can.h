// CAN frames, the bus interface, and the loop clock.
#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace p4wrist {

struct CanFrame {
	uint32_t id = 0;   // 11-bit identifier
	uint8_t len = 0;
	uint8_t data[8] = {};
};

// One CAN bus. send() throws BusError if the frame cannot be queued.
// receive() waits up to timeout_s (0 polls) and returns false if nothing came.
class CanBus {
public:
	virtual ~CanBus() = default;
	virtual void send(const CanFrame& frame) = 0;
	virtual bool receive(CanFrame* frame, double timeout_s) = 0;
	// True for the emulator, which answers inside send(): an empty queue then
	// means nothing more will arrive, so waits end at once instead of timing out.
	virtual bool synchronous() const { return false; }
};

struct BusError : std::runtime_error {
	using std::runtime_error::runtime_error;
};

// SocketCAN raw socket (Linux only; throws BusError elsewhere).
std::unique_ptr<CanBus> openSocketCan(const std::string& interface);

// Monotonic seconds, and an absolute sleep on the same clock.
double monotonicNow();
void sleepUntil(double t);

}  // namespace p4wrist
