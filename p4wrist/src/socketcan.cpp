#include "p4wrist/can.h"

#ifdef __linux__

#include <cerrno>
#include <cmath>
#include <cstring>

#include <linux/can.h>
#include <linux/can/error.h>
#include <linux/can/raw.h>
#include <net/if.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

namespace p4wrist {

namespace {
class SocketCan : public CanBus {
public:
	explicit SocketCan(const std::string& interface)
	{
		fd_ = socket(PF_CAN, SOCK_RAW, CAN_RAW);
		if (fd_ < 0) throw BusError(std::string("socket(PF_CAN): ") + std::strerror(errno));
		ifreq ifr{};
		std::strncpy(ifr.ifr_name, interface.c_str(), IFNAMSIZ - 1);
		if (ioctl(fd_, SIOCGIFINDEX, &ifr) < 0) {
			const std::string why = std::strerror(errno);
			close(fd_);
			throw BusError("CAN interface " + interface + ": " + why
					+ " (bring it up with: sudo ip link set " + interface + " up type can bitrate 1000000)");
		}
		// Bus-off and controller problems arrive as error frames.
		can_err_mask_t err_mask = CAN_ERR_BUSOFF | CAN_ERR_CRTL | CAN_ERR_RESTARTED;
		setsockopt(fd_, SOL_CAN_RAW, CAN_RAW_ERR_FILTER, &err_mask, sizeof(err_mask));
		int rcvbuf = 1 << 20;
		setsockopt(fd_, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));
		sockaddr_can addr{};
		addr.can_family = AF_CAN;
		addr.can_ifindex = ifr.ifr_ifindex;
		if (bind(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
			const std::string why = std::strerror(errno);
			close(fd_);
			throw BusError("bind " + interface + ": " + why);
		}
	}

	~SocketCan() override { close(fd_); }

	void send(const CanFrame& frame) override
	{
		can_frame f{};
		f.can_id = frame.id & CAN_SFF_MASK;
		f.can_dlc = frame.len;
		std::memcpy(f.data, frame.data, frame.len);
		// A full transmit queue (ENOBUFS) clears within a frame time or two;
		// anything longer means the bus is down or nobody acknowledges.
		for (int attempt = 0; attempt < 50; ++attempt) {
			const ssize_t n = write(fd_, &f, sizeof(f));
			if (n == ssize_t(sizeof(f))) return;
			if (n < 0 && errno != ENOBUFS && errno != EAGAIN && errno != EINTR)
				throw BusError(std::string("CAN write: ") + std::strerror(errno));
			sleepUntil(monotonicNow() + 20e-6);
		}
		throw BusError("CAN transmit queue stayed full (no other node acknowledging? bus-off?)");
	}

	bool receive(CanFrame* frame, double timeout_s) override
	{
		pollfd p{fd_, POLLIN, 0};
		const int ms = timeout_s <= 0.0 ? 0 : int(std::ceil(timeout_s * 1000.0));
		while (true) {
			const int r = poll(&p, 1, ms);
			if (r < 0 && errno == EINTR) continue;
			if (r <= 0) return false;
			can_frame f{};
			const ssize_t n = read(fd_, &f, sizeof(f));
			if (n != ssize_t(sizeof(f))) return false;
			if (f.can_id & CAN_ERR_FLAG) {
				if (f.can_id & CAN_ERR_BUSOFF) throw BusError("CAN controller went bus-off");
				continue;
			}
			if (f.can_id & (CAN_EFF_FLAG | CAN_RTR_FLAG)) continue;
			frame->id = f.can_id & CAN_SFF_MASK;
			frame->len = f.can_dlc > 8 ? 8 : f.can_dlc;
			std::memcpy(frame->data, f.data, frame->len);
			return true;
		}
	}

private:
	int fd_ = -1;
};
}  // namespace

std::unique_ptr<CanBus> openSocketCan(const std::string& interface)
{
	return std::unique_ptr<CanBus>(new SocketCan(interface));
}

}  // namespace p4wrist

#else

namespace p4wrist {
std::unique_ptr<CanBus> openSocketCan(const std::string&)
{
	throw BusError("SocketCAN is Linux only; use a callback bus (simulation) on this OS");
}
}  // namespace p4wrist

#endif
