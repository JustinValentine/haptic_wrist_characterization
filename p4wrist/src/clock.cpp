#include "p4wrist/can.h"

#include <chrono>
#include <thread>

#ifdef __linux__
#include <cerrno>
#include <ctime>
#endif

namespace p4wrist {

#ifdef __linux__
double monotonicNow()
{
	timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return double(ts.tv_sec) + 1e-9 * double(ts.tv_nsec);
}

void sleepUntil(double t)
{
	timespec ts;
	ts.tv_sec = time_t(t);
	ts.tv_nsec = long((t - double(ts.tv_sec)) * 1e9);
	if (ts.tv_nsec >= 1000000000L) {
		ts.tv_sec += 1;
		ts.tv_nsec -= 1000000000L;
	}
	while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr) == EINTR) {
	}
}
#else
double monotonicNow()
{
	using namespace std::chrono;
	return duration<double>(steady_clock::now().time_since_epoch()).count();
}

void sleepUntil(double t)
{
	using namespace std::chrono;
	const double dt = t - monotonicNow();
	if (dt > 0) std::this_thread::sleep_for(duration<double>(dt));
}
#endif

}  // namespace p4wrist
