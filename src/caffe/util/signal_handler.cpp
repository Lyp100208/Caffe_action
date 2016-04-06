#include <boost/bind.hpp>
#include <glog/logging.h>

#include <signal.h>
#include <csignal>

#include "caffe/util/signal_handler.hpp"

namespace {

static volatile sig_atomic_t got_sigusr1 = false;
static volatile sig_atomic_t got_sigusr2 = false;

static bool already_hooked_up = false;

void handle_signal(int signal) {
	switch (signal) {
		case SIGUSR1:
			got_sigusr1 = true;
			break;
		case SIGUSR2:
			got_sigusr2 = true;
			break;
	}
}

void HookupHandler() {
	if (already_hooked_up) {
		LOG(FATAL) << "Tried to hookup signal handlers more than once.";
	}
	already_hooked_up = true;

	struct sigaction sa;
	// Setup the handler
	sa.sa_handler = &handle_signal;
	// Restart the system call, if at all possible
	sa.sa_flags = SA_RESTART;
	// Block every signal during the handler
	sigfillset(&sa.sa_mask);
	// Intercept SIGUSR1 and SIGUSR2
	if (sigaction(SIGUSR1, &sa, NULL) == -1) {
		LOG(FATAL) << "Cannot install SIGUSR1 handler.";
	}
	if (sigaction(SIGUSR2, &sa, NULL) == -1) {
		LOG(FATAL) << "Cannot install SIGUSR2 handler.";
	}
}

// Set the signal handlers to the default.
void UnhookHandler() {
	if (already_hooked_up) {
		struct sigaction sa;
		// Setup the sighub handler
		sa.sa_handler = SIG_DFL;
		// Restart the system call, if at all possible
		sa.sa_flags = SA_RESTART;
		// Block every signal during the handler
		sigfillset(&sa.sa_mask);
		// Intercept SIGUSR1 and SIGUSR2
		if (sigaction(SIGUSR1, &sa, NULL) == -1) {
			LOG(FATAL) << "Cannot uninstall SIGUSR1 handler.";
		}
		if (sigaction(SIGUSR2, &sa, NULL) == -1) {
			LOG(FATAL) << "Cannot uninstall SIGUSR2 handler.";
		}

		already_hooked_up = false;
	}
}

// Return true iff a SIGUSR1 has been received since the last time this
// function was called.
bool GotSIGUSR1() {
	bool result = got_sigusr1;
	got_sigusr1 = false;
	return result;
}
// Return true iff a SIGHUP has been received since the last time this
// function was called.
bool GotSIGUSR2() {
	bool result = got_sigusr2;
	got_sigusr2 = false;
	return result;
}

}  // namespace

namespace caffe {

SignalHandler::SignalHandler(SolverAction::Enum SIGUSR1_action, SolverAction::Enum SIGUSR2_action)
  : SIGUSR1_action_(SIGUSR1_action), SIGUSR2_action_(SIGUSR2_action) {
	HookupHandler();
}

SignalHandler::~SignalHandler() {
	UnhookHandler();
}

SolverAction::Enum SignalHandler::CheckForSignals() const {
	if (GotSIGUSR1()) {
		return SIGUSR1_action_;
	}
	if (GotSIGUSR2()) {
		return SIGUSR2_action_;
	}
	return SolverAction::NONE;
}

// Return the function that the solver can use to find out if a snapshot or
// early exit is being requested.
ActionCallback SignalHandler::GetActionFunction() {
	return boost::bind(&SignalHandler::CheckForSignals, this);
}

}  // namespace caffe