#ifndef INCLUDE_CAFFE_UTIL_SIGNAL_HANDLER_H_
#define INCLUDE_CAFFE_UTIL_SIGNAL_HANDLER_H_

#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"

namespace caffe {

class SignalHandler {
public:
	// Contructor. Specify what action to take when a signal is received.
	SignalHandler(SolverAction::Enum SIGUSR1_action, SolverAction::Enum SIGUSR2_action);
	~SignalHandler();
	ActionCallback GetActionFunction();
private:
	SolverAction::Enum CheckForSignals() const;
	SolverAction::Enum SIGUSR1_action_;
	SolverAction::Enum SIGUSR2_action_;
};

}  // namespace caffe

#endif  // INCLUDE_CAFFE_UTIL_SIGNAL_HANDLER_H_