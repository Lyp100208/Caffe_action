#ifndef CAFFE_OPTIMIZATION_SOLVER_HPP_
#define CAFFE_OPTIMIZATION_SOLVER_HPP_

#include <boost/function.hpp>
#include <string>
#include <vector>

#include "caffe/net.hpp"

namespace caffe {

/**
 * @brief Enumeration of actions that a client of the Solver may request by
 * implementing the Solver's action request function, which a
 * a client may optionally provide in order to request early termination
 * or saving a snapshot without exiting. In the executable caffe, this
 * mechanism is used to allow the snapshot to be saved when stopping
 * execution with a SIGINT (Ctrl-C).
 */
 namespace SolverAction {
  enum Enum {
    NONE = 0,  // Take no special action.
    STOP = 1,  // Stop training. snapshot_after_train controls whether a
               // snapshot is created.
    LR = 2,    // learing rate *= 0.1
    SNAPSHOT = 3  // Take a snapshot, and keep training.
  };
 }

/**
 * @brief Type of a function that returns a Solver Action enumeration.
 */
typedef boost::function<SolverAction::Enum()> ActionCallback;

/**
 * @brief An interface for classes that perform optimization on Net%s.
 *
 * Requires implementation of ApplyUpdate to compute a parameter update
 * given the current state of the Net parameters.
 */
template <typename Dtype>
class Solver {
 public:
  explicit Solver(const SolverParameter& param);
  explicit Solver(const string& param_file);
  void Init(const SolverParameter& param);
  void InitTrainNet();
  void InitTestNets();

  // Client of the Solver optionally may call this in order to set the function
  // that the solver uses to see what action it should take (e.g. snapshot or
  // exit training early).
  void SetActionFunction(ActionCallback func);
  SolverAction::Enum GetRequestedAction();
  // The main entry of the solver function. In default, iter will be zero. Pass
  // in a non-zero iter number to resume training for a pre-trained net.
  virtual void Solve(const char* resume_file = NULL);
  inline void Solve(const string resume_file) { Solve(resume_file.c_str()); }
  void Step(int iters);
  // The Restore function implements how one should restore the solver to a
  // previously snapshotted state. You should implement the RestoreSolverState()
  // function that restores the state from a SolverState protocol buffer.
  void Restore(const char* resume_file);
  virtual ~Solver() {}
  inline shared_ptr<Net<Dtype> > net() { return net_; }
  inline const vector<shared_ptr<Net<Dtype> > >& test_nets() {
    return test_nets_;
  }
  int iter() { return iter_; }

  // Invoked at specific points during an iteration
  class Callback {
  protected:
    virtual void on_start() = 0;
    virtual void on_gradients_ready() = 0;

    template <typename T>
    friend class Solver;
  };
  const vector<Callback*>& callbacks() const {return callbacks_;}
  void add_callback(Callback* value) {
    callbacks_.push_back(value);
  }

 protected:
  // Make and apply the update value for the current iteration.
  virtual void ApplyUpdate() = 0;
  // The Solver::Snapshot function implements the basic snapshotting utility
  // that stores the learned net. You should implement the SnapshotSolverState()
  // function that produces a SolverState protocol buffer that needs to be
  // written to disk together with the learned net.
  void Snapshot();
  // The test routine
  void TestAll();
  void Test(const int test_net_id = 0);
  virtual void SnapshotSolverState(SolverState* state) = 0;
  virtual void RestoreSolverState(const SolverState& state) = 0;
  void DisplayOutputBlobs(const int net_id);

  SolverParameter param_;
  int iter_;
  int current_step_;
  shared_ptr<Net<Dtype> > net_;
  vector<shared_ptr<Net<Dtype> > > test_nets_;
  vector<Callback*> callbacks_;

  // A function that can be set by a client of the Solver to provide indication
  // that it wants a snapshot saved and/or to exit early.
  ActionCallback action_request_function_;

  // True iff a request to stop early was received.
  bool requested_early_exit_;

  float lr_decay_;

  DISABLE_COPY_AND_ASSIGN(Solver);
};


/**
 * @brief Optimizes the parameters of a Net using
 *        stochastic gradient descent (SGD) with momentum.
 */
template <typename Dtype>
class SGDSolver : public Solver<Dtype> {
 public:
  explicit SGDSolver(const SolverParameter& param)
      : Solver<Dtype>(param) { PreSolve(); }
  explicit SGDSolver(const string& param_file)
      : Solver<Dtype>(param_file) { PreSolve(); }

  const vector<shared_ptr<Blob<Dtype> > >& history() { return history_; }

 protected:
  void PreSolve();
  Dtype GetLearningRate();
  virtual void ApplyUpdate();
  virtual void Normalize(int param_id);
  virtual void Regularize(int param_id);
  virtual void ComputeUpdateValue(int param_id, Dtype rate);
  virtual void ClipGradients();
  virtual void SnapshotSolverState(SolverState * state);
  virtual void RestoreSolverState(const SolverState& state);
  // history maintains the historical momentum data.
  // update maintains update related data and is not needed in snapshots.
  // temp maintains other information that might be needed in computation
  //   of gradients/updates and is not needed in snapshots
  vector<shared_ptr<Blob<Dtype> > > history_, update_, temp_;

  DISABLE_COPY_AND_ASSIGN(SGDSolver);
};

template <typename Dtype>
class NesterovSolver : public SGDSolver<Dtype> {
 public:
  explicit NesterovSolver(const SolverParameter& param)
      : SGDSolver<Dtype>(param) {}
  explicit NesterovSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) {}

 protected:
  virtual void ComputeUpdateValue(int param_id, Dtype rate);

  DISABLE_COPY_AND_ASSIGN(NesterovSolver);
};

template <typename Dtype>
class AdaGradSolver : public SGDSolver<Dtype> {
 public:
  explicit AdaGradSolver(const SolverParameter& param)
      : SGDSolver<Dtype>(param) { constructor_sanity_check(); }
  explicit AdaGradSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) { constructor_sanity_check(); }

 protected:
  virtual void ComputeUpdateValue(int param_id, Dtype rate);
  void constructor_sanity_check() {
    CHECK_EQ(0, this->param_.momentum())
        << "Momentum cannot be used with AdaGrad.";
  }

  DISABLE_COPY_AND_ASSIGN(AdaGradSolver);
};

template <typename Dtype>
Solver<Dtype>* GetSolver(const SolverParameter& param) {
  SolverParameter_SolverType type = param.solver_type();

  switch (type) {
  case SolverParameter_SolverType_SGD:
      return new SGDSolver<Dtype>(param);
  case SolverParameter_SolverType_NESTEROV:
      return new NesterovSolver<Dtype>(param);
  case SolverParameter_SolverType_ADAGRAD:
      return new AdaGradSolver<Dtype>(param);
  default:
      LOG(FATAL) << "Unknown SolverType: " << type;
  }
  return (Solver<Dtype>*) NULL;
}

}  // namespace caffe

#endif  // CAFFE_OPTIMIZATION_SOLVER_HPP_
