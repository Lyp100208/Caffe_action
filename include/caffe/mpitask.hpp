#ifndef CAFFE_MPITASK_HPP_
#define CAFFE_MPITASK_HPP_

#include <algorithm>
#include <string>
#include <vector>
#include "boost/thread/thread.hpp"  
#include "boost/bind.hpp"  
#include "boost/thread/mutex.hpp" 
#include <unistd.h>

#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/internal_thread.hpp"

namespace caffe {

/**
 */

template <typename Dtype>
class MpiTask {
 public:
  explicit MpiTask(Blob<Dtype> *send_blob,int send_mem, Blob<Dtype> *recv_blob,
     int recv_mem, int count, int tag = -1) {
      send_blob_ = send_blob;
      recv_blob_ = recv_blob;
      send_mem_ = send_mem;
      recv_mem_ = recv_mem;
      count_ = count;
      tag_ = tag;
      mpi_func_ = 0; // MPI Function, 0 allreduce
      mpi_datatype_ = MPI_FLOAT;
      mpi_op_ = MPI_SUM;
      mpi_comm_ = MPI_COMM_WORLD;
  }

  explicit MpiTask(Blob<Dtype> *send_blob,int send_mem, Blob<Dtype> *recv_blob,
     int recv_mem, int count, int mpi_func, MPI_Datatype mpi_datatype, 
     MPI_Op mpi_op, MPI_Comm mpi_comm, int tag = -1) {
      send_blob_ = send_blob;
      recv_blob_ = recv_blob;
      send_mem_ = send_mem;
      recv_mem_ = recv_mem;
      count_ = count;
      tag_ = tag;
      mpi_func_ = mpi_func; // MPI Function, 0 allreduce
      mpi_datatype_ = mpi_datatype;
      mpi_op_ = mpi_op;
      mpi_comm_ = mpi_comm;
  }

  inline void sed_recv_blob (Blob<Dtype> * recv_blob) { recv_blob_ = recv_blob; }
  inline void sed_send_blob (Blob<Dtype> * send_blob) { send_blob_ = send_blob; }
  inline void set_count (int count) { count_ = count; }
  inline void set_tag (int tag) { tag_ = tag; }
  inline void set_send_mem (int send_mem) { send_mem_ = send_mem; }
  inline void set_recv_mem (int recv_mem) { recv_mem_ = recv_mem; }
  inline void set_mpi_func (int mpi_func) { mpi_func_ = mpi_func; }
  inline void set_mpi_datatype (MPI_Datatype mpi_datatype) { mpi_datatype_ = mpi_datatype; }
  inline void set_mpi_op (MPI_Op mpi_op) {mpi_op_ = mpi_op; }
  inline void set_mpi_comm (MPI_Comm mpi_comm) {mpi_comm_ = mpi_comm; }

  inline Blob<Dtype> * recv_blob() { return recv_blob_; }
  inline Blob<Dtype> * send_blob() { return send_blob_; }
  inline int count() { return count_; }
  inline int tag() { return tag_; }
  inline int send_mem() { return send_mem_; }
  inline int recv_mem() { return recv_mem_; } //0 cpu data,1 cpu mem, 2 gpu data, 3 gpu mem
  inline int mpi_func() { return mpi_func_; }
  inline MPI_Datatype mpi_datatype() { return mpi_datatype_; }
  inline MPI_Op mpi_op() { return mpi_op_; }
  inline MPI_Comm mpi_comm() { return mpi_comm_; }
 private:
  int count_;
  int tag_;
  int send_mem_;
  int recv_mem_;
  Blob<Dtype>* recv_blob_;
  Blob<Dtype>* send_blob_;
  int mpi_func_; // MPI Function, 0 allreduce, 1 reduce,
  MPI_Datatype mpi_datatype_;
  MPI_Op mpi_op_;
  MPI_Comm mpi_comm_;
};

template <typename Dtype>
class MpiTaskList : public InternalThread {
 public:
  virtual inline void InternalThreadEntry() {
    while(1) {
      if(size() > 0) {
        MpiTask<Dtype>* task = task_list_[0];
        Dtype* recvbuf = NULL;
        const Dtype* sendbuf = NULL; 
        if(task->send_blob() != NULL){
          if( task->send_mem() == 0) 
            sendbuf = task->send_blob()->cpu_data();
          if ( task->send_mem() == 1)
            sendbuf = task->send_blob()->cpu_diff();
        }
        if(task->recv_mem() == 0)
          recvbuf = task->recv_blob()->mutable_cpu_data();
        if(task->recv_mem() == 1)
          recvbuf = task->recv_blob()->mutable_cpu_diff();
        if(task->mpi_func() == 0) {
          if(task->send_blob() != NULL)
            MPI_Allreduce(sendbuf,recvbuf,task->count(), 
              task->mpi_datatype(),task->mpi_op(),task->mpi_comm());
          else
            MPI_Allreduce(MPI_IN_PLACE,recvbuf,task->count(), 
              task->mpi_datatype(),task->mpi_op(),task->mpi_comm());
        }
        delete task;
        erase(0);
      }
    }
  }
  inline int push_back( MpiTask<Dtype>* task ){
    int list_size = 0;
    lock.lock();
    task_list_.push_back(task);
    list_size = size();
    if(task->tag() == -1){
      task->set_tag(list_size - 1);
    }
    lock.unlock();
    return list_size - 1;
  }
  inline void pop_back(){
    lock.lock();
    task_list_.pop_back(); 
    lock.unlock();
  }
  inline void erase(int idx){
    lock.lock();
    task_list_.erase(task_list_.begin()+idx); 
    lock.unlock();
  }

  inline int size() {
    usleep(1); 
    return task_list_.size(); 
  }
  inline void wait_task(int tag) {
    while(1) {
      int exist = 0;
      for(int i = 0; i < size(); i++) {
        if(task_list_[i]->tag() == tag) exist = 1;
      }
      if(exist == 0) return;
    }
  }
  inline void wait_all_task() {
    while(1) {
      if(size() == 0) return;
    }
  }
 private:
  vector<MpiTask<Dtype> *> task_list_;
  boost::mutex lock;
};

}  // namespace caffe

#endif  // CAFFE_MPITASK_HPP_
