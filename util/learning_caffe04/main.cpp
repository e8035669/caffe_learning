#include <caffe/util/signal_handler.h>
#include <glog/stl_logging.h>
#include <caffe/caffe.hpp>
#include <caffe/util/db.hpp>
#include <caffe/util/format.hpp>
#include <iostream>
#include <memory>
#include "iris_dataset.h"

using namespace std;
using namespace caffe;

class SolveCallback : public Solver<float>::Callback {
   public:
    SolveCallback(std::shared_ptr<Solver<float>> solver)
        : solver(solver), input(solver->net()->input_blobs()), indexCounter(0) {
        LOG(INFO) << "input.size()" << input.size();
        this->num = input[0]->num();
    }

   protected:
    virtual void on_start() override {
        float* data = input[0]->mutable_cpu_data();
        float* label = input[1]->mutable_cpu_data();
        for (int i = 0; i < this->num; ++i) {
            //caffe_copy(4, IRIS_DATASET + indexCounter * 4, data + i * 4);
            for (int j = 0; j < 4; j++) {
                data[i * 4 + j] = IRIS_DATASET[indexCounter * 4 + j];
            }
            label[i] = IRIS_TRUTH[indexCounter];
            nextData();
        }
    }

    virtual void on_gradients_ready() override {}

   private:
    std::shared_ptr<Solver<float>> solver;
    const vector<Blob<float>*>& input;
    int indexCounter;
    int num;

    void nextData() {
        indexCounter = (indexCounter + 1) % 150;
    }
};

int main(int argc, char** argv) {
    FLAGS_logtostderr = true;
    caffe::GlobalInit(&argc, &argv);
    Caffe::set_mode(Caffe::CPU);

    SolverParameter solverParam;
    ReadSolverParamsFromTextFileOrDie(argv[1], &solverParam);

    SignalHandler signalHandler(SolverAction::STOP, SolverAction::SNAPSHOT);

    std::shared_ptr<Solver<float>> solver(
        SolverRegistry<float>::CreateSolver(solverParam));
    solver->SetActionFunction(signalHandler.GetActionFunction());

    unique_ptr<SolveCallback> callback(new SolveCallback(solver));
    solver->add_callback(callback.get());

    solver->Solve();

    LOG(INFO) << "Optimization Done.";

    return 0;
}
