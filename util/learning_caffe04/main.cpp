#include <caffe/util/signal_handler.h>
#include <caffe/caffe.hpp>
#include <caffe/util/db.hpp>
#include <caffe/util/format.hpp>
#include <iostream>
#include <memory>

using namespace std;
using namespace caffe;

template <typename Dtype>
class SolveCallback : Solver<Dtype>::Callback {
   public:
    SolveCallback(std::shared_ptr<Solver<Dtype>> solver) : solver(solver) {}

   protected:
    virtual void on_start() override {
        auto nets = solver->net();
    }

    virtual void on_gradients_ready() override {}

   private:
    std::shared_ptr<Solver<float>> solver;
};

int main(int argc, char** argv) {
    SolverParameter solverParam;
    ReadSolverParamsFromTextFileOrDie(argv[1], &solverParam);

    SignalHandler signalHandler(SolverAction::STOP, SolverAction::SNAPSHOT);

    std::shared_ptr<Solver<float>> solver(
        SolverRegistry<float>::CreateSolver(solverParam));
    solver->SetActionFunction(signalHandler.GetActionFunction());

    unique_ptr<SolveCallback<float>> callback(new SolveCallback<float>(solver));

    cout << "Hello world." << endl;

    return 0;
}
