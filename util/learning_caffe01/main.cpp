#include <iostream>
#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include <caffe/caffe.hpp>
#include "print_vec.h"

using namespace std;
using namespace caffe;
using google::protobuf::TextFormat;

const string protoContext = R"foo(
name: "My network"
state {
    phase: TEST
}
layer {
    name: "data"
    type: "Input"
    top: "data"
    input_param {
        shape: {dim: 3 dim: 5}
    }
}
layer {
    name: "bias0"
    type: "Bias"
    top: "bias0"
    bottom: "data"
    bias_param {
        filler {
            value: 10
        }
    }
}
)foo";

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    Caffe::set_mode(Caffe::Brew::CPU);
    NetParameter netParam;
    TextFormat::ParseFromString(protoContext, &netParam);
    Net<float> net(netParam);
    cout << "Load success" << endl;

    cout << "net.num_inputs()" << net.num_inputs() << endl;
    auto& inputData = *net.input_blobs()[0];
    cout << "inputData.shape()" << inputData.shape() << endl;

    cout << "net.num_outputs()" << net.num_outputs() << endl;
    auto& outData = *net.output_blobs()[0];
    cout << "outData.shape()" << outData.shape() << endl;

    float* inputRaw = inputData.mutable_cpu_data();
    for (int i = 0; i < inputData.shape(0); ++i) {
        for (int j = 0; j < inputData.shape(1); ++j) {
            *inputRaw = i * 10 + j;
            ++inputRaw;
        }
    }
    net.Forward();

    float* outRaw = outData.mutable_cpu_data();
    cout << "Out data: [";
    for (int i = 0; i < outData.shape(0); ++i) {
        if (i) cout << ", ";
        cout << "[";
        for (int j = 0; j < outData.shape(1); ++j) {
            if (j) cout << ", ";
            cout << *(outRaw + i * outData.shape(1) + j);
        }
        cout << "]";
    }
    cout << "]" << endl;

	return 0;
}
