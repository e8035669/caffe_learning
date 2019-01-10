#include <caffe/util/signal_handler.h>
#include <glog/stl_logging.h>
#include <caffe/caffe.hpp>
#include <caffe/util/db.hpp>
#include <caffe/util/format.hpp>
#include <caffe/layers/conv_layer.hpp>
#include <iostream>
#include <memory>
#include <google/protobuf/text_format.h>

using namespace std;
using namespace caffe;
using google::protobuf::TextFormat;

string layerDescript = R"foo(
name: "conv"
type: "Convolution"
blobs {
    data: 2 data: 2 data: 1
    data: 1 data: 2 data: 3
    data: 3 data: 2 data: 1

    data: 2 data: 1 data: 3
    data: 1 data: 3 data: 2
    data: 2 data: 3 data: 1

    data: 1 data: 3 data: 3
    data: 1 data: 1 data: 2
    data: 3 data: 1 data: 3

    data: 2 data: 3 data: 1
    data: 3 data: 1 data: 2
    data: 1 data: 2 data: 3

    data: 3 data: 1 data: 2
    data: 2 data: 1 data: 3
    data: 3 data: 2 data: 1

    data: 2 data: 1 data: 3
    data: 3 data: 1 data: 2
    data: 3 data: 2 data: 1

    shape {dim: 2 dim: 3 dim: 3 dim: 3}
}
blobs {
    data: 1 data: 2
    shape {dim: 2}
}
convolution_param {
    num_output: 2
    pad: 1
    kernel_size: 3
    stride: 1
}
)foo";



int main(int argc, char** argv) {
    FLAGS_logtostderr = true;
    caffe::GlobalInit(&argc, &argv);
    Caffe::set_mode(Caffe::CPU);

    LOG(INFO) << "This util will make a simple convolution test";

    LayerParameter layerParam;
    //layerParam.ParseFromString(layerDescript);
    TextFormat::ParseFromString(layerDescript, &layerParam);

    ConvolutionLayer<float> convLayer(layerParam);

    vector<Blob<float>*> input;
    vector<Blob<float>*> output;
    Blob<float> inputBlob(1, 3, 3, 5);
    Blob<float> outputBlob;
    input.push_back(&inputBlob);
    output.push_back(&outputBlob);

    convLayer.SetUp(input, output);

    auto& convBlobs = convLayer.blobs();
    for (size_t i = 0; i < convBlobs.size(); ++i) {
        LOG(INFO) << "Convolution Blobs " << i << ": ";
        LOG(INFO) << convBlobs[i]->shape_string();
    }

    LOG(INFO) << "Input Blob: " << inputBlob.shape_string();
    LOG(INFO) << "Output Blob: " << outputBlob.shape_string();

    float inputValues[] = {
        // a1
        1, 2, 3, 2, 1,
        3, 2, 1, 2, 3,
        2, 3, 1, 3, 2,

        // a2
        2, 1, 3, 2, 1,
        1, 3, 2, 1, 3,
        3, 1, 2, 1, 2,

        // a3
        1, 3, 2, 3, 1,
        2, 1, 3, 2, 3,
        1, 3, 2, 1, 3
    };
    std::copy_n(inputValues, 45, inputBlob.mutable_cpu_data());

    convLayer.Forward(input, output);

    LOG(INFO) << "Output Data: ";
    for (int i = 0; i < outputBlob.channels(); i++) {
        for (int j = 0; j < outputBlob.height(); j++) {
            for (int k = 0; k < outputBlob.width(); k++) {
                float val = outputBlob.data_at(0, i, j, k);
                printf("%-5.0f", val);
            }
            cout << endl;
        }
        cout << endl;
    }

    LOG(INFO) << "Done.";

    return 0;
}
