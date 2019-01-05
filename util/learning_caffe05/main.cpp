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

int main(int argc, char** argv) {
    FLAGS_logtostderr = true;
    caffe::GlobalInit(&argc, &argv);
    Caffe::set_mode(Caffe::CPU);

    NetParameter netPatameter;
    ReadNetParamsFromTextFileOrDie(argv[1], &netPatameter);

    unique_ptr<Net<float>> net(new Net<float>(netPatameter));

    net->CopyTrainedLayersFromBinaryProto(argv[2]);

    Blob<float>& inputBlob = *net->input_blobs()[0];
    Blob<float>& outputBlob = *net->output_blobs()[0];
    CHECK_EQ(inputBlob.num(), 1) << "batch is not 1";
    CHECK_EQ(outputBlob.num(), 1) << "batch is not 1";
    printf("Input Blob Shape: [%d %d %d %d]\n",
            inputBlob.num(), inputBlob.channels(),
            inputBlob.height(), inputBlob.width());
    printf("Output Blob Shape: [%d %d %d %d]\n",
            outputBlob.num(), outputBlob.channels(),
            outputBlob.height(), outputBlob.width());

    int hit = 0;
    for (int i = 0; i < 150; i++) {
        float* dataIdx = IRIS_DATASET + i * 4;
        int truth = IRIS_TRUTH[i];
        for (int j = 0; j < 4; j++) {
            inputBlob.mutable_cpu_data()[j] = dataIdx[j];
        }

        net->Forward();

        float* out = outputBlob.mutable_cpu_data();
        printf("out: [%.2f %.2f %.2f] ",
                out[0], out[1], out[2]);
        printf("truth: %d\n", truth);
        if (max_element(out, out+3)-out == truth) {
            hit++;
        }
    }

    LOG(INFO) << "Hit: " << hit;
    LOG(INFO) << "Accuracy: " << hit / 150.;

    return 0;
}
