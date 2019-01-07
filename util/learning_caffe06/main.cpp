#include <caffe/util/signal_handler.h>
#include <glog/stl_logging.h>
#include <caffe/caffe.hpp>
#include <caffe/util/db.hpp>
#include <caffe/util/format.hpp>
#include <iostream>
#include <memory>
#include <google/protobuf/text_format.h>
#include "iris_dataset.h"

using namespace std;
using namespace caffe;
using google::protobuf::TextFormat;

int main(int argc, char** argv) {
    FLAGS_logtostderr = true;
    caffe::GlobalInit(&argc, &argv);
    Caffe::set_mode(Caffe::CPU);

    LOG(INFO) << "This script will dump a caffemodel into a text file.";
    LOG(INFO) << "So that we can see what's actually in the model file.";

    if (argc < 3) {
        cout << "\nUsage: " << argv[0] << " xxx.caffemodel output.prototxt" << endl;
        return -1;
    }

    NetParameter netParameter;
    ReadNetParamsFromBinaryFileOrDie(argv[1], &netParameter);

    string s;
    TextFormat::PrintToString(netParameter, &s);

    ofstream(argv[2]) << s;

    LOG(INFO) << "Done.";

    return 0;
}
