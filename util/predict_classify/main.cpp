#include <yaml-cpp/yaml.h>
#include <caffe/caffe.hpp>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>

using namespace std;
using namespace caffe;

vector<string> getFileList(const string& path) {
    vector<cv::String> ret;
    cv::glob(path, ret, true);
    vector<string> ret2(ret.begin(), ret.end());
    std::random_shuffle(ret2.begin(), ret2.end());
    return ret2;
}

void wrapInputLayer(vector<cv::Mat>* inputChannels, caffe::Net<float>* net) {
    Blob<float>* inputLayer = net->input_blobs()[0];
    int width = inputLayer->width();
    int height = inputLayer->height();
    float* inputData = inputLayer->mutable_cpu_data();
    for (int i = 0; i < inputLayer->channels(); i++) {
        cv::Mat channel(height, width, CV_32FC1, inputData);
        inputChannels->push_back(channel);
        inputData += width * height;
    }
}

vector<string> getLabels(const string& labelFile) {
    ifstream iFile(labelFile);
    vector<string> labels;
    string buf;
    while(std::getline(iFile, buf)) {
        labels.push_back(buf);
    }
    return labels;
}

int main(int argc, char** argv) {
    caffe::GlobalInit(&argc, &argv);
    google::LogToStderr();

    YAML::Node setting = YAML::LoadFile(argv[1]);
    string modelSetting = setting["ModelSetting"].as<string>();
    string modelFile = setting["ModelFile"].as<string>();
    string labelFile = setting["LabelFile"].as<string>();
    string fileList = setting["FileList"].as<string>();
    Caffe::set_mode(Caffe::Brew::GPU);
    Net<float> net(modelSetting, caffe::Phase::TEST);
    net.CopyTrainedLayersFrom(modelFile);
    cout << net.num_inputs() << endl;
    cout << net.num_outputs() << endl;

    Blob<float>* inputLayer = net.input_blobs()[0];
    int numChannels = inputLayer->channels();
    cout << "channel: " << numChannels << endl;
    cout << "nums: " << inputLayer->num() << endl;

    cout << "input shape: ";
    auto shape = inputLayer->shape();
    for (auto& i : shape) {
        cout << i << ", ";
    }
    cout << endl;

    Blob<float>* outputLayer = net.output_blobs()[0];
    cout << "output shape: " << outputLayer->shape_string() << endl;
    cout << "output shape: ";
    for (auto& i : outputLayer->shape()) {
        cout << i << ", ";
    }
    cout << endl;

    auto list = getFileList(fileList);
    auto labelList = getLabels(labelFile);

    for (auto& i : list) {
        printf("read file: %s\n", i.c_str());
        cv::Mat img = cv::imread(i);
        //system(("eog " + i + " &").c_str());
        cout << "start predict" << endl;
        vector<cv::Mat> inputChannels;
        wrapInputLayer(&inputChannels, &net);
        cv::Mat img_float;
        img.convertTo(img_float, CV_32FC3, 1./256.);
        cv::split(img_float, inputChannels);

        net.Forward();
        const float* begin = outputLayer->cpu_data();
        const float* end = begin + outputLayer->channels();
        vector<float> output(begin, end);
        cout << "output: ";
        for (auto& f : output) {
            printf("%.1f, ", f);
        }
        cout << endl;
        int maxIndex = max_element(output.begin(), output.end()) - output.begin();
        cout << "Label: " << labelList[maxIndex] << endl;
        cv::imshow("Prediction", img);
        int input = cv::waitKeyEx();
        if (input == 27) {
            break;
        }
        //string input;
        //cin >> input;
    }


    return 0;
}
