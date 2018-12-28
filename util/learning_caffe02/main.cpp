#include <iostream>
#include <caffe/caffe.hpp>

using namespace std;
using namespace caffe;

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    Caffe::set_mode(Caffe::Brew::CPU);



    cout << "Hello World" << endl;


    return 0;
}
