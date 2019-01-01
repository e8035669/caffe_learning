#include <iostream>
#include <memory>
#include <caffe/caffe.hpp>
#include <caffe/util/db.hpp>
#include <caffe/util/io.hpp>
#include <caffe/util/format.hpp>
#include "iris_dataset.h"

using namespace std;
using namespace caffe;

int main(int argc, char** argv) {
    FLAGS_logtostderr = true;
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    LOG(INFO) << "This Script will write iris dataset into lmdb.";

    unique_ptr<db::DB> db(db::GetDB("lmdb"));
    db->Open("iris_lmdb", db::NEW);
    unique_ptr<db::Transaction> txn(db->NewTransaction());

    Datum datum;
    datum.set_channels(1);
    datum.set_width(4);
    datum.set_height(1);
    for (int i = 0; i < 150; i++) {
        datum.clear_label();
        datum.clear_float_data();

        for (int j = 0; j < 4; j++) {
            datum.add_float_data(IRIS_DATASET[4 * i + j]);
        }
        datum.set_label(IRIS_TRUTH[i]);
        string val;
        datum.SerializeToString(&val);
        string key = caffe::format_int(i, 4);
        txn->Put(key, val);
    }
    txn->Commit();

    return 0;
}
