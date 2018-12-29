#include <iostream>
#include <memory>
#include <caffe/caffe.hpp>
#include <caffe/util/db.hpp>
#include <caffe/util/io.hpp>
#include <caffe/util/format.hpp>

using namespace std;
using namespace caffe;

int main(int argc, char** argv) {
    FLAGS_logtostderr = 1;
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    LOG(INFO) << "This project will list all data in lmdb";

    unique_ptr<db::DB> db(db::GetDB("lmdb"));
    db->Open(argv[1], db::READ);
    unique_ptr<db::Cursor> cursor(db->NewCursor());

    Datum datum;
    while (cursor->valid()) {
        cout << "------------------------------" << endl;
        cout << cursor->key() << ": " << cursor->value().size() << " bytes." << endl;
        datum.ParseFromString(cursor->value());
        if (datum.has_channels()) {
            cout << "Channel: " << datum.channels() << endl;
        }
        if (datum.has_height()) {
            cout << "Height: " << datum.height() << endl;
        }
        if (datum.has_width()) {
            cout << "Width: " << datum.width() << endl;
        }
        if (datum.has_data()) {
            cout << "Data: " << datum.data().size() << " bytes." << endl;
        }
        if (datum.has_label()) {
            cout << "Label: " << datum.label() << endl;
        }
        if (datum.has_encoded()) {
            cout << "Encoded: " << boolalpha << datum.encoded() << endl;
        }
        cursor->Next();
    }

    return 0;
}
