#ifndef PRINT_VEC_H
#define PRINT_VEC_H

#include <vector>
#include <ostream>

std::ostream& operator<<(std::ostream& os, std::vector<int> vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i) {
            os << ", ";
        }
        os << vec[i];
    }
    os << "]";
    return os;
}


#endif
