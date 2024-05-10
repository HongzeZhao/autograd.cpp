#pragma once

namespace autograd {

template <typename T>
class Allocator {
public:
    static T * alloc(int n) {
        return new T[n]{0};
    }

    static void free(T *p) {
        delete [] p;
    }
};

}