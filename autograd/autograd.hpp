#pragma once

namespace autograd {
    template <typename T>
    class Tensor {
    public:
        explicit Tensor(T * data):_data(data) {}


        T get(int i) {
            return *(_data + i);
        }
    
    private:
        T *_data;
    };
}