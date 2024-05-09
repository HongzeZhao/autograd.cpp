#pragma once
#include <ostream>
#include "shape.hpp"
#include "allocator.hpp"


namespace autograd {
    template <typename DataType, typename AllocatorType=Allocator<DataType>>
    class Tensor {
    public:
        explicit Tensor(Shape shape): _shape(shape) {
            int len = shape.numel();
            _data = AllocatorType::alloc(len);
        }

        // shallow copy
        Tensor(DataType *data, Shape shape): _shape(shape) {
            _data = data;
        }

        const Shape& shape() const {
            return _shape;
        }

        /**
        * a view of the original tensor
        */
        Tensor<DataType> operator[](int i) const {
            Shape shape;
            for (int j = 0; j < _shape.dim()-1; j++) {
                shape.push_back(_shape[j+1]);
            }
            int offset = i * shape.numel();
            return Tensor<DataType>(_data + offset, shape);
        }

        DataType operator()(int i) const {
            assert(_shape.dim() == 1);
            return *(_data + i);
        }

        DataType operator()(int i, int j) const {
            assert(_shape.dim() == 2);
            int n1 = _shape[1];
            return *(_data + i * n1 + j);
        }

        DataType operator()(int i, int j, int k) const {
            assert(_shape.dim() == 3);
            int n1 = _shape[1];
            int n2 = _shape[2];
            return *(_data + i * n1 + j*n2 + k);
        }

        DataType operator()(int i, int j, int k, int l) const {
            assert(_shape.dim() == 4);
            int n1 = _shape[1];
            int n2 = _shape[2];
            int n3 = _shape[3];
            return *(_data + i * n1 + j*n2 + k*n3 + l);
        }

        DataType& operator()(int i) {
            assert(_shape.dim() == 1);
            return *(_data + i);
        }

        DataType& operator()(int i, int j) {
            assert(_shape.dim() == 2);
            int n1 = _shape[1];
            return *(_data + i * n1 + j);
        }

        DataType& operator()(int i, int j, int k) {
            assert(_shape.dim() == 3);
            int n1 = _shape[1];
            int n2 = _shape[2];
            return *(_data + i * n1 + j*n2 + k);
        }

        DataType& operator()(int i, int j, int k, int l) {
            assert(_shape.dim() == 4);
            int n1 = _shape[1];
            int n2 = _shape[2];
            int n3 = _shape[3];
            return *(_data + i * n1 + j*n2 + k*n3 + l);
        }

        Tensor operator+(const Tensor &rhs) const {
            assert(_shape == rhs.shape());
            Tensor ret(_shape);
            int numel = _shape.numel();
            for (int i = 0; i < numel; i++) {
                ret._data[i] = _data[i] + rhs._data[i];
            }
            return ret;
        }

    private:
        DataType *_data;
        Shape _shape;
    };

    typedef Tensor<float> Tensorf;

    template <typename DataType, typename AllocatorType=Allocator<DataType>>
    std::ostream& operator<<(std::ostream& os, const autograd::Tensor<DataType, AllocatorType>& t) {
        if (t.shape().dim() <= 1) {
            int len = t.shape()[0];
            os << "[";
            for (int i = 0; i < len; i++) {
                os << t(i);
                if (i != len - 1) os << ",";
            }
            os << "]";
        } else {
            os << "[" << std::endl;
            int len = t.shape()[0];
            for (int i = 0; i < len; i++) {
                os << t[i] << std::endl;
            }
            os << "]";
        }
        return os;
    }
}