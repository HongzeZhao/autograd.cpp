#pragma once
#include <vector>
#include "tensor.hpp"

namespace autograd {

template <typename ValueType>
class Layer {
public:
    typedef Tensor<ValueType> TensorType;
    typedef typename Tensor<ValueType>::ValueMapFunc ActivateFunc;

public:
    Layer() = default;
    virtual ~Layer() {}
    virtual TensorType forward(const TensorType &input) = 0;
    virtual TensorType backward(const TensorType &input) = 0;
};


template <typename ValueType>
class DenseLayer : public Layer<ValueType> {
public:
    typedef typename Layer<ValueType>::TensorType TensorType;
    typedef typename Layer<ValueType>::ActivateFunc ActivateFunc;
    typedef std::vector<int> HiddenShape;

    DenseLayer(const HiddenShape &hidden_dims, const ActivateFunc &hidden_activator):
        _dims(hidden_dims), _activator(hidden_activator) {}

    TensorType forward(const TensorType &input) override {
        return {};
    }

    TensorType backward(const TensorType &input) override {
        return {};
    }

private:
    HiddenShape _dims;
    ActivateFunc _activator;
};

}