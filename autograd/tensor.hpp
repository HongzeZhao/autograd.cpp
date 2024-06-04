#pragma once
#include <algorithm>
#include <cstdio>
#include <ostream>
#include <random>
#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <unordered_set>
#include <ctime>
#include <limits>
#include "shape.hpp"
#include "allocator.hpp"

namespace autograd {
    typedef std::unordered_set<int> Axis;
    typedef std::vector<int> AxisPerm;
    typedef std::vector<int> Index;

    static std::random_device _g_random_device;
    static std::default_random_engine _g_random_generator(_g_random_device());

    template <typename ValueType, typename AllocatorType=Allocator<ValueType>>
    class Tensor {
    public:
        typedef std::function<void(ValueType&,ValueType)> ReducerFunc;
        typedef std::function<ValueType(ValueType)> ValueMapFunc;

    public:
        // init with given shape
        explicit Tensor(const Shape &shape): _shape(shape), _view(false) {
            this->_data = AllocatorType::alloc(_shape.numel());
        }

        // init with all elements filled by static value
        Tensor(const Shape &shape, ValueType value): _shape(shape), _view(false) {
            int n = _shape.numel();
            this->_data = AllocatorType::alloc(n);
            fill(value);
        }

        // deep copy constructor
        Tensor(const Tensor &rhs): _view(rhs._view) {
            if (_view) {
                this->_data = rhs._data;
            } else {
                *this->_data = *rhs._data;
            }
            this->_shape = rhs._shape;
        }

        // move constructor
        Tensor(Tensor &&rhs): _view(rhs._view), _data(rhs._data), _shape(std::move(rhs._shape)) {
            rhs._view = true;
        }

        // deep copy assign
        Tensor & operator=(const Tensor &rhs) {
            this->_view = rhs._view;
            if (_view) {
                this->_data = rhs._data;
            } else {
                *this->_data = *rhs._data;
            }
            this->_shape = rhs._shape;
            return *this;
        }

        // move assign
        Tensor & operator=(const Tensor &&rhs) {
            this->_view = rhs._view;
            this->_data = rhs._data;
            this->_shape = std::move(rhs._shape);
            rhs._view = true;
            return *this;
        }

        // shallow copy
        Tensor(ValueType *data, const Shape &shape): _shape(shape), _view(true) {
            this->_data = data;
        }

        // init with 1-dimension initializer_list
        Tensor(const std::initializer_list<ValueType>& init_list): _view(false) {
            _shape.push_back(init_list.size());
            this->_data = AllocatorType::alloc(_shape.numel());
            ValueType *p = this->_data;
            for (ValueType v : init_list) {
                (*p++) = v;
            }
        }

        // init with 2-dimension initializer_list
        Tensor(const std::initializer_list<std::vector<ValueType>>& init_list): _view(false) {
            _shape.push_back(init_list.size());
            if (init_list.size() > 0) {
                _shape.push_back(init_list.begin()->size());
            }
            this->_data = AllocatorType::alloc(_shape.numel());
            ValueType *p = this->_data;
            for (const auto &il : init_list) {
                for (ValueType v : il) (*p++) = v;
            }
        }

        // init with 3-dimension initializer_list
        Tensor(const std::initializer_list<std::vector<std::vector<ValueType>>>& init_list): _view(false) {
            _shape.push_back(init_list.size());
            if (init_list.size() > 0) {
                _shape.push_back(init_list.begin()->size());
                if (init_list.begin()->size() > 0) {
                    _shape.push_back(init_list.begin()->begin()->size());
                }
            }
            this->_data = AllocatorType::alloc(_shape.numel());
            ValueType *p = this->_data;
            for (const auto &il : init_list) {
                for (const auto &ill : il) {
                    for (ValueType v : ill) (*p++) = v;
                }
            }
        }

        ~Tensor() {
            if (!_view) {
                if (_data != nullptr) {
                    AllocatorType::free(_data);
                    _data = nullptr;
                }
            }
        }

        /**
        * get tensor shape
        */
        const Shape& shape() const {
            return _shape;
        }

        /**
        * Reshapes a tensor.
        * If one component of shape is the special value -1, the size of that dimension is computed
        * so that the total size remains constant. In particular, a shape of [-1] flattens into 1-D.
        * At most one component of shape can be -1.
        */
        Tensor &reshape(const Shape &shape) {
            _shape = shape;
            int cur_numel = _shape.numel();
            int numel = shape.numel();
            if (numel != cur_numel) {
                assert(numel < 0);
                int derive_dim = -1;
                for (int d = 0; d < shape.dim(); d++) {
                    if (shape[d] < 0) {
                        derive_dim = d;
                    } else {
                        assert(cur_numel % shape[d] == 0);
                        cur_numel /= shape[d];
                    }
                }
                assert(derive_dim >= 0);
                assert(cur_numel > 0);
                _shape[derive_dim] = cur_numel;
            }
            return *this;
        }

        /**
        * fill all element with a single value
        */
        Tensor &fill(ValueType val) {
            int len = _shape.numel();
            for (int i = 0; i < len; i++) {
                this->_data[i] = val;
            }
            return *this;
        }

        /**
        * fill all elements by uniform gaussian with range of mean and stddev
        */
        Tensor &fillGaussianRandom(ValueType mean, ValueType stddev) {
            std::normal_distribution<ValueType> distribution(mean, stddev);
            int len = _shape.numel();
            for (int i = 0; i < len; i++) {
                this->_data[i] = distribution(_g_random_generator);
            }
            return *this;
        }

        /**
        * fill all elements by uniform random with range of [low, high]
        */
        Tensor &fillUniformRandom(ValueType low, ValueType high) {
            std::uniform_real_distribution<ValueType> distribution(low, high);
            int len = _shape.numel();
            for (int i = 0; i < len; i++) {
                this->_data[i] = distribution(_g_random_generator);
            }
            return *this;
        }

        /**
        * a view of the original tensor
        */
        Tensor<ValueType> operator[](int i) const {
            Shape shape;
            for (int j = 0; j < _shape.dim()-1; j++) {
                shape.push_back(_shape[j+1]);
            }
            int offset = i * shape.numel();
            return Tensor<ValueType>(_data + offset, shape);
        }

        /**
        * element accessor with a index vector.
        */
        ValueType operator()(const ValueMapFunc &mapper) const {
            Tensor ret(_shape);
            int numel = _shape.numel();
            for (int i = 0; i < numel; i++) {
                ret._data[i] = mapper(_data[i]);
            }
            return ret;
        }

        /**
        * element accessor with a index vector.
        */
        ValueType& operator()(const ValueMapFunc &mapper) {
            int numel = _shape.numel();
            for (int i = 0; i < numel; i++) {
                _data[i] = mapper(_data[i]);
            }
            return *this;
        }

        /**
        * const element accessor with a index vector.
        */
        ValueType operator()(const std::vector<int> &index) const {
            int n = _shape.numel();
            int pos = 0;
            for (int d = 0; d < _shape.dim(); d++) {
                n /= _shape[d];
                pos += index[d] * n;
            }
            return *(_data + pos);
        }

        /**
        * element accessor with a index vector.
        */
        ValueType& operator()(const std::vector<int> &index) {
            int n = _shape.numel();
            int pos = 0;
            for (int d = 0; d < _shape.dim(); d++) {
                n /= _shape[d];
                pos += index[d] * n;
            }
            return *(_data + pos);
        }

        /**
        * const element accessor for a tensor with shape(A)
        */
        ValueType operator()(int i) const {
            assert(_shape.dim() == 1);
            return *(_data + i);
        }

        /**
        * const element accessor for a tensor with shape(A,B)
        */
        ValueType operator()(int i, int j) const {
            assert(_shape.dim() == 2);
            int n1 = _shape[1];
            return *(_data + i * n1 + j);
        }

        /**
        * const element accessor for a tensor with shape(A,B,C)
        */
        ValueType operator()(int i, int j, int k) const {
            assert(_shape.dim() == 3);
            int n1 = _shape[1];
            int n2 = _shape[2];
            return *(_data + i * n1 + j*n2 + k);
        }

        /**
        * const element accessor for a tensor with shape(A,B,C,D)
        */
        ValueType operator()(int i, int j, int k, int l) const {
            assert(_shape.dim() == 4);
            int n1 = _shape[1];
            int n2 = _shape[2];
            int n3 = _shape[3];
            return *(_data + i * n1 + j*n2 + k*n3 + l);
        }

        /**
        * element accessor for a tensor with shape(A)
        */
        ValueType& operator()(int i) {
            assert(_shape.dim() == 1);
            return *(_data + i);
        }

        /**
        * element accessor for a tensor with shape(A,B)
        */
        ValueType& operator()(int i, int j) {
            assert(_shape.dim() == 2);
            int n1 = _shape[1];
            return *(_data + i * n1 + j);
        }

        /**
        * element accessor for a tensor with shape(A,B,C)
        */
        ValueType& operator()(int i, int j, int k) {
            assert(_shape.dim() == 3);
            int n1 = _shape[1];
            int n2 = _shape[2];
            return *(_data + i * n1 + j*n2 + k);
        }

        /**
        * element accessor for a tensor with shape(A,B,C,D)
        */
        ValueType& operator()(int i, int j, int k, int l) {
            assert(_shape.dim() == 4);
            int n1 = _shape[1];
            int n2 = _shape[2];
            int n3 = _shape[3];
            return *(_data + i * n1 + j*n2 + k*n3 + l);
        }

        /**
        * element-wise operator +
        */
        Tensor operator+(const Tensor &rhs) const {
            assert(_shape == rhs.shape());
            Tensor ret(_shape);
            int numel = _shape.numel();
            for (int i = 0; i < numel; i++) {
                ret._data[i] = _data[i] + rhs._data[i];
            }
            return ret;
        }

        /**
        * element-wise operator +
        */
        Tensor operator+(ValueType val) const {
            Tensor ret(_shape);
            int numel = _shape.numel();
            for (int i = 0; i < numel; i++) {
                ret._data[i] = _data[i] + val;
            }
            return ret;
        }

        /**
        * element-wise operator -
        */
        Tensor operator-(const Tensor &rhs) const {
            assert(_shape == rhs.shape());
            Tensor ret(_shape);
            int numel = _shape.numel();
            for (int i = 0; i < numel; i++) {
                ret._data[i] = _data[i] - rhs._data[i];
            }
            return ret;
        }

        /**
        * element-wise operator -
        */
        Tensor operator-(ValueType val) const {
            Tensor ret(_shape);
            int numel = _shape.numel();
            for (int i = 0; i < numel; i++) {
                ret._data[i] = _data[i] - val;
            }
            return ret;
        }

        /**
        * element-wise operator negative
        */
        Tensor operator-() const {
            Tensor ret(_shape);
            int numel = _shape.numel();
            for (int i = 0; i < numel; i++) {
                ret._data[i] = -_data[i];
            }
            return ret;
        }

        /**
        * element-wise operator *
        */
        Tensor operator*(const Tensor &rhs) const {
            assert(_shape == rhs.shape());
            Tensor ret(_shape);
            int numel = _shape.numel();
            for (int i = 0; i < numel; i++) {
                ret._data[i] = _data[i] * rhs._data[i];
            }
            return ret;
        }

        /**
        * element-wise operator *
        */
        Tensor operator*(ValueType a) const {
            Tensor ret(_shape);
            int numel = _shape.numel();
            for (int i = 0; i < numel; i++) {
                ret._data[i] = _data[i] * a;
            }
            return ret;
        }

        /**
        * element-wise operator /
        */
        Tensor operator/(const Tensor &rhs) const {
            assert(_shape == rhs.shape());
            Tensor ret(_shape);
            int numel = _shape.numel();
            for (int i = 0; i < numel; i++) {
                ret._data[i] = _data[i] / rhs._data[i];
            }
            return ret;
        }

        /**
        * element-wise operator /
        */
        Tensor operator/(ValueType a) const {
            assert(a != 0);
            Tensor ret(_shape);
            int numel = _shape.numel();
            for (int i = 0; i < numel; i++) {
                ret._data[i] = _data[i] / a;
            }
            return ret;
        }

        /**
        * element-wise operator power
        */
        Tensor operator^(ValueType exp) const {
            Tensor ret(_shape);
            int numel = _shape.numel();
            for (int i = 0; i < numel; i++) {
                ret._data[i] = std::powf(_data[i], exp);
            }
            return ret;
        }

        /**
        * element-wise equal check
        */
        bool operator==(const Tensor &rhs) const {
            if (_shape != rhs.shape()) return false;
            for (int i = 0; i < _shape.numel(); i++) {
                if (_data[i] != rhs._data[i]) return false;
            }
            return true;
        }

        /**
        * element-wise non-equal check
        */
        bool operator!=(const Tensor &rhs) const {
            if (_shape != rhs.shape()) return true;
            for (int i = 0; i < _shape.numel(); i++) {
                if (_data[i] != rhs._data[i]) return true;
            }
            return false;
        }

        /**
        * element-wise operator +=
        */
        Tensor &operator+=(const Tensor &rhs) {
            assert(_shape == rhs.shape());
            int numel = _shape.numel();
            for (int i = 0; i < numel; i++) {
                _data[i] += rhs._data[i];
            }
            return *this;
        }

        /**
        * element-wise operator -=
        */
        Tensor &operator-=(const Tensor &rhs) {
            assert(_shape == rhs.shape());
            int numel = _shape.numel();
            for (int i = 0; i < numel; i++) {
                _data[i] -= rhs._data[i];
            }
            return *this;
        }

        /**
        * element-wise operator *=
        */
        Tensor &operator*=(const Tensor &rhs) {
            assert(_shape == rhs.shape());
            int numel = _shape.numel();
            for (int i = 0; i < numel; i++) {
                _data[i] *= rhs._data[i];
            }
            return *this;
        }

        /**
        * element-wise operator /=
        */
        Tensor &operator/=(const Tensor &rhs) {
            assert(_shape == rhs.shape());
            int numel = _shape.numel();
            for (int i = 0; i < numel; i++) {
                _data[i] /= rhs._data[i];
            }
            return *this;
        }

        /**
        * element-wise operator +=
        */
        Tensor &operator+=(ValueType val) {
            int numel = _shape.numel();
            for (int i = 0; i < numel; i++) {
                _data[i] += val;
            }
            return *this;
        }

        /**
        * element-wise operator -=
        */
        Tensor &operator-=(ValueType val) {
            int numel = _shape.numel();
            for (int i = 0; i < numel; i++) {
                _data[i] -= val;
            }
            return *this;
        }

        /**
        * element-wise operator *=
        */
        Tensor &operator*=(ValueType val) {
            int numel = _shape.numel();
            for (int i = 0; i < numel; i++) {
                _data[i] *= val;
            }
            return *this;
        }

        /**
        * element-wise operator /=
        */
        Tensor &operator/=(ValueType val) {
            int numel = _shape.numel();
            for (int i = 0; i < numel; i++) {
                _data[i] /= val;
            }
            return *this;
        }

        /**
        * element-wise operator power
        */
        Tensor &operator^=(ValueType exp) {
            int numel = _shape.numel();
            for (int i = 0; i < numel; i++) {
                _data[i] = std::powf(_data[i], exp);
            }
            return *this;
        }

        /**
        * This is matrix product, not element-wise product.
        * A Tensor of the same type as a and b where each inner-most matrix is the
        * product of the corresponding matrices in a and b, e.g. if all transpose or
        * adjoint attributes are False:
        * output[..., i, j] = sum_k (a[..., i, k] * b[..., k, j]), for all indices i, j.
        */
        Tensor matmul(const Tensor &rhs, bool transpose_b=false) const {
            int dim = _shape.dim();
            assert(dim == rhs.shape().dim());
            assert(dim >= 2);
            Shape shape;
            for (int i = 0; i < dim - 2; i++) {
                assert(_shape[i] == rhs.shape()[i]);
                shape.push_back(_shape[i]);
            }

            if (transpose_b) {
                assert(_shape[dim - 1] == rhs.shape()[dim - 1]);
                shape.push_back(_shape[dim - 2]);
                shape.push_back(rhs.shape()[dim - 2]);
                Tensor ret(shape);
                
                int a = _shape[dim - 2];
                int b = _shape[dim - 1];
                int c = rhs.shape()[dim - 2];

                int block_size_a = a * b;
                int block_size_b = b * c;
                int block_size_ret = a * c;
                int numel = shape.numel();
                ValueType *dst = ret._data;
                ValueType *src_a = this->_data;
                ValueType *src_b = rhs._data;

                for (int i = 0; i < numel; i += block_size_ret) {
                    _matmul(dst, src_a, src_b, a, b, c, true);
                    src_a += block_size_a;
                    src_b += block_size_b;
                    dst += block_size_ret;
                }
                return ret;
            } else {
                assert(_shape[dim - 1] == rhs.shape()[dim - 2]);
                shape.push_back(_shape[dim - 2]);
                shape.push_back(rhs.shape()[dim - 1]);
                Tensor ret(shape);

                int a = _shape[dim - 2];
                int b = _shape[dim - 1];
                int c = rhs.shape()[dim - 1];

                int block_size_a = a * b;
                int block_size_b = b * c;
                int block_size_ret = a * c;
                int numel = shape.numel();
                ValueType *dst = ret._data;
                ValueType *src_a = this->_data;
                ValueType *src_b = rhs._data;
                for (int i = 0; i < numel; i += block_size_ret) {
                    _matmul(dst, src_a, src_b, a, b, c, false);
                    src_a += block_size_a;
                    src_b += block_size_b;
                    dst += block_size_ret;
                }
                return ret;
            }
        }


        /**
        * Permutes the dimensions according to the value of perm.
        * The returned tensor's dimension i will correspond to the input dimension perm[i].
        * If perm is not given, it is set to (n-1...0), where n is the rank of the input tensor.
        * Hence, by default, this operation performs a regular matrix transpose on 2-D input Tensors.
        */
        Tensor transpose(AxisPerm perm={}) const {
            if (_shape.dim() <= 1) return *this;

            int dim = _shape.dim();
            if (perm.empty()) {
                for (int i = dim - 1; i >= 0; i--) perm.push_back(i);
            }
            assert(perm.size() == _shape.dim());

            Shape shape;
            for (auto d : perm) shape.push_back(_shape[d]);
            
            Tensor ret(shape);

            int n = _shape.numel();
            std::vector<int> idx(dim);
            for (int i = 0; i < n; i++) {
                int k = i;
                for (int d = dim - 1; d >= 0; d--) {
                    idx[perm[d]] = k % _shape[d];
                    k /= _shape[d];
                }
                
                int t = idx[0];
                for (int d = 1; d < dim; d++) {
                    t = t * shape[d] + idx[d];
                }
                ret._data[t] = this->_data[i];
            }
            return ret;
        }

        /**
        * Computes the sum of elements across dimensions of a tensor, axis starts with 0.
        * Reduces along the dimensions given in axis. Unless keepdims is true,
        * the rank of the tensor is reduced by 1 for each of the entries in axis, which must be unique.
        * If keepdims is true, the reduced dimensions are retained with length 1.
        * If axis is None, all dimensions are reduced, and a tensor with a single element is returned.
        */
        Tensor reduceSum(Axis axis={}, bool keep_dims=false) const {
            return _reduce(axis, keep_dims, [](ValueType &dst, ValueType src) {dst += src;}, 0);
        }

        /**
        * Computes the maximum of elements across dimensions of a tensor, axis starts with 0.
        * Reduces along the dimensions given in axis. Unless keepdims is true,
        * the rank of the tensor is reduced by 1 for each of the entries in axis, which must be unique.
        * If keepdims is true, the reduced dimensions are retained with length 1.
        * If axis is None, all dimensions are reduced, and a tensor with a single element is returned.
        */
        Tensor reduceMax(Axis axis={}, bool keep_dims=false) const {
            ValueType initVal = -std::numeric_limits<ValueType>::max();
            return _reduce(axis, keep_dims, [](ValueType &dst, ValueType src) {dst = std::max(dst, src);}, initVal);
        }

        /**
        * Computes the minimum of elements across dimensions of a tensor, axis starts with 0.
        * Reduces along the dimensions given in axis. Unless keepdims is true,
        * the rank of the tensor is reduced by 1 for each of the entries in axis, which must be unique.
        * If keepdims is true, the reduced dimensions are retained with length 1.
        * If axis is None, all dimensions are reduced, and a tensor with a single element is returned.
        */
        Tensor reduceMin(Axis axis={}, bool keep_dims=false) const {
            ValueType initVal = std::numeric_limits<ValueType>::max();
            return _reduce(axis, keep_dims, [](ValueType &dst, ValueType src) {dst = std::min(dst, src);}, initVal);
        }

        /**
        * Computes the mean of elements across dimensions of a tensor, axis starts with 0.
        * Reduces along the dimensions given in axis. Unless keepdims is true,
        * the rank of the tensor is reduced by 1 for each of the entries in axis, which must be unique.
        * If keepdims is true, the reduced dimensions are retained with length 1.
        * If axis is None, all dimensions are reduced, and a tensor with a single element is returned.
        */
        Tensor reduceMean(Axis axis={}, bool keep_dims=false) const {
            int n = 1;
            if (axis.empty()) {
                n = _shape.numel();
            } else {
                for (int d = 0; d < _shape.dim(); d++) {
                    if (axis.count(d)) {
                        n *= _shape[d];
                    }
                }
            }
            return _reduce(axis, keep_dims, [&n](ValueType &dst, ValueType src) {dst += src / n;}, 0);
        }

        /**
        * return the only value of the tensor with shape (1)
        */
        ValueType value() const {
            assert(_shape.dim() == 1);
            assert(_shape[0] == 1);
            return _data[0];
        }

    private:
        // M(axb) x M(bxc) = M(axc)
        // matmul block matrix (last 2 dims)
        static void _matmul(ValueType *dst, ValueType *src_a, ValueType *src_b, int a, int b, int c, bool transpose_b) {
            for (int i = 0; i < a; i++) {
                for (int j = 0; j < c; j++) {
                    ValueType *p = dst + (i * c + j);
                    for (int k = 0; k < b; k++) {
                        if (transpose_b) {
                            *p += *(src_a + i * b + k) * (*(src_b + j * b + k));
                        } else {
                            *p += *(src_a + i * b + k) * (*(src_b + k * c + j));
                        }
                    }
                }
            }
        }

        Tensor _reduce(Axis axis, bool keep_dims, const ReducerFunc &reducer, ValueType initVal) const {
            Shape shape;
            if (axis.empty()) {
                for (int d : _shape) shape.push_back(1);
            } else {
                for (int d = 0; d < _shape.dim(); d++) {
                    if (axis.count(d) > 0) {
                        shape.push_back(1);
                    } else {
                        shape.push_back(_shape[d]);
                    }
                }
            }

            Tensor ret(shape, initVal);
            Index indexDst(_shape.dim(), 0);
            Index indexSrc(_shape.dim(), 0);
            _reduce(ret, *this, indexDst, indexSrc, axis, 0, reducer);

            if (!keep_dims) {
                Shape shapeReduced;
                for (int d = 0; d < shape.dim(); d++) {
                    if (!axis.empty() && !axis.count(d)) {
                        shapeReduced.push_back(shape[d]);
                    }
                }
                if (shapeReduced.empty()) {
                    shapeReduced.push_back(1);
                }
                ret._shape = std::move(shapeReduced);
            }
            return ret;
        }

        static void _reduce(Tensor &dst, const Tensor &src, Index &indexDst, Index &indexSrc, const Axis &axis,
                            int d, const ReducerFunc &reducer) {
            if (d >= src.shape().dim()) return;
            bool is_last_dim = (d == src.shape().dim() - 1);
            bool is_reduce_dim = axis.empty() || axis.count(d);
            for (int i = 0; i < src.shape()[d]; i++) {
                indexSrc[d] = i;
                indexDst[d] = is_reduce_dim ? 0 : i;
                if (is_last_dim) {
                    reducer(dst(indexDst), src(indexSrc));
                } else {
                    _reduce(dst, src, indexDst, indexSrc, axis, d+1, reducer);
                }
            }
        }

    private:
        bool _view; // the destructor does not free _data if true (tensor is build with shallow copy)
        ValueType *_data;
        Shape _shape;
    };

    typedef Tensor<float> Tensorf;

    template <typename ValueType, typename AllocatorType=Allocator<ValueType>>
    std::ostream& operator<<(std::ostream& os, const Tensor<ValueType, AllocatorType>& t) {
        if (t.shape().dim() <= 1) {
            int len = t.shape()[0];
            os << "(";
            for (int i = 0; i < len; i++) {
                os << t(i);
                if (i != len - 1) os << ",";
            }
            os << ") ";
        } else {
            os << "[";
            int len = t.shape()[0];
            for (int i = 0; i < len; i++) {
                os << t[i];
            }
            os << "] ";
        }
        return os;
    }

    template <typename ValueType, typename AllocatorType=Allocator<ValueType>>
    Tensor<ValueType, AllocatorType> operator*(ValueType k, const Tensor<ValueType, AllocatorType>& t) {
        return t * k;
    }

    template <typename ValueType, typename AllocatorType=Allocator<ValueType>>
    Tensor<ValueType, AllocatorType> operator/(ValueType k, const Tensor<ValueType, AllocatorType>& t) {
        return Tensor<ValueType, AllocatorType>(t.shape(), k) / t;
    }
}