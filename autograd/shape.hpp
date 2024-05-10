#pragma once
#include <vector>
#include <ostream>

namespace autograd {

class Shape : public std::vector<int> {
public:
    Shape() {}
    Shape(const std::initializer_list<int>& init_list): std::vector<int>(init_list) {}

    /**
    Returns the total number of elements defined by this shape.
    */
    int numel() const {
        int len = 1;
        for (int i = 0; i < dim(); i++) {
            len *= (*this)[i];
        }
        return len;
    }

    bool operator==(const Shape &rhs) const {
        for (int i = 0; i < dim(); i++) {
            if ((*this)[i] != rhs[i]) return false;
        }
        return true;
    }

    bool operator!=(const Shape &rhs) const {
        for (int i = 0; i < dim(); i++) {
            if ((*this)[i] != rhs[i]) return true;
        }
        return false;
    }

    size_t dim() const {
        return size();
    }

    Shape reversed() const {
        Shape ret;
        for (auto it=rbegin(); it != rend(); ++it) {
            ret.push_back(*it);
        }
        return ret;
    }
};

inline std::ostream& operator<<(std::ostream& os, const Shape& shape) {
    os << "(";
        for (int i = 0; i < shape.dim(); i++) {
            os << shape[i];
            if (i != shape.dim() - 1) os << ",";
        }
        os << ")";
    return os;
}

inline static Shape make_shape(int x1) {
    return {x1};
}

inline static Shape make_shape(int x1, int x2) {
    return {x1, x2};
}

inline static Shape make_shape(int x1, int x2, int x3) {
    return {x1, x2, x3};
}

inline static Shape make_shape(int x1, int x2, int x3, int x4) {
    return {x1, x2, x3, x4};
}

}