#include "autograd/autograd.hpp"

#include <iostream>

using namespace std;

int main() {
    autograd::Shape shape = autograd::make_shape(6, 4);
    cout << "dim:" << shape.dim() << endl;
    for (int i = 0; i < shape.dim(); i++) {
        cout << shape[i] << endl;
    }

    autograd::Tensorf t(shape);

    cout << "t=" << t << endl;
}