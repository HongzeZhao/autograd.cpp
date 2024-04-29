#include "autograd/autograd.hpp"

#include <iostream>

using namespace std;

int main() {
    int *a = new int[64];

    autograd::Tensor<int> t(a);

    cout << t.get(0) << endl;
}