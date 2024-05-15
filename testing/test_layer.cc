#include "autograd/autograd.hpp"
#include "autograd/shape.hpp"

#include "testing/testing.hpp"

#include <iostream>

using namespace std;
using namespace autograd;

TestCase(BasicLayer, {
    Tensorf a({
        {1, 2},
        {2, 3}
    });

    cout << "basic layer" << endl;
});