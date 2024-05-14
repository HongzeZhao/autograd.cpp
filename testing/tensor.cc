#include "autograd/autograd.hpp"
#include "autograd/shape.hpp"

#include "testing/testing.hpp"

#include <iostream>

using namespace std;

TestCase(BasicOps, {
    autograd::Shape shape = autograd::make_shape(6, 4);
    cout << "dim:" << shape.dim() << endl;
    for (int i = 0; i < shape.dim(); i++) {
        cout << shape[i] << endl;
    }

    autograd::Tensorf t1(shape), t2(shape);

    t1.fillGaussianRandom(0, 1);
    t2.fillUniformRandom(0, 1);

    cout << "t1=" << t1 << endl << "t2=" << t2 << endl;
    cout << "t1+t2=" << t1 + t2 << endl;
    cout << "t1 - t2=" << t1 - t2 << endl;
    cout << "-t1 + t2=" << -t1 + t2 << endl;
    cout << "t1 * t2=" << t1 * t2 << endl;
    cout << "t1 * 2=" << t1 * 2 << endl;
    cout << "t1 / t2=" << t1 / t2 << endl;
    cout << "t1 / 2=" << t1 / 2 << endl;
});

TestCase(MatmulAndTranspose, {
    autograd::Tensorf a = {
        {{0, 1}, {2, 3}, {4, 5}},
        {{0, 1}, {2, 3}, {4, 5}},
        {{0, 1}, {2, 3}, {4, 5}}
    };
    cout << "a = " << a << endl;

    autograd::Tensorf b = {
        {{2, 2}, {1, 2}},
        {{3, 1}, {4, 1}},
        {{3, 1}, {4, 1}}
    };
    cout << "b = " << b << endl;

    cout << "a x b = " << a.matmul(b, false) << endl;
    cout << "a x a^t = " << a.matmul(a, true) << endl;

    cout << "a.transpose() = " << a.transpose() << endl;
    cout << "a.transpose(0,2,1) = " << a.transpose({0, 2, 1}) << endl;

    autograd::Tensorf x = {{1, 2}, {3, 4}};
    cout << "x = " << x << endl;
    cout << "x.transpose() = " << x.transpose() << endl;
});