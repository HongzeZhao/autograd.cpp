#include "autograd/autograd.hpp"
#include "autograd/shape.hpp"

#include "autograd/tensor.hpp"
#include "testing/testing.hpp"

#include <iostream>
#include <cstdlib>

using namespace std;
using namespace autograd;


TestCase(RandomInitializer, {
    Shape shape = make_shape(128, 256);
    Tensorf t1(shape), t2(shape);

    const float eps = 1e-2;

    // gaussian random with mean=0, stddev=1
    t1.fillGaussianRandom(0, 1);
    cout << "random.gaussian = " << t1.reduceMean() << endl;
    Assert(abs(t1.reduceMean().value()) < eps);

    // uniform random with range [0, 1)
    t2.fillUniformRandom(0, 1);
    cout << "random.uniform = " << t2.reduceMean() << endl;
    Assert(abs(t2.reduceMean().value() - 0.5) < eps);
});

TestCase(BasicOps, {
    Tensorf a({{1, 2}, {3, 4}});
    Tensorf b({{5, 6}, {7, 8}});

    Assert(a + b == Tensorf({{6, 8}, {10, 12}}));
    Assert(a - b == Tensorf({{-4, -4}, {-4, -4}}));
    Assert(a - b == -b + a);
    Assert(a * b == Tensorf({{5, 12}, {21, 32}}));
    Assert(a / b == Tensorf({{0.2, 1.0/3}, {3.0/7, 0.5}}));
    Assert(a * 2 == Tensorf({{2, 4}, {6, 8}}));
    Assert(2.0f * a == Tensorf({{2, 4}, {6, 8}}));
    Assert(a / 2 == Tensorf({{0.5, 1}, {1.5, 2}}));
    Assert(1.0f / a == Tensorf({{1.0f/1, 1.0f/2}, {1.0f/3.0, 1.0f/4}}));

    a += b;
    Assert(a == Tensorf({{6, 8}, {10, 12}}));
    a -= b;
    Assert(a == Tensorf({{1, 2}, {3, 4}}));
    a *= b;
    Assert(a == Tensorf({{5, 12}, {21, 32}}));
    a /= b;
    Assert(a == Tensorf({{1, 2}, {3, 4}}));
    a += 1;
    Assert(a == Tensorf({{2, 3}, {4, 5}}));
    a -= 1;
    Assert(a == Tensorf({{1, 2}, {3, 4}}));
    a *= 2;
    Assert(a == Tensorf({{2, 4}, {6, 8}}));
    a /= 2;
    Assert(a == Tensorf({{1, 2}, {3, 4}}));
});

TestCase(MatmulAndTranspose, {
    Tensorf a = {
        {{0, 1}, {2, 3}, {4, 5}},
        {{0, 1}, {2, 3}, {4, 5}},
        {{0, 1}, {2, 3}, {4, 5}}
    };
    Assert(a.shape() == Shape({3, 3, 2}));

    Tensorf b = {
        {{2, 2}, {1, 2}},
        {{3, 1}, {4, 1}},
        {{3, 1}, {4, 1}}
    };
    Assert(b.shape() == Shape({3, 2, 2}));

    // matmul
    Assert(a.matmul(b, false).shape() == Shape({3, 3, 2}));
    Assert(a.matmul(b, false) == Tensorf({
        {{1, 2}, {7, 10}, {13, 18}},
        {{4, 1}, {18, 5}, {32, 9}},
        {{4, 1}, {18, 5}, {32, 9}}
    }));

    Assert(a.matmul(a, true) == Tensorf({
        {{1, 3, 5}, {3, 13, 23}, {5, 23, 41}},
        {{1, 3, 5}, {3, 13, 23}, {5, 23, 41}},
        {{1, 3, 5}, {3, 13, 23}, {5, 23, 41}}
    }));

    Assert(a.matmul(a, true) == a.matmul(a.transpose({0, 2, 1}), false));

    // transpose
    Assert(a.transpose() == a.transpose({2, 1, 0}));
    Assert(a.transpose() == Tensorf({
        {{0, 0, 0}, {2, 2, 2}, {4, 4, 4}},
        {{1, 1, 1}, {3, 3, 3}, {5, 5, 5}},
        }));

    Assert(a.transpose({0, 2, 1}) == Tensorf({
        {{0, 2, 4}, {1, 3, 5}},
        {{0, 2, 4}, {1, 3, 5}},
        {{0, 2, 4}, {1, 3, 5}}}));

    Tensorf x = {{1, 2}, {3, 4}};
    Assert(x == Tensorf({{1, 2}, {3, 4}}));
    Assert(x.transpose() == Tensorf({{1, 3}, {2, 4}}));

    // access by index vec
    Assert(x({0,0}) == 1);
    Assert(x({0,1}) == 2);
    Assert(x({1,0}) == 3);
    Assert(x({1,1}) == 4);

    // reduce sum
    Assert(x.reduceSum().shape() == Shape({1}));
    Assert(x.reduceSum() == Tensorf({10}));
    Assert(x.reduceSum({0, 1}) == Tensorf({10}));
    Assert(x.reduceSum({0}) == Tensorf({4, 6}));
    Assert(x.reduceSum({1}) == Tensorf({3, 7}));

    // reduce min/max/mean
    Assert(x.reduceMin().value() == 1);
    Assert(x.reduceMax().value() == 4);
    Assert(x.reduceMean().value() == 2.5f);

    Assert(x.reduceMin({0}) == Tensorf({1, 2}));
    Assert(x.reduceMax({0}) == Tensorf({3, 4}));
    Assert(x.reduceMean({0}) == Tensorf({2, 3}));

    Assert(x.reduceMin({1}) == Tensorf({1, 3}));
    Assert(x.reduceMax({1}) == Tensorf({2, 4}));
    Assert(x.reduceMean({1}) == Tensorf({1.5, 3.5}));
});