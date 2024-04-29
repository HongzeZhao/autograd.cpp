# autograd.cpp
This is a naive implementation of autograd with pure c++, for teaching and study.

## Build Requirements
The code is build with [Bazel](https://bazel.build/about), follow [the guide to install Bazel](https://bazel.build/install).

Alternatively, you can copy the cpp files to your own project as part of your project. Feel free to take away. (MIT License)

## Compile and Run
We build the autograd.cpp as c++ lib and make an executable to run and test.

```
bazel run //testing:autograd.test
```
