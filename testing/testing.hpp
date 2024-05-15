#pragma once

#include <vector>
#include <string>
#include <functional>
#include <iostream>
#include <cassert>

struct TestClassBase {
    std::string name;
    TestClassBase(const char *name):name(name) {}
    virtual void run() = 0;
    virtual ~TestClassBase() = default;
};

#define Assert(expr) (void)(!(expr) ? fail(#expr) : success(#expr))
#define success(str)   (std::cout << "\033[1;36m" << str << "\033[0m\t\033[1;32m[ok]\033[0m" << std::endl, 0)
#define fail(str)   (std::cerr << "\033[1;31m" << str << "\t[fail]\033[0m" << std::endl, 1)

#define TestCase(CASE_NAME, ...) \
struct Test##CASE_NAME : public TestClassBase { \
    Test##CASE_NAME() : TestClassBase(#CASE_NAME) { \
        run(); \
    } \
    void run() override { \
         std::cout << std::endl << "\033[1;33mTest" << this->name \
            << " : " << __FILE__ << "(L" << __LINE__ <<")\033[0m" << std::endl; \
         { __VA_ARGS__; } \
    } \
}; \
static Test##CASE_NAME _test_case_##CASE_NAME;

#define RUN_ALL_TESTS() \
int main(int argc, char*argv[]) { \
    std::cout << "finished running all test cases." << std::endl; \
    return 0; \
}