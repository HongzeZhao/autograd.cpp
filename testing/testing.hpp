#pragma once

#include <vector>
#include <string>
#include <functional>
#include <iostream>

struct TestClassBase {
    std::string name;
    TestClassBase(const char *name):name(name) {}
    virtual void run() = 0;
    virtual ~TestClassBase() = default;
};

#define Assert(expr) assert((expr) ? (void)0 : fail(#expr))
#define fail(str)   (std::cerr << "Assert failed: " << str << std::endl, 1)

#define TestCase(CASE_NAME, ...) \
struct Test##CASE_NAME : public TestClassBase { \
    Test##CASE_NAME() : TestClassBase(#CASE_NAME) { \
        run(); \
    } \
    void run() override { \
         std::cout << "Test" << this->name << " : " << __FILE__ << "(L" << __LINE__ <<")" << std::endl; \
         { __VA_ARGS__; } \
    } \
}; \
static Test##CASE_NAME _test_case_##CASE_NAME;

#define RUN_ALL_TESTS() \
int main(int argc, char*argv[]) { \
    std::cout << "finished running all test cases." << std::endl; \
    return 0; \
}