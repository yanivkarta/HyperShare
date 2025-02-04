#ifndef __TESTS_H__
#define __TESTS_H__
#include <gtest/gtest.h>
#endif

//add unit tests for isolation matrix 


#include "test1.h"







int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}   


