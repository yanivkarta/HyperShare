#include <gtest/gtest.h>


//add unit tests for isolation matrix 


#include "test1.h"
#include "test2.h"

//for builtin AVX512
#include <immintrin.h>

std::atomic_uint64_t provallo::tag_hyperplane::hplane_count(0); 


//turn on AVX512:
//
//declare naked for AVX512
//
 
int main(int argc, char **argv) {
     testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}   


