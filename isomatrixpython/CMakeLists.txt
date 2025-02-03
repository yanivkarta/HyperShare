#cmake file for python wrapper for isolation matrix classifiers and transfer learning 
#mechanisms.

cmake_minimum_required(VERSION 3.0) 

project(isolation_mat_python)
find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
#set the python version
set(PYTHON_VERSION 3.8)
#get python path:
get_filename_component(PY_PATH ${PYTHON_EXECUTABLE} DIRECTORY) 


#set the python path:
#use regex to get the path from the bin folder
string(REGEX REPLACE "/bin.*" "" PY_PATH ${PY_PATH})

set(PYTHON_PATH ${PY_PATH})
set(PYTHON_LIB_PATH ${PY_PATH}/lib)
set(PYTHON_LIB ${PYTHON_LIB_PATH}/libpython${PYTHON_VERSION}.so) 
#set python lib:

 # Find Python libraries

#debug message:
message(STATUS "PYTHON_PATH: ${PYTHON_PATH}")
message(STATUS "PYTHON_LIB_PATH: ${PYTHON_LIB_PATH}")

#set the python include path
set(PYTHON_INCLUDE_PATH ${PYTHON_PATH}/include/python${PYTHON_VERSION})

#set the python numpy include path
set(PYTHON_NUMPY_INCLUDE_PATH ${PYTHON_PATH}/include/python${PYTHON_VERSION}/numpy)

#set the python numpy path
set(PYTHON_NUMPY_PATH ${PYTHON_PATH}/lib/python${PYTHON_VERSION}/site-packages/numpy/core/include)


#add numpy include path and library path and enable compilation flags for numpy support:
add_definitions(-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION) 

include(FetchContent)
FetchContent_Declare(
  googletest
  #  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
  URL https://github.com/google/googletest/releases/download/v1.15.2/googletest-1.15.2.tar.gz
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)



#set the include directories :
include_directories(${PYTHON_INCLUDE_PATH} ${PYTHON_NUMPY_INCLUDE_PATH} ${PYTHON_PATH} ${PYTHON_LIB_PATH} ${PYTHON_NUMPY_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../include) 
#add lib path for object search: 

#set cpp include directories:
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include) 

set (src_cpp src/isolation_matrix_python.cpp)   

#enable AVX options to the compiler
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -mavx2 -mfma -mfpmath=sse -march=native -mtune=native -O3 -fPIC -fno-omit-frame-pointer -fno-strict-aliasing  -fno-stack-protector -fno-strict-overflow -fno-delete-null-pointer-checks -fno-lifetime-dse -fno-merge-all-constants")    
#set CMAKE_CXX_LINKER_FLAGS,add gtest and python libraries and include directories : 

#set(CMAKE_CXX_LINKER_FLAGS "${CMAKE_CXX_LINKER_FLAGS} -lstdc++ -lgtest -l${PYTHON_LIB}") 

# Link with Python libraries


add_library(isolation_mat_python SHARED ${src_cpp}) 

#add stdc++ static library,gtest, 

target_link_libraries(isolation_mat_python stdc++ gtest ${PYTHON_LIB})
  

#add gtest library:
#target_link_libraries(isolation_mat_python gtest) 
#run gtest tests: 



#set test source files :
set(test_src tests/run_tests.cpp)
#set test_src compilation flags:

add_definitions(-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION) 

#add linker flags , libraries (gtest) and include directories: 
#link with gtest_main library:

#add_library for gtest tests:
add_link_options(-g) 


#add CMAKE_CXX_LINKER_FLAGS:
set (CMAKE_CXX_LINKER_FLAGS "${CMAKE_CXX_LINKER_FLAGS} -lstdc++ -lgtest -l${PYTHON_LIB}") 

#add gtest_main library:
add_executable(isolation_mat_python_test  ${test_src} ) 


target_link_libraries(isolation_mat_python_test  ${PYTHON_LIB} gtest)    




#run gtest tests:




#install the library, make debian package and python module : 
install(TARGETS isolation_mat_python DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../lib) 
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/test.py DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../lib)

#set the python library path
set(PYTHON_LIB_PATH /usr/lib/python${PYTHON_VERSION})
