#cmake file for python wrapper for isolation matrix classifiers and transfer learning 
#mechanisms.

cmake_minimum_required(VERSION 3.0) 

project(isolation_mat_python)

#set the python version
set(PYTHON_VERSION 3.6)

#set the python path
set(PYTHON_PATH /usr/include/python${PYTHON_VERSION})

#set the python library path
set(PYTHON_LIB_PATH /usr/lib/python${PYTHON_VERSION})

#set the python library
set(PYTHON_LIB python${PYTHON_VERSION})

#set the python include path
set(PYTHON_INCLUDE_PATH /usr/include/python${PYTHON_VERSION})

#set the python numpy path
set(PYTHON_NUMPY_PATH /usr/lib/python${PYTHON_VERSION}/dist-packages/numpy/core/include)

#set the python numpy library
set(PYTHON_NUMPY_LIB python${PYTHON_VERSION})

#set the python numpy include path
set(PYTHON_NUMPY_INCLUDE_PATH /usr/lib/python${PYTHON_VERSION}/dist-packages/numpy/core/include)

 # Find Python libraries
find_package(PythonLibs REQUIRED)

#set the include directories :
include_directories(${PYTHON_INCLUDE_PATH} ${PYTHON_NUMPY_INCLUDE_PATH} ${PYTHON_PATH} ${PYTHON_LIB_PATH} ${PYTHON_NUMPY_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../include) 

#set cpp include directories:
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include) 

set (src_cpp src/isolation_matrix_python.cpp)   


# Link with Python libraries



add_library(isolation_mat_python SHARED ${src_cpp}) 

#add stdc++ static library:
target_link_libraries(isolation_mat_python stdc++ ) 

#install the library, make debian package and python module : 
install(TARGETS isolation_mat_python DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../lib) 
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/isolation_matrix_python.py DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../lib)

#set the python library path
set(PYTHON_LIB_PATH /usr/lib/python${PYTHON_VERSION})
