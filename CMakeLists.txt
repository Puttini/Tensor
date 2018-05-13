project( test_tensor )

set(CMAKE_CXX_STANDARD 11) # C++11...

add_subdirectory( "pybind11" )
include_directories( "eigen" )

add_executable( test0 test0.cpp )
add_executable( truc truc.cpp )

pybind11_add_module( binding_test binding_test.cpp )

