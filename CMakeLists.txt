project( test_tensor )

set(CMAKE_CXX_STANDARD 11) # C++11...

include_directories( "eigen" )

add_executable( test0 test0.cpp )
add_executable( truc truc.cpp )
