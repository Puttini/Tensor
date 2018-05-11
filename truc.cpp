#include <iostream>
#include "TensorMap.hpp"

int main()
{
    TensorMap<const float,2> t( nullptr, 3, 4 );
    TensorMap<float,3> t2( nullptr, 3, 4, 5 );
    TensorMap<const float, 5> t5 = t2.reshape( 12, 5, 6, 7, 8 );

    //TensorMap<const float,3> t2 = t.reshape( 1, 7, 1 );
    //TensorMap<float,2> t2( nullptr, 3, 4 );
    //TensorMap<const float,2> t3(t2);

    return 0;
}
