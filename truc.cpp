#include <iostream>
#include "TensorMap.hpp"

int main()
{
    TensorMap<const float,2> t( nullptr, 3, 4 );
    //TensorMap<float,2> t2( nullptr, 3, 4 );
    //TensorMap<const float,2> t3(t2);

    return 0;
}
