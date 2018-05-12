#include <iostream>
#include "TensorMap.hpp"

template< typename Derived >
void f( Eigen::MatrixBase< Derived >& mat ) {}

int main()
{
    TensorMap<float,2> t( nullptr, 3, 4 );
    TensorMap<float,3> t2( nullptr, 3, 4, 5 );

    t2();
    t2()();
    t2()()();
    MatrixRM<float,3,4> m;
    t << m;

    t2.data();
    t2.size();
    t.size();

    TensorOwn<float,4> t_own;
    t_own.resize( 1, 2, 3, 4 );
    t_own.data();
    TensorMap<const float,2> new_t = t_own.reshape( 6, 4 );

    t2(0)()(3);
    t2()(0)();
    t2(1,2)();
    t2(1,2,3) = 3;

    //TensorMap<const float,3> t2 = t.reshape( 1, 7, 1 );
    //TensorMap<float,2> t2( nullptr, 3, 4 );
    //TensorMap<const float,2> t3(t2);

    return 0;
}
