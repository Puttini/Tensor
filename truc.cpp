#include <iostream>
#include "TensorMap.hpp"

template< typename Derived >
void f( Eigen::MatrixBase< Derived >& mat ) {}

/*
void static_failures()
{
    // Invalid dims
    TensorMap<double,0> zero_tensor_map;
    TensorOwn<double,-1> neg_tensor_own;

    // Incompatible dims
    TensorMap<double,2> tensor_map_2;
    TensorMap<double,3> tensor_map_3 = tensor_map_2;

    // Incompatible types (obvious)
    TensorMap<const float,4>  tensor_map_float;
    TensorMap<const double,4> tensor_map_double = tensor_map_float;
    TensorMap<float,4> tensor_map_non_const = tensor_map_float;

    // Too many use or operator()
    TensorMap<int,4> tensor_map_4;
    tensor_map_4()()()()();
    tensor_map_4(0,1,2,3,4);
    tensor_map_4(0)(1)(2)(3)(4);
}
*/

void my_f( const TensorMap<const float,2>& t ) {}

int main()
{
    TensorMap<float,2> t( nullptr, 3, 4 );
    TensorMap<float,3> t2( nullptr, 3, 4, 5 );

    static_assert( TensorMapTools::ConstCompatible< float, const float >(),
           "Not const compatible" );

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

    // Try using a bit of operator()
    t2(0)()(3);
    t2()(0)();
    t2(1,2)();
    t2(1,2,3) = 3;

    t2()(2)().noalias() = t(1).ref();

    TensorMap<const float,3> t3 = t.reshape( 1, 7, 1 );

    TensorMap<float,3> aze;
    TensorMap<const float,3> rty;
    aze(0)()() = rty(0)()();
    TensorMap<const float,2> uiop( rty()(0) );
    //my_f( TensorMap<const float,2>(rty()(0)) );
    
    new (&aze) TensorMap<float,3>( m, 3, 2, 2 );

    return 0;
}
