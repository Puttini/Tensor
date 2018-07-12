#include <iostream>
#include "TensorMap.hpp"

template< typename Derived >
void f( Eigen::MatrixBase< Derived >& mat ) {}

template< typename TensorDerived >
void printTensorInfo( const TensorMapTools::TensorBase< TensorDerived >& t )
{
    std::cout << "  shape: ";
    for ( int i = 0 ; i < TensorDerived::dim ; ++i )
        std::cout << t.shape(i) << " ";
    std::cout << std::endl;

    std::cout << "  stride: ";
    for ( int i = 0 ; i < TensorDerived::dim ; ++i )
        std::cout << t.stride(i) << " ";
    std::cout << std::endl;
}

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
    float data[100];
    for ( int i = 0 ; i < 100 ; ++i )
        data[i] = i;

    TensorMap<float,2> t( data, 3, 4 );
    TensorMap<float,3> t2( data, 3, 5, 4 );

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

    t2()(2)().noalias() = t.ref();

    TensorMap<const float,3> t3 = t.reshape( 2, 3, 2 );

    TensorMap<const float,2> uiop( t_own()(0)()(2) );
    //my_f( TensorMap<const float,2>(rty()(0)) );
    
    new (&t3) TensorMap<float,3>( m, 3, 2, 2 );

    TensorOwn<float,3> a( 25, 2, 3 );
    TensorMap<const float,3> b( a, 5, 5, 6 );
    TensorMap<const float,3> c( a, 25, 1, 6 );
    TensorMap<float,3> d( a, 1, 5, 30 );
    TensorMap<float,2> e( a, 150, 1 );

    TensorOwn<float,2> empty( 0, 6 );
    TensorMap<float,3> empty_map0( empty, 2, 3, 0 );
    TensorMap<float,3> empty_map1( empty, 0, 3, 1 );

    std::cout << " - a -" << std::endl;
    printTensorInfo(a);
    std::cout << " - b -" << std::endl;
    printTensorInfo(b);
    std::cout << " - c -" << std::endl;
    printTensorInfo(c);
    std::cout << " - d -" << std::endl;
    printTensorInfo(d);

    return 0;
}
