#include "TensorMap.hpp"
#include "TensorBinding.hpp"

#include <iostream>

struct MyStruct
{
    TensorOwn<double,3> t;
};

void f(
    TensorMap<float,2> t2 )
{
}

PYBIND11_MODULE( tensor, m )
{
    //m.def( "f", &f,
    //        py::arg( "t2" ) );
    static_assert( !py::detail::is_template_base_of<Eigen::EigenBase,TensorMap<float,2>>::value,
            "Meh" );
    static_assert( !py::detail::is_eigen_other<TensorMap<float,2>>::value, "Yup" );

    /*
    py::class_< MyStruct > s( m, "MyStruct" );
    s.def_readwrite( "t", &MyStruct::t );
    */
}
