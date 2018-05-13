#ifndef TENSOR_BINDING_HPP
#define TENSOR_BINDING_HPP

#include <pybind11/pybind11.h>
#include <pybind11/detail/common.h>
#include "TensorMap.hpp"
namespace pybind11 { namespace detail {
template<>
struct is_template_base_of_impl<Eigen::EigenBase>
{
    template< typename T,
        typename = TensorMapTools::EnableIf<
            !is_template_base_of<TensorMapTools::TensorBase, T>::value > >
    static std::true_type check( Eigen::EigenBase< T >* )
    {
        static_assert( std::is_same<T,TensorMap<float,2>>::value, "Wtf" );
    }
    static std::false_type check(...);
};
} }
#include <pybind11/eigen.h>

#include <iostream>

namespace py = pybind11;

namespace pybind11
{

namespace detail
{

//template<typename ScalType,int dim>
//using is_eigen_other<TensorMap<ScalType,dim>> = false;



/*
template< template<typename...> class Base >
struct is_template_base_of_impl
{
    template <typename... Us> static std::true_type check( Base<Us...> * );
    static std::false_type check(...);
};
*/

template<typename ScalType,int dim>
struct type_caster< TensorMap<ScalType,dim> >
{
public:
    typedef TensorMap<ScalType,dim> Type;
    PYBIND11_TYPE_CASTER( Type, _("TensorMap") );

    bool load( handle src, bool implicit)
    {
        std::cout << "Trying to cast to TensorMap" << std::endl;
        return false;
    }

    static handle cast(
            Type tensor,
            return_value_policy policy,
            handle parent )
    {
        return py::array_t<ScalType>();
    }
};

} // namespace detail

} // namespace pybind11

#endif // TENSOR_BINDING_HPP
