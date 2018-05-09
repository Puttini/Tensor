
// Example program
#include <iostream>
#include <string>
#include <cassert>

struct EnableIfType {};

template< bool cond >
using EnableIf = typename std::enable_if< cond, EnableIfType >::type;

template< int dim, typename Integer >
struct Shape
{
    const Integer* shape;

    Shape( const Integer* shape )
     : shape(shape)
    {}

    template< int d, typename = EnableIf<(d<dim-1)> >
    inline int size() const
    { return shape[d]*shape[d-1]; }
};

template< int dim, typename Integer >
struct Stride
{
    const Integer* stride;

    Stride( const Integer* stride )
     : stride(stride)
    {}

    inline Integer innerStride() const
    { return stride[dim-1]; }
};

// We need structures for template specialization
template< int n >
struct NthOfPack
{
    // This is used to determine the type without auto
    // Cannot be static, unfortunately
    template< typename Arg0, typename ... Args >
    typename std::result_of< NthOfPack<n-1>(Args&&...) >::type&& operator()( Arg0&&, Args&& ... args )
    { return NthOfPack<n-1>::nth( std::forward<Args>(args)... ); }

    // This is the constexpr with perfect forwarding
    template< typename Arg0, typename ... Args >
    constexpr static typename std::result_of< NthOfPack<n-1>(Args&&...) >::type&& nth( Arg0&&, Args&& ... args )
    { return NthOfPack<n-1>::nth( std::forward<Args>(args)... ); }
};

template<>
struct NthOfPack<0>
{
    template< typename Arg0, typename ... Args >
    Arg0&& operator()( Arg0&& arg0, Args&& ... args )
    { return std::forward<Arg0>(arg0); }

    template< typename Arg0, typename ... Args >
    constexpr static Arg0&& nth( Arg0&& arg0, Args&& ... args )
    { return std::forward<Arg0>(arg0); }
};

template< int n, typename ... Pack >
constexpr typename std::result_of< NthOfPack<n>(Pack&&...) >::type&&
nth_of_pack( Pack&& ... pack )
{ return std::forward<typename std::result_of< NthOfPack<n>(Pack&&...) >::type>(NthOfPack<n>::nth(pack...)); }

template< typename ... Pack >
constexpr typename std::result_of< NthOfPack<0>(Pack&&...) >::type&&
first_of_pack( Pack&& ... pack )
{ return std::forward<typename std::result_of< NthOfPack<0>(Pack&&...) >::type>(NthOfPack<0>::nth(pack...)); }

template< typename ... Pack >
constexpr typename std::result_of< NthOfPack<sizeof...(Pack)-1>(Pack&&...) >::type&&
first_of_pack( Pack&& ... pack )
{ return std::forward<typename std::result_of< NthOfPack<sizeof...(Pack)-1>(Pack&&...) >::type>(NthOfPack<sizeof...(Pack)-1>::nth(pack...)); }



template< typename ScalType, int dim >
class TensorMap
{
public:
    ScalType* data_;
    int stride_[dim];
    int shape_[dim];

    // A bit more tricky : from another Tensor (including strange strides)
    // It can be a pybind array, an Eigen matrix, another Tensor, or whatever
    template< int s, int other_s, int other_dim, typename ShapeType, typename StrideType, typename ... Dimensions, typename = EnableIf< (s>=0&&other_s>=0) > >
    inline void init_reshape_tensor( const Shape<other_dim,ShapeType>& other_shape,
                              const Stride<other_dim,StrideType>& other_stride,
                              int current_total_size,
                              int other_current_total_size,
                              Dimensions ... dimensions )
    {
        int new_total_size = current_total_size*nth_of_pack<s>(dimensions...);
        int other_new_total_size = other_current_total_size*other_shape.shape[other_s];

        std::cout << s << "," << other_s << ":" << std::endl;
        std::cout << " " << current_total_size << std::endl;
        std::cout << " " << other_current_total_size << std::endl;
        shape_[s] = nth_of_pack<s>(dimensions...);

        if ( current_total_size*nth_of_pack<s>(dimensions...) == other_current_total_size*other_shape.shape[other_s] )
        {
            std::cout << " ==" << std::endl;
            stride_[s] = other_stride.stride[other_s];

            if ( s > 0 )
            {
                init_reshape_tensor<s-1,other_s,other_dim,ShapeType,StrideType>(
                        other_shape, other_stride,
                        new_total_size,
                        other_current_total_size,
                        dimensions... );
            }
            else
            {
                init_reshape_tensor<s,other_s-1,other_dim,ShapeType,StrideType>(
                        other_shape, other_stride,
                        current_total_size,
                        other_new_total_size,
                        dimensions... );
            }
        }
        else if ( current_total_size*nth_of_pack<s>(dimensions...) > other_current_total_size*other_shape.shape[other_s] )
        {
            std::cout << " >" << std::endl;
            std::cout << " " << (other_current_total_size*other_shape.shape[other_s]*other_stride.stride[other_s]) << " " << other_stride.stride[other_s-1] << std::endl;
            // Split other dimension. The strides must be compatible
            assert( other_dim > 0
                    && (other_current_total_size*other_shape.shape[other_s]*other_stride.stride[other_s] == other_stride.stride[other_s-1])
                    && "Incompatible stride/shape" );

            init_reshape_tensor<s,other_s-1,other_dim,ShapeType,StrideType>(
                    other_shape, other_stride,
                    current_total_size,
                    other_current_total_size*other_shape.shape[other_s],
                    dimensions... );
        }
        else // current_total_size < other_current_total_size
        {
            std::cout << " <" << std::endl;
            // Split this dimension. The strides have no constraint
            stride_[s] = (s == dim-1)
                         ? other_stride.innerStride()
                         : stride_[s+1] * shape_[s+1];

            init_reshape_tensor<s-1,other_s,other_dim,ShapeType,StrideType>(
                    other_shape, other_stride,
                    current_total_size*nth_of_pack<s>(dimensions...),
                    other_current_total_size,
                    dimensions... );
        }
    }
    
    template< int s, int other_s, int other_dim, typename ShapeType, typename StrideType, typename ... Dimensions, typename = EnableIf< (s<0||other_s<0) > >
    inline void init_reshape_tensor( const Shape<other_dim,ShapeType>& other_shape,
                              const Stride<other_dim,StrideType>& other_stride,
                              int current_total_size,
                              int other_current_total_size,
                              Dimensions ... dimensions ) const
    {}
};

int main()
{
  TensorMap<float,4> t1;
  t1.shape_[0] = 2;
  t1.shape_[1] = 4;
  t1.shape_[2] = 3;
  t1.shape_[3] = 2;
  
  t1.stride_[3] = 5;
  t1.stride_[2] = 10;
  t1.stride_[1] = 40;
  t1.stride_[0] = 200;
  
  TensorMap<float,5> t2;
  t2.init_reshape_tensor<4,3,4,int,int>( Shape<4,int>( t1.shape_ ), Stride<4,int>( t1.stride_ ),
        1, 1,
        1, 2, 2, 2, 6 );
        
  std::cout << "Old shape: " << t1.shape_[0] << " " <<  t1.shape_[1] << " " << t1.shape_[2] << " " << t1.shape_[3] << " " << std::endl;
  std::cout << "Old stride: " << t1.stride_[0] << " " <<  t1.stride_[1] << " " << t1.stride_[2] << " " << t1.stride_[3] << " " << std::endl;
  std::cout << "New shape: " << t2.shape_[0] << " " <<  t2.shape_[1] << " " << t2.shape_[2] << " " << t2.shape_[3] << " " << t2.shape_[4] << std::endl;
  std::cout << "New stride: " << t2.stride_[0] << " " <<  t2.stride_[1] << " " << t2.stride_[2] << " " << t2.stride_[3] << " " << t2.stride_[4] << std::endl;

  return 0;
}
