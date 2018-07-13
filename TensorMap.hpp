#ifndef TENSOR_MAP_HPP
#define TENSOR_MAP_HPP

#include <Eigen/Core>
#include <cassert>

// Eigen matrix and vector
// REMOVE ME
template< typename ScalType, int rows = Eigen::Dynamic, int cols = Eigen::Dynamic >
using MatrixRM = Eigen::Matrix<ScalType,rows,cols,Eigen::RowMajor>;

template< typename ScalType, int rows = Eigen::Dynamic, int cols = Eigen::Dynamic >
using MatrixCM = Eigen::Matrix<ScalType,rows,cols,Eigen::ColMajor>;

template< typename ScalType, int size = Eigen::Dynamic >
using Vector = MatrixCM<ScalType,size,1>;

#ifdef _MSC_VER
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

namespace TensorMapTools
{

// ----- Constness tricks -----

template< typename ScalType >
struct NonConst_
{ typedef ScalType type; };
template< typename ScalType >
struct NonConst_< const ScalType >
{ typedef ScalType type; };

template< typename ScalType >
struct Const_
{ typedef const ScalType type; };
template< typename ScalType >
struct Const_< const ScalType >
{ typedef const ScalType type; };


template< typename ScalType >
using Const = typename Const_<ScalType>::type;

template< typename ScalType >
using NonConst = typename NonConst_<ScalType>::type;

// Template variable only available in C++14
template< typename ScalType >
constexpr bool IsConst() { return std::is_const<ScalType>::value; }

template< typename FromScalType, typename ToScalType >
constexpr bool ConstCompatible() { return !IsConst<FromScalType>() || IsConst<ToScalType>(); }

template< typename AsThis, typename Type >
using ConstAs = typename std::conditional< IsConst<AsThis>(), Const<Type>, NonConst<Type> >::type;

// ----- EnableIf tricks -----

struct EnableIfType {};

template< bool cond >
using EnableIf = typename std::enable_if< cond, EnableIfType >::type;

// ----- Explicitly use empty constructor -----

struct EmptyConstructor {};

// ----- Static max and min -----

// std::max and std::min are constexpr since C++14 only

template< typename T >
constexpr T max( T x, T y )
{ return x > y ? x : y; }

template< typename T >
constexpr T min( T x, T y )
{ return x > y ? y : x; }

// ----- Pack accessors -----

// We need structures for template specialization
template< int n >
struct NthOfPack
{
    // This is used to determine the type without auto (C++14)
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

// ----- Forward declarations of final types -----

// Tensor types
template< typename ScalType, int dim >
class TensorOwn;

template< typename ScalType, int dim >
class TensorMap;

// Shape types
template< int dim, typename Integer = int >
class ShapeMap;

template< int dim, typename Integer = int >
class ShapeOwn;

// Stride types
template< int dim, typename Integer = int >
class StrideMap;

template< int dim, typename Integer = int >
class StrideOwn;

// ----- Traits -----

template< typename Type >
struct Traits;

// Tensor traits
template< typename _ScalType, int _dim >
struct Traits< TensorMap<_ScalType,_dim> >
{
    typedef _ScalType ScalType;
    static constexpr int dim = _dim;
    static constexpr bool owns = false;
};

template< typename _ScalType, int _dim >
struct Traits< TensorOwn<_ScalType,_dim> >
{
    typedef _ScalType ScalType;
    static constexpr int dim = _dim;
    static constexpr bool owns = true;
};

// Shape traits
template< int _dim, typename _Integer >
struct Traits< ShapeMap<_dim,_Integer> >
{
    typedef _Integer Integer;
    static constexpr int dim = _dim;
    static constexpr bool owns = false;
};

template< int _dim, typename _Integer >
struct Traits< ShapeOwn<_dim,_Integer> >
{
    typedef _Integer Integer;
    static constexpr int dim = _dim;
    static constexpr bool owns = true;
};

// Stride traits
template< int _dim, typename _Integer >
struct Traits< StrideMap<_dim,_Integer> >
{
    typedef _Integer Integer;
    static constexpr int dim = _dim;
    static constexpr bool owns = false;
};

template< int _dim, typename _Integer >
struct Traits< StrideOwn<_dim,_Integer> >
{
    typedef _Integer Integer;
    static constexpr int dim = _dim;
    static constexpr bool owns = true;
};

// ----- Shape tool structures -----

template< typename Derived >
struct Shape
{
    typedef typename Traits<Derived>::Integer Integer;
    static constexpr int dim = Traits<Derived>::dim;
    static constexpr bool owns = Traits<Derived>::owns;

    Derived& derived() { return *static_cast<Derived*>(this); }
    const Derived& derived() const { return *static_cast<const Derived*>(this); }

    inline int size() const
    {
        int sz = 1;
        for ( int s = 0 ; s < dim ; ++s )
            sz *= derived().get(s)*size(s+1);
        return sz;
    }

    inline Integer operator[]( int i ) const
    { return derived().get(i); }
};

template< int _dim, typename _Integer >
struct ShapeMap : public Shape< ShapeMap<_dim,_Integer> >
{
    typedef Shape<ShapeMap> Base;
    using typename Base::Integer;
    using Base::dim;
    using Base::owns;

    const Integer* shape;

    explicit ShapeMap( const Integer* shape )
     : shape(shape)
    {}

    Integer get( int i ) const
    { return shape[i]; }
};

template< int _dim, typename _Integer >
struct ShapeOwn : public Shape< ShapeOwn<_dim,_Integer> >
{
    typedef Shape<ShapeOwn> Base;
    using typename Base::Integer;
    using Base::dim;
    using Base::owns;

    Integer shape[dim];

    ShapeOwn() = default;

    template< typename ... Dimensions, typename = EnableIf<sizeof...(Dimensions)==dim> >
    explicit ShapeOwn( Dimensions ... dimensions )
    {
        init<0>( dimensions... );
    }

    template< int s, typename ... OtherDimensions >
    void init( int dimension, OtherDimensions ... other_dimensions )
    {
        shape[s] = dimension;
        init<s+1>( other_dimensions... );
    }

    template< int s >
    void init()
    { }

    inline Integer get( int i ) const
    { return shape[i]; }
};

// ----- Stride tool structures -----

// Eigen dynamic strides
typedef Eigen::InnerStride<Eigen::Dynamic> DynInnerStride;
typedef Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> DynStride;

template< typename Derived >
struct Stride
{
    typedef typename Traits<Derived>::Integer Integer;
    static constexpr int dim = Traits<Derived>::dim;
    static constexpr bool owns = Traits<Derived>::owns;

    Derived& derived() { return *static_cast<Derived*>(this); }
    const Derived& derived() const { return *static_cast<const Derived*>(this); }

    inline Integer innerStride() const
    { return derived().get(dim-1); }

    inline Integer operator[]( int i ) const
    { return derived().get(i); }
};

template< int _dim, typename _Integer >
struct StrideMap : public Stride< StrideMap<_dim,_Integer> >
{
    typedef Stride<StrideMap> Base;
    using typename Base::Integer;
    using Base::dim;
    using Base::owns;

    const Integer* stride;

    explicit StrideMap( const Integer* stride )
     : stride(stride)
    {}

    Integer get( int i ) const
    { return stride[i]; }
};

template< int _dim, typename _Integer >
struct StrideOwn : public Stride< StrideOwn<_dim,_Integer> >
{
    typedef Stride<StrideOwn> Base;
    using typename Base::Integer;
    using Base::dim;
    using Base::owns;

    Integer stride[dim];

    StrideOwn() = default;

    template< typename ... Dimensions, typename = EnableIf<sizeof...(Dimensions)==dim> >
    explicit StrideOwn( Dimensions ... dimensions )
    {
        init<0>( dimensions... );
    }

    template< int subst = dim, typename = EnableIf<subst==2> >
    StrideOwn( const DynStride& other )
    {
        stride[0] = other.outer();
        stride[1] = other.inner();
    }

    template< int subst = dim, typename = EnableIf<subst==1> >
    StrideOwn( const DynInnerStride& other )
    {
        stride[0] = other.inner();
    }

    template< int s, typename ... OtherDimensions >
    void init( int dimension, OtherDimensions ... other_dimensions )
    {
        stride[s] = dimension;
        init<s+1>( other_dimensions... );
    }

    template< int s >
    void init()
    { }

    Integer get( int i ) const
    { return stride[i]; }
};

struct InnerStride
{
    int inner;
    explicit InnerStride( int i )
     : inner(i)
    {}

    InnerStride( const DynInnerStride& other )
     : inner( other.inner() )
    {}
};

// ----- Some other usefull forward declarations -----

template< typename Derived, int _dim = Traits<Derived>::dim, int _current_dim = 0 >
class TensorOperator;

// ----- TensorBase -----

// This CRTP class describes the generic behaviour
template< typename Derived >
class TensorBase
{
public:
    typedef typename Traits<Derived>::ScalType ScalType;
    constexpr static int dim = Traits<Derived>::dim;
    constexpr static bool owns = Traits<Derived>::owns;

    typedef Eigen::Ref<ConstAs<ScalType, MatrixRM<NonConst<ScalType>>>,
            Eigen::Unaligned, DynStride>
            MatrixRef;
    typedef Eigen::Ref<ConstAs<ScalType, Vector<NonConst<ScalType>>>,
            Eigen::Unaligned, DynInnerStride>
            VectorRef;
    static_assert(dim > 0, "You are a weird person");

public:
    Derived& derived() { return *static_cast<Derived*>(this); }
    const Derived& derived() const { return *static_cast<const Derived*>(this); }

protected:
    TensorBase() = default;
    explicit TensorBase( EmptyConstructor ) {}

    // Used to initialize strides and shapes in constructors
    // (inner stride must be specified before calling this)
    template< int s, typename ... OtherDimensions >
    inline void init_sns_from_shape( int dimension, OtherDimensions ... other_dimensions )
    {
        Derived& d = derived();
        d.set_shape( s, dimension );
        init_sns_from_shape<s+1>( other_dimensions... );
        d.set_stride( s, d.shape(s + 1) * d.stride(s + 1) );
    }
    template< int s >
    inline void init_sns_from_shape( int dimension )
    {
        Derived& d = derived();
        d.set_shape( s, dimension );
    }

    // Give the total shape of the tensor
    template< typename ... OtherDimensions >
    inline static int total_size( int dimension, OtherDimensions ... other_dimensions )
    {
        return dimension*total_size( other_dimensions... );
    }
    inline static int total_size()
    { return 1; }

    // A bit tricky : from another Tensor (including strange strides)
    // It can be a pybind array, an Eigen matrix, another Tensor, or whatever
    // It has to be recursive to statically access the members of Dimensions...
    template< typename ShapeDerived, typename StrideDerived,
        typename ... Dimensions,
        typename = EnableIf< sizeof...(Dimensions) == dim > >
    void init_sns_reshape_tensor(
            ScalType* other_data,
            const Shape<ShapeDerived>& other_shape,
            const Stride<StrideDerived>& other_stride,
            Dimensions ... dimensions )
    {
        static_assert( ShapeDerived::dim == StrideDerived::dim,
                "Invalid input shape/stride" );

        Derived& d = derived();
        d.set_data( other_data );
        d.set_shape( dim-1, nth_of_pack<dim-1>( dimensions... ) );
        d.set_stride( dim-1, other_stride.innerStride() );

        init_sns_reshape_tensor_loop<
            dim-1, ShapeDerived::dim-1,
            ShapeDerived, StrideDerived >(
                    other_shape, other_stride,
                    nth_of_pack<dim-1>( dimensions... ),
                    other_shape[ ShapeDerived::dim-1 ],
                    dimensions... );
    }

    template< int s, int other_s,
        typename ShapeDerived, typename StrideDerived,
        typename ... Dimensions >
    void init_sns_reshape_tensor_loop(
            const Shape<ShapeDerived>& other_shape,
            const Stride<StrideDerived>& other_stride,
            int current_total_size,
            int other_current_total_size,
            Dimensions ... dimensions )
    {
        Derived& d = derived();

        if ( current_total_size == other_current_total_size )
        {
            if ( s > 0 )
            {
                d.set_shape( s-1, nth_of_pack< max(0,s-1) >( dimensions... ) );
                if ( other_s > 0 )
                    d.set_stride( s-1, other_stride[other_s-1] );
                else
                    d.set_stride( s-1, d.stride(s) * d.shape(s) );
                current_total_size *= nth_of_pack< max(0,s-1) >( dimensions... );
            }

            if ( other_s > 0 )
            {
                other_current_total_size *= other_shape[other_s-1];
            }

            if ( s > 0 || other_s > 0 )
            {
                init_sns_reshape_tensor_loop<
                    max(0,s-1), max(0,other_s-1),
                    ShapeDerived, StrideDerived >(
                        other_shape, other_stride,
                        current_total_size,
                        other_current_total_size,
                        dimensions... );
            }
        }
        else if ( current_total_size > other_current_total_size )
        {
            // Split other dimension. The strides must be compatible
            if ( other_current_total_size > 0 )
            {
                assert( other_s > 0 && other_stride[other_s-1] ==
                                other_shape[other_s] * other_stride[other_s]
                        && "Incompatible stride/shape" );

                other_current_total_size *= other_shape[other_s-1];

                init_sns_reshape_tensor_loop<
                    s, max(0,other_s-1),
                    ShapeDerived, StrideDerived >(
                        other_shape, other_stride,
                        current_total_size,
                        other_current_total_size,
                        dimensions... );
            }
            else
            {
                if ( s > 0 )
                {
                    current_total_size *= nth_of_pack< max(0,s-1) >( dimensions... );

                    init_sns_reshape_tensor_loop<
                        max(0,s-1), max(0,other_s-1),
                        ShapeDerived, StrideDerived >(
                            other_shape, other_stride,
                            current_total_size,
                            other_current_total_size,
                            dimensions... );
                }
                else // current_total_size > 0, other_current_total_size == 0, s = 0
                {
                    assert( false && "Incompatible size (reshape zero-size tensor)" );
                }
            }
        }
        else // new_total_size < other_new_total_size
        {
            assert( s > 0 && "Incompatible sizes" );

            // Split this dimension. The strides have no constraint
            d.set_shape( s-1, nth_of_pack< max(0,s-1) >( dimensions... ) );
            d.set_stride( s-1, d.stride(s) * d.shape(s) );
            current_total_size *= nth_of_pack< max(0,s-1) >( dimensions... );

            init_sns_reshape_tensor_loop<
                max(0,s-1), other_s,
                ShapeDerived, StrideDerived>(
                    other_shape, other_stride,
                    current_total_size,
                    other_current_total_size,
                    dimensions... );
        }
    }

public:
    // data, shape, stride from derived
    inline ScalType* data()
    { return derived().data(); }

    inline Const<ScalType>* data() const
    { return derived().data(); }

    inline int shape( int i ) const
    { return derived().shape(i); }

    inline int stride( int i ) const
    { return derived().stride(i); }

    // Reshape the tensor
    template< int new_dim, typename ... Dimensions,
        typename = EnableIf< sizeof...(Dimensions)==new_dim && !IsConst<ScalType>() > >
    TensorMap< ScalType, new_dim >
    reshape( Dimensions ... dimensions );

    template< int new_dim, typename ... Dimensions,
        typename = EnableIf< sizeof...(Dimensions)==new_dim > >
    TensorMap< Const<ScalType>, new_dim >
    reshape( Dimensions ... dimensions ) const;

    // You can omit the new dimension
    template< typename ... Dimensions, typename Subst = ScalType, typename = EnableIf< !IsConst<Subst>() > >
    inline TensorMap< ScalType, sizeof...(Dimensions) >
    reshape( Dimensions ... dimensions  )
    { return reshape< sizeof...(Dimensions), Dimensions... >( dimensions... ); }

    template< typename ... Dimensions >
    inline TensorMap< Const<ScalType>, sizeof...(Dimensions) >
    reshape( Dimensions ... dimensions ) const
    { return reshape< sizeof...(Dimensions), Dimensions... >( dimensions... ); }

    // Returns a slice, along SliceDim
    template< int slice_dim >
    TensorMap< Const<ScalType>, dim-1 > slice( int idx ) const
    {
        const auto& op = *static_cast< const TensorOperator<Derived,dim,slice_dim>* >(this);
        return op( idx );
    }
    template< int slice_dim >
    TensorMap<ScalType,dim-1> slice( int idx )
    {
        auto& op = *static_cast< TensorOperator<Derived,dim,slice_dim>* >(this);
        return op( idx );
    }

    // Equivalent to tensor block
    template< int slice_dim >
    TensorMap<Const<ScalType>,dim> middleSlices( int begin, int sz ) const
    {
        static_assert( slice_dim < dim, "Invalid slice dimension" );
        assert( begin >= 0 && begin+sz <= derived().shape(slice_dim)
                && "Invalid indices" );
        TensorMap<Const<ScalType>,dim> res( *this );
        res.set_data( res.data() + begin * stride( slice_dim ) );
        res.set_shape( slice_dim, sz );
        return res;
    }
    template< int slice_dim >
    TensorMap<ScalType,dim> middleSlices( int begin, int sz )
    {
        static_assert( slice_dim < dim, "Invalid slice dimension" );
        assert( begin >= 0 && begin+sz <= derived().shape(slice_dim)
                && "Invalid indices" );
        TensorMap<ScalType,dim> res( *this );
        res.set_data( res.data() + begin * stride( slice_dim ) );
        res.set_shape( slice_dim, sz );
        return res;
    }

    // --- Utility methods ---

    bool ravelable() const
    {
        const Derived& d = derived();
        for (int i = 0; i < dim - 1; ++i)
        {
            if ( d.stride(i) != d.shape(i+1) * d.stride(i+1) )
                return false;
        }
        return true;
    }

    int size() const
    {
        int s = 1;
        for (int i = 0; i < dim; ++i)
            s *= derived().shape(i);
        return s;
    }

    // Returns the pointer to the data after the contained buffer
    // Note: this is not the data()+size() pointer, it takes the strides into account
    template< typename Subst = ScalType >
    inline ScalType* next_data( EnableIf< !IsConst<Subst>() > = {} )
    { return derived().data() + maxMemorySize(); }

    inline Const<ScalType>* next_data() const
    { return derived().data() + maxMemorySize(); }

    int maxMemorySize() const
    {
        const Derived& d = derived();
        int res = 0;
        for ( int s = 0 ; s < dim ; ++s )
            res = std::max( res, d.stride(s)*d.shape(s) );
        return res;
    }

    inline bool empty() const
    { return derived().data() == nullptr; }

    // Returns the dimension whose stride is the biggest
    int outerDim() const
    {
        int biggestStride = 0;
        int res = 0;
        const Derived& d = derived();
        for ( int i = 0 ; i < dim ; ++i )
        {
            int currentStride = d.stride(i);
            if ( currentStride > biggestStride )
            {
                biggestStride = currentStride;
                res = i;
            }
        }
        return res;
    }

    inline int innerStride() const
    {
        return derived().stride(dim-1);
    }

    // Returns a Map of the corresponding col vector
    Eigen::Map< ConstAs<ScalType,Vector<NonConst<ScalType>>>, Eigen::Unaligned, DynInnerStride >
    dyn_ravel()
    {
        assert( ravelable() && "Cannot be raveled" );
        return Eigen::Map<ConstAs<ScalType,Vector<NonConst<ScalType>>>,
                Eigen::Unaligned, DynInnerStride>(
                        derived().data(),
                        size(),
                        DynInnerStride( derived().stride(dim - 1) ) );
    }
    Eigen::Map< const Vector<NonConst<ScalType>>, Eigen::Unaligned, DynInnerStride >
    dyn_ravel() const
    {
        assert( ravelable() && "Cannot be raveled" );
        return Eigen::Map<const Vector<NonConst<ScalType>>,
                Eigen::Unaligned, DynInnerStride>(
                        derived().data(),
                        size(),
                        DynInnerStride( derived().stride(dim-1) ) );
    }

    // Returns a Map of the corresponding col vector, with an inner stride of 1
    Eigen::Map< ConstAs<ScalType,Vector<NonConst<ScalType>>>, Eigen::Unaligned >
    ravel()
    {
        assert( innerStride() == 1 && ravelable() && "Cannot be raveled" );
        return Eigen::Map<ConstAs<ScalType,Vector<NonConst<ScalType>>>,
                Eigen::Unaligned>( derived().data(), size() );
    }
    Eigen::Map< const Vector<NonConst<ScalType>>, Eigen::Unaligned >
    ravel() const
    {
        assert( innerStride() == 1 && ravelable() && "Cannot be raveled" );
        return Eigen::Map<const Vector<NonConst<ScalType>>,
                Eigen::Unaligned>( derived().data(), size() );
    }

    // Contracts ContractDim with ContractDim+1 dimensions
    template< int contract_dim,
        typename Subst = ScalType, typename = EnableIf<!IsConst<Subst>()> >
    TensorMap<ScalType,dim-1>
    contract()
    {
        static_assert( contract_dim < dim-1,
                "Cannot contract this dimension with the next one" );
        Derived& d = derived();
        assert( d.stride(contract_dim) ==
                d.stride(contract_dim+1) * d.shape(contract_dim+1)
                && "Cannot be trivially contracted" );

        TensorMap<ScalType,dim-1> new_tensor( (EmptyConstructor()) );

        new_tensor.set_data( d.data() );

        for ( int s = 0 ; s < contract_dim ; ++s )
        {
            new_tensor.set_shape( s, d.shape(s) );
            new_tensor.set_stride( s, d.stride(s) );
        }

        new_tensor.set_shape( contract_dim, d.shape(contract_dim)*d.shape(contract_dim+1) );
        new_tensor.set_stride( contract_dim, d.stride(contract_dim+1) );

        for ( int s = contract_dim+1 ; s < dim-1 ; ++s )
        {
            new_tensor.set_shape( s, d.shape(s+1) );
            new_tensor.set_stride( s, d.stride(s+1) );
        }

        return new_tensor;
    }

    template< int contract_dim >
    TensorMap<Const<ScalType>,dim-1>
    contract() const
    {
        static_assert( contract_dim < dim-1,
                "Cannot contract this dimension with the next one" );
        const Derived& d = derived();
        assert( d.stride(contract_dim) ==
                d.stride(contract_dim+1) * d.shape(contract_dim+1)
                && "Cannot be trivially contracted" );

        TensorMap<Const<ScalType>,dim-1> new_tensor( (EmptyConstructor()) );

        new_tensor.set_data( d.data() );

        for ( int s = 0 ; s < contract_dim ; ++s )
        {
            new_tensor.set_shape( s, d.shape(s) );
            new_tensor.set_stride( s, d.stride(s) );
        }

        new_tensor.set_shape( contract_dim, d.shape(contract_dim)*d.shape(contract_dim+1) );
        new_tensor.set_stride( contract_dim, d.stride(contract_dim+1) );

        for ( int s = contract_dim+1 ; s < dim-1 ; ++s )
        {
            new_tensor.set_shape( s, d.shape(s+1) );
            new_tensor.set_stride( s, d.stride(s+1) );
        }

        return new_tensor;
    }

    TensorMap<ScalType,dim-1> contractFirst()
    { return contract<0>(); }
    TensorMap<Const<ScalType>,dim-1> contractFirst() const
    { return contract<0>(); }

    TensorMap<ScalType,dim-1> contractLast()
    { return contract<dim-2>(); }
    TensorMap<Const<ScalType>,dim-1> contractLast() const
    { return contract<dim-2>(); }
};

// ----- TensorDim -----

template< typename Derived, int _dim = Traits<Derived>::dim >
class TensorDim : public TensorBase<Derived>
{
public:
    typedef TensorBase<Derived> Base;
    friend Base;

    using Base::Base;
    using Base::dim;
    using Base::owns;
    using typename Base::ScalType;
    using typename Base::MatrixRef;
    using typename Base::VectorRef;
    using Base::derived;

    inline int shape( int s ) const
    { return m_shape[s]; }

    inline int stride( int s ) const
    { return m_stride[s]; }

    inline ShapeMap<dim,int> shape() const
    { return ShapeMap<dim,int>( m_shape ); }

    inline StrideMap<dim,int> stride() const
    { return StrideMap<dim,int>( m_stride ); }

    inline ScalType* data()
    { return m_data; }

    inline Const<ScalType>* data() const
    { return m_data; }


protected:
    inline void set_shape( int s, int value )
    { m_shape[s] = value; }

    inline void set_stride( int s, int value )
    { m_stride[s] = value; }

    inline void set_data( ScalType* value )
    { m_data = value; }

protected:
    ScalType* m_data;
    int m_shape[dim];
    int m_stride[dim];
};

// ----- TensorDim<2> -----

// We mess up encapsulation here
class PublicDynStride : public DynStride
{
public:
    typedef Eigen::Index Index;
    enum
    {
        InnerStrideAtCompileTime = Eigen::Dynamic,
        OuterStrideAtCompileTime = Eigen::Dynamic
    };

    PublicDynStride()
     : DynStride(0,0)
    { }

    //TODO define friendship instead of letting this public
    void set_inner( int i )
    { DynStride::m_inner.setValue(i); }

    void set_outer( int i )
    { DynStride::m_outer.setValue(i); }
};

template< typename Derived >
class TensorBase_MatrixLike :
    public Eigen::Map<
        ConstAs< typename Traits<Derived>::ScalType,
            MatrixRM< NonConst<typename Traits<Derived>::ScalType> > >,
        Eigen::Unaligned, PublicDynStride >
{
public:
    typedef Eigen::Map<
        ConstAs< typename Traits<Derived>::ScalType,
            MatrixRM< NonConst<typename Traits<Derived>::ScalType> > >,
        Eigen::Unaligned, PublicDynStride >
        Base;

    typedef typename Traits<Derived>::ScalType ScalType;

    using Base::data;
    using Base::operator=;

    // Must define default constructor...
    TensorBase_MatrixLike()
     : Base( nullptr, 0, 0, PublicDynStride() )
    {}

    inline int shape( int s ) const
    {
        if ( s == 0 )
            return Base::rows();
        else if ( s == 1 )
            return Base::cols();
        assert( "Invalid requested shape" );
        return -1;
    }

    inline int stride( int s ) const
    {
        if ( s == 0 )
            return Base::outerStride();
        else if ( s == 1 )
            return Base::innerStride();
        assert( "Invalid requested stride" );
        return -1;
    }

    inline ShapeOwn<2,int> shape() const
    {
        return ShapeOwn<2,int>( Base::rows(), Base::cols() );
    }

    inline StrideOwn<2,int> stride() const
    {
        return StrideOwn<2,int>(
                Base::outerStride(),
                Base::innerStride() );
    }

protected:
    typedef Eigen::internal::variable_if_dynamic<typename Base::Index, Eigen::Dynamic> AttrType;

    inline void set_shape( int s, int value )
    {
        if ( s == 0 )
            const_cast< AttrType* >(&(Eigen::MapBase<Base>::m_rows))->setValue(value);
        else if ( s == 1 )
            const_cast< AttrType* >(&(Eigen::MapBase<Base>::m_cols))->setValue(value);
        assert( "Invalid requested shape" );
    }

    inline void set_stride( int s, int value )
    {
        if ( s == 0 )
            Base::m_stride.set_outer( value );
        else if ( s == 1 )
            Base::m_stride.set_inner( value );
        assert( "Invalid requested stride" );
    }

    inline void set_data( ScalType* value )
    { Eigen::MapBase<Base>::m_data = value; }
};

// The order of inheritance is important here
// We want to call the default constructor of TensorBase_MatrixLike
// before calling any constructor of TensorBase
template< typename Derived >
class TensorDim<Derived,2> :
    public TensorBase_MatrixLike<Derived>,
    public TensorBase<Derived>
{
public:
    typedef TensorBase<Derived> Base;
    typedef TensorBase_MatrixLike<Derived> AttrBase;

    friend Base;
    friend AttrBase;

    using Base::Base;
    using Base::dim;
    using Base::owns;
    using typename Base::ScalType;
    using typename Base::MatrixRef;
    using typename Base::VectorRef;
    using Base::derived;
    using AttrBase::data;
    using AttrBase::size;
    using AttrBase::shape;
    using AttrBase::stride;
    using AttrBase::operator=;

protected:
    using AttrBase::set_data;
    using AttrBase::set_shape;
    using AttrBase::set_stride;

public:
    // Use these functions to explicitly get an Eigen::Map
    template< typename Subst = ScalType, typename = EnableIf< !IsConst<Subst>() > >
    Eigen::Map< MatrixRM<ScalType>, Eigen::Unaligned, DynStride >
    map()
    {
        return Eigen::Map< MatrixRM<ScalType>, Eigen::Unaligned, DynStride >(
                derived().data(), derived().shape(0), derived().shape(1),
                DynStride(derived().stride(0), derived().stride(1)) );
    }

    Eigen::Map< const MatrixRM< NonConst<ScalType> >, Eigen::Unaligned, DynStride >
    const_map() const
    {
        return Eigen::Map< const MatrixRM< NonConst<ScalType> >, Eigen::Unaligned, DynStride >(
                derived().data(), derived().shape(0), derived().shape(1),
                DynStride(derived().stride(0), derived().stride(1)) );
    }

    // Same with contiguity guarantee
    template< typename Subst = ScalType, typename = EnableIf< !IsConst<Subst>() > >
    Eigen::Map< MatrixRM<ScalType>, Eigen::Unaligned, Eigen::OuterStride<> >
    ref()
    {
        assert( derived().stride(dim-1) == 1 && "This map is not contiguous" );
        return Eigen::Map< MatrixRM<ScalType>, Eigen::Unaligned, Eigen::OuterStride<> >(
                derived().data(), derived().shape(0), derived().shape(1),
                Eigen::OuterStride<>(derived().stride(0)) );
    }

    Eigen::Map< const MatrixRM< NonConst<ScalType> >, Eigen::Unaligned, Eigen::OuterStride<> >
    const_ref() const
    {
        assert( derived().stride(dim-1) == 1 && "This map is not contiguous" );
        return Eigen::Map< const MatrixRM< NonConst<ScalType> >, Eigen::Unaligned, Eigen::OuterStride<> >(
                derived().data(), derived().shape(0), derived().shape(1),
                Eigen::OuterStride<>(derived().stride(0)) );
    }
};

// ----- TensorDim<1> -----

// We mess up encapsulation here
class PublicDynInnerStride : public DynInnerStride
{
public:
    typedef Eigen::Index Index;
    enum
    {
        InnerStrideAtCompileTime = Eigen::Dynamic,
        OuterStrideAtCompileTime = 0
    };

    PublicDynInnerStride()
     : DynInnerStride(0)
    { }

    //TODO define friendship instead of letting this public
    void set_inner( int i )
    { DynInnerStride::m_inner.setValue(i); }
};

template< typename Derived >
class TensorBase_VectorLike :
    public Eigen::Map<
        ConstAs< typename Traits<Derived>::ScalType,
            Vector< NonConst<typename Traits<Derived>::ScalType> > >,
        Eigen::Unaligned, PublicDynInnerStride >
{
public:
    typedef Eigen::Map<
        ConstAs< typename Traits<Derived>::ScalType,
            Vector< NonConst<typename Traits<Derived>::ScalType> > >,
        Eigen::Unaligned, PublicDynInnerStride >
        Base;

    typedef typename Traits<Derived>::ScalType ScalType;

    using Base::data;
    using Base::operator=;

    // Must define default constructor...
    TensorBase_VectorLike()
     : Base( nullptr, 0, PublicDynInnerStride() )
    {}

    inline int shape( int s ) const
    {
        if ( s == 0 )
            return Base::rows();
        assert( "Invalid requested shape" );
        return -1;
    }

    inline int stride( int s ) const
    {
        if ( s == 0 )
            return Base::innerStride();
        assert( "Invalid requested stride" );
        return -1;
    }

    inline ShapeOwn<1,int> shape() const
    {
        return ShapeOwn<1,int>( Base::rows() );
    }

    inline StrideOwn<1,int> stride() const
    {
        return StrideOwn<1,int>( Base::innerStride() );
    }

protected:
    typedef Eigen::internal::variable_if_dynamic<typename Base::Index, Eigen::Dynamic> AttrType;

    inline void set_shape( int s, int value )
    {
        if ( s == 0 )
            const_cast< AttrType* >(&(Eigen::MapBase<Base>::m_rows))->setValue(value);
        assert( "Invalid requested shape" );
    }

    inline void set_stride( int s, int value )
    {
        if ( s == 0 )
            Base::m_stride.set_inner( value );
        assert( "Invalid requested stride" );
    }

    inline void set_data( ScalType* value )
    { Eigen::MapBase<Base>::m_data = value; }
};

// The order of inheritance is important here
// We want to call the default constructor of TensorBase_VectorLike
// before calling any constructor of TensorBase
template< typename Derived >
class TensorDim<Derived,1> :
    public TensorBase_VectorLike<Derived>,
    public TensorBase<Derived>
{
public:
    typedef TensorBase<Derived> Base;
    typedef TensorBase_VectorLike<Derived> AttrBase;

    friend Base;
    friend AttrBase;

    using Base::Base;
    using Base::dim;
    using Base::owns;
    using typename Base::ScalType;
    using typename Base::MatrixRef;
    using typename Base::VectorRef;
    using Base::derived;
    using AttrBase::data;
    using AttrBase::size;
    using AttrBase::shape;
    using AttrBase::stride;
    using AttrBase::operator=;

protected:
    using AttrBase::set_data;
    using AttrBase::set_shape;
    using AttrBase::set_stride;

public:
    // Use these functions to explicitly get an Eigen::Map
    template< typename Subst = ScalType, typename = EnableIf< !IsConst<Subst>() > >
    Eigen::Map< Vector<ScalType>, Eigen::Unaligned, DynInnerStride >
    map()
    {
        return Eigen::Map< Vector<ScalType>, Eigen::Unaligned, DynInnerStride >(
                derived().data(), derived().shape(0),
                DynInnerStride( derived().stride(0) ) );
    }

    Eigen::Map< const MatrixRM< NonConst<ScalType> >, Eigen::Unaligned, DynInnerStride >
    const_map() const
    {
        return Eigen::Map< const Vector< NonConst<ScalType> >, Eigen::Unaligned, DynInnerStride >(
                derived().data(), derived().shape(0),
                DynInnerStride( derived().stride(0) ) );
    }

    // Same with contiguity guarantee
    template< typename Subst = ScalType, typename = EnableIf< !IsConst<Subst>() > >
    Eigen::Map< Vector<ScalType> >
    ref()
    {
        assert( derived().stride(dim-1) == 1 && "This map is not contiguous" );
        return Eigen::Map< Vector<ScalType> >(
                derived().data(), derived().shape(0) );
    }

    Eigen::Map< const Vector< NonConst<ScalType> > >
    const_ref() const
    {
        assert( derived().stride(dim-1) == 1 && "This map is not contiguous" );
        return Eigen::Map< const Vector< NonConst<ScalType> > >(
                derived().data(), derived().shape(0) );
    }
};

// ----- TensorOperator -----

template< typename Derived, int _dim, int _current_dim >
class TensorOperator :
    public TensorOperator< Derived, _dim, _current_dim+1 >
{
    static_assert( _current_dim >= 0 && _current_dim <= _dim,
           "Wrong usage of TensorOperator" );

public:
    typedef TensorOperator< Derived, _dim, _current_dim+1 > Base;

    template< typename OtherDerived, int other_dim, int other_current_dim >
    friend class TensorOperator;

    using Base::Base;
    using Base::dim;
    using Base::owns;
    using typename Base::MatrixRef;
    using typename Base::VectorRef;
    using Base::derived;
    using Base::data;
    using Base::shape;
    using Base::stride;
    using Base::operator=;

    typedef typename Traits<Derived>::ScalType ScalType;

    inline TensorOperator< Derived, dim, _current_dim+1 >& operator()(void)
    { return *this; }

    inline const TensorOperator< Derived, dim, _current_dim+1 >& operator()(void) const
    { return *this; }

    template< typename ... Dimensions,
        typename = EnableIf<
            (sizeof...(Dimensions)<=dim-_current_dim)
            && (sizeof...(Dimensions)<dim) > >
    TensorOperator<
        TensorMap<ScalType,dim-sizeof...(Dimensions)>,
        dim-sizeof...(Dimensions),
        _current_dim >
    operator()( Dimensions ... dimensions )
    {
        TensorOperator<
            TensorMap<ScalType,dim-sizeof...(Dimensions)>,
            dim-sizeof...(Dimensions),
            _current_dim > new_tensor( (EmptyConstructor()) );

        new_tensor.derived().set_data(
                derived().data() +
                Base::template compute_offset<_current_dim>( dimensions... ) );

        for ( int s = 0 ; s < _current_dim ; ++s )
        {
            new_tensor.derived().set_shape( s, derived().shape(s) );
            new_tensor.derived().set_stride( s, derived().stride(s) );
        }

        for ( int s = _current_dim+sizeof...(Dimensions) ; s < dim ; ++s )
        {
            new_tensor.derived().set_shape( s-sizeof...(Dimensions), derived().shape(s) );
            new_tensor.derived().set_stride( s-sizeof...(Dimensions), derived().stride(s) );
        }

        return new_tensor;
    }

    template< typename ... Dimensions,
        typename = EnableIf<
            (sizeof...(Dimensions)<=dim-_current_dim)
            && (sizeof...(Dimensions)<dim) > >
    TensorOperator<
        TensorMap<Const<ScalType>,dim-sizeof...(Dimensions)>,
        dim-sizeof...(Dimensions),
        _current_dim >
    operator()( Dimensions ... dimensions ) const
    {
        TensorOperator<
            TensorMap<Const<ScalType>,dim-sizeof...(Dimensions)>,
            dim-sizeof...(Dimensions),
            _current_dim > new_tensor( (EmptyConstructor()) );

        new_tensor.derived().set_data(
                derived().data() +
                Base::template compute_offset<_current_dim>( dimensions... ) );

        for ( int s = 0 ; s < _current_dim ; ++s )
        {
            new_tensor.derived().set_shape( s, derived().shape(s) );
            new_tensor.derived().set_stride( s, derived().stride(s) );
        }

        for ( int s = _current_dim+sizeof...(Dimensions) ; s < dim ; ++s )
        {
            new_tensor.derived().set_shape( s-sizeof...(Dimensions), derived().shape(s) );
            new_tensor.derived().set_stride( s-sizeof...(Dimensions), derived().stride(s) );
        }

        return new_tensor;
    }

    template< typename ... Dimensions,
        typename = EnableIf<
            (sizeof...(Dimensions)<=dim-_current_dim)
            && (sizeof...(Dimensions)==dim) > >
    ScalType&
    operator()( Dimensions ... dimensions )
    { return *(derived().data() + Base::template compute_offset<_current_dim>( dimensions... )); }

    template< typename ... Dimensions,
        typename = EnableIf<
            (sizeof...(Dimensions)<=dim-_current_dim)
            && (sizeof...(Dimensions)==dim) > >
    Const<ScalType>&
    operator()( Dimensions ... dimensions ) const
    { return *(derived().data() + Base::template compute_offset<_current_dim>( dimensions... )); }

protected:
    using Base::set_data;
    using Base::set_shape;
    using Base::set_stride;
};

template< typename Derived, int _dim >
class TensorOperator< Derived, _dim, _dim > :
    public TensorDim<Derived>
{
public:
    typedef TensorDim<Derived> Base;
    friend Base;

    template< typename OtherDerived, int other_dim, int other_current_dim >
    friend class TensorOperator;

    using Base::Base;
    using Base::dim;
    using Base::owns;
    using typename Base::ScalType;
    using typename Base::MatrixRef;
    using typename Base::VectorRef;
    using Base::derived;
    using Base::data;
    using Base::shape;
    using Base::stride;
    using Base::operator=;

protected:
    using Base::set_data;
    using Base::set_shape;
    using Base::set_stride;

    template< int s, typename ... OtherDims >
    int compute_offset( int i, OtherDims ... other_dimensions ) const
    {
        assert( i < derived().shape(s) && "Index out of shape" );
        return i*derived().stride(s) + compute_offset<s+1>( other_dimensions... );
    }

    template< int s >
    inline int compute_offset() const
    { return 0; }
};

// ----- TensorMap -----

template< typename Derived >
class TensorMapBase : public TensorOperator<Derived>
{
public:
    typedef TensorOperator<Derived> Base;
    friend Base;
    using Base::Base;
    using Base::dim;
    using Base::owns;
    typedef typename Traits<Derived>::ScalType ScalType;
    using typename Base::MatrixRef;
    using typename Base::VectorRef;
    using Base::derived;
    using Base::data;
    using Base::shape;
    using Base::stride;
    using Base::operator=;

protected:
    using Base::set_data;
    using Base::set_shape;
    using Base::set_stride;

public:
    // Default constructor
    TensorMapBase()
    {
        derived().set_data( nullptr );
        for ( int s = 0 ; s < dim ; ++s )
        {
            derived().set_shape(s,0);
            derived().set_stride(s,0);
        }
    }

    // You can specify the inner stride...
    template< typename ... Dimensions, typename = EnableIf<sizeof...(Dimensions)==dim> >
    TensorMapBase<Derived>( ScalType* data, const InnerStride& inner_stride, Dimensions ... dimensions )
    {
        derived().set_data( data );
        derived().set_stride( dim-1, inner_stride.inner );
        Base::template init_sns_from_shape<0>(dimensions...);
    }

    // ... or let it to 1 in the default case
    template< typename ... Dimensions, typename = EnableIf<sizeof...(Dimensions)==dim> >
    inline TensorMapBase( ScalType* data, Dimensions ... dimensions )
     : TensorMapBase( data, InnerStride(1), dimensions... )
    { }

    // You can also specify both shape and stride
    template< typename ShapeDerived, typename StrideDerived,
       typename = EnableIf<
          Traits<ShapeDerived>::dim == dim
          && Traits<StrideDerived>::dim == dim > >
    TensorMapBase( ScalType* data,
                   const Shape<ShapeDerived>& custom_shape,
                   const Stride<StrideDerived>& custom_stride )
    {
        derived().set_data( data );
        for ( int s = 0 ; s < dim ; ++s )
        {
            derived().set_shape( s, custom_shape[s] );
            derived().set_stride( s, custom_stride[s] );
        }
    }

    template< typename OtherDerived, typename = EnableIf<
        Traits<OtherDerived>::dim == dim > >
    TensorMapBase( const TensorBase<OtherDerived>& other )
    {
        derived().set_data( other.derived().data() );
        for ( int s = 0 ; s < dim ; ++s )
        {
            derived().set_shape( s, other.derived().shape(s) );
            derived().set_stride( s, other.derived().stride(s) );
        }
    }

    template< typename OtherDerived, typename = EnableIf<
        !IsConst< typename Traits<OtherDerived>::ScalType >()
        && Traits<OtherDerived>::dim == dim > >
    TensorMapBase( TensorBase< OtherDerived >& other )
    {
        derived().set_data( other.derived().data() );
        for ( int s = 0 ; s < dim ; ++s )
        {
            derived().set_shape( s, other.derived().shape(s) );
            derived().set_stride( s, other.derived().stride(s) );
        }
    }

    template< typename OtherDerived, typename = EnableIf<
        !IsConst< typename Traits<OtherDerived>::ScalType >()
        && Traits<OtherDerived>::dim == dim > >
    TensorMapBase( TensorBase< OtherDerived >&& other )
    {
        derived().set_data( other.derived().data() );
        for ( int s = 0 ; s < dim ; ++s )
        {
            derived().set_shape( s, other.derived().shape(s) );
            derived().set_stride( s, other.derived().stride(s) );
        }
    }

    /* UNCOMMENT ME
    // This is used to allocate the tensor-map through a BufMap
    template< typename ... Dimensions,
        typename = EnableIf< sizeof...(Dimensions) == dim > >
    TensorMapBase( DxyzUtils::BufMap< NonConst<ScalType> >& bufmap, Dimensions ... dimensions )
     : TensorMapBase( bufmap.ptr(total_size(dimensions...)), dimensions... )
    {}
    */

    template< typename OtherDerived, typename ... Dimensions,
        typename = EnableIf< sizeof...(Dimensions) == dim > >
    TensorMapBase( const TensorBase< OtherDerived >& other, Dimensions ... dimensions )
    {
        Base::template init_sns_reshape_tensor(
                other.data(),
                other.shape(),
                other.stride(),
                dimensions... );
    }

    template< typename OtherDerived, typename ... Dimensions,
        typename = EnableIf< sizeof...(Dimensions) == dim > >
    TensorMapBase( TensorBase< OtherDerived >& other, Dimensions ... dimensions )
    {
        Base::template init_sns_reshape_tensor(
                other.data(),
                other.derived().shape(),
                other.derived().stride(),
                dimensions... );
    }

    // This can take inner and outer strides into account (like blocks)
    template<typename ... Dimensions,
        typename = EnableIf< sizeof...(Dimensions) == dim > >
    TensorMapBase( MatrixRef&& mat, Dimensions ... dimensions )
    {
        typedef typename MatrixRef::Index Index;

        Base::template init_sns_reshape_tensor(
                mat.data(),
                ShapeOwn<2,Index>( mat.rows(), mat.cols() ),
                StrideOwn<2,Index>( mat.outerStride(), mat.innerStride() ),
                dimensions... );
    }
};

// ----- TensorMap -----

template< typename _ScalType, int _dim >
class TensorMap : public TensorMapBase< TensorMap<_ScalType,_dim> >
{
public:
    typedef TensorMapBase<TensorMap> Base;
    friend Base;

    template< typename Derived >
    friend class TensorBase;

    template< typename OtherDerived, int other_dim, int other_current_dim >
    friend class TensorOperator;

    using Base::Base;
    using Base::dim;
    using Base::owns;
    using typename Base::ScalType;
    using typename Base::MatrixRef;
    using typename Base::VectorRef;
    using Base::derived;
    using Base::data;
    using Base::shape;
    using Base::stride;
    using Base::operator=;

protected:
    using Base::set_data;
    using Base::set_shape;
    using Base::set_stride;
};

template< typename _ScalType, int _dim >
class TensorOwn : public TensorMap<_ScalType, _dim>
{
public:
    typedef TensorMap<_ScalType,_dim> Base;
    using Base::dim;
    using Base::owns;
    using typename Base::ScalType;
    using typename Base::MatrixRef;
    using typename Base::VectorRef;
    using Base::operator=;

    TensorOwn()
     : Base()
    {}

    template< typename ... Dimensions, typename = EnableIf<sizeof...(Dimensions)==dim> >
    TensorOwn( Dimensions ... dimensions )
     : Base( new ScalType[ Base::total_size(dimensions...) ], dimensions... )
    {}

    ~TensorOwn()
    {
        // deleting nullptr has no effect
        delete [] Base::data();
    }

    template< typename ... Dimensions, typename = EnableIf<sizeof...(Dimensions)==dim> >
    void resize( Dimensions ... dimensions )
    {
        int sz = Base::size();
        int new_sz = Base::total_size( dimensions... );
        if ( new_sz != sz )
        {
            // deleting nullptr has no effect
            delete [] Base::data();
            Base::set_data( new ScalType[ new_sz ] );
        }
        Base::set_stride( dim-1, 1 );
        Base::template init_sns_from_shape<0>( dimensions... );
    }
};

// ----- Method implementations -----

template< typename Derived >
template< int new_dim, typename ... Dimensions, typename >
TensorMap< typename TensorBase<Derived>::ScalType, new_dim >
TensorBase<Derived>::reshape( Dimensions ... dimensions )
{
    Derived& d = derived();
    TensorMap< ScalType, new_dim > new_tensor( (EmptyConstructor()) );
    new_tensor.init_sns_reshape_tensor(
            d.data(),
            d.shape(), d.stride(),
            dimensions... );
    return new_tensor;
}

template< typename Derived >
template< int new_dim, typename ... Dimensions, typename >
TensorMap< Const<typename TensorBase<Derived>::ScalType>, new_dim >
TensorBase<Derived>::reshape( Dimensions ... dimensions ) const
{
    const Derived& d = derived();
    TensorMap< Const<ScalType>, new_dim > new_tensor( (EmptyConstructor()) );
    new_tensor.init_sns_reshape_tensor(
            d.data(),
            d.shape(), d.stride(),
            dimensions... );
    return new_tensor;
}

} // namespace TensorMapTools

template< typename ScalType, int dim >
using TensorMap = TensorMapTools::TensorMap<ScalType,dim>;

template< typename ScalType, int dim >
using TensorOwn = TensorMapTools::TensorOwn<ScalType,dim>;

template< typename TensorDerived >
using TensorBase = TensorMapTools::TensorBase<TensorDerived>;

template< typename TensorDerived, int dim >
using TensorDim = TensorMapTools::TensorDim<TensorDerived,dim>;

#ifdef _MSC_VER
#pragma warning( pop )
#endif

#endif // TENSOR_MAP_HPP
