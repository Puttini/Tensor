#include <Eigen/Core>
#include <cassert>

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

    Integer get( int i )
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

    inline Integer get( int i )
    { return shape[i]; }
};

// ----- Stride tool structures -----

// Eigen dynamic strides
typedef Eigen::InnerStride<Eigen::Dynamic> DynInnerStride;
typedef Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> DynStride;

// Eigen matrix and vector
// REMOVE ME
template< typename ScalType, int rows = Eigen::Dynamic, int cols = Eigen::Dynamic >
using MatrixRM = Eigen::Matrix<ScalType,rows,cols,Eigen::RowMajor>;

template< typename ScalType, int rows = Eigen::Dynamic, int cols = Eigen::Dynamic >
using MatrixCM = Eigen::Matrix<ScalType,rows,cols,Eigen::ColMajor>;

template< typename ScalType, int size = Eigen::Dynamic >
using Vector = MatrixCM<ScalType,size,1>;

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

    Integer get( int i )
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

    template< typename = EnableIf<dim==2> >
    StrideOwn( const DynStride& other )
    {
        stride[0] = other.outer();
        stride[1] = other.inner();
    }

    template< typename = EnableIf<dim==1> >
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

    Integer get( int i )
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
    TensorBase( EmptyConstructor ) {}

    // Used to initialize strides and shapes in constructors
    // (inner stride must be specified before calling this)
    template< int s, typename ... OtherDimensions >
    inline void init_sns_from_shape( int dimension, OtherDimensions ... other_dimensions )
    {
        Derived& d = derived();
        d.set_shape( s, dimension );
        init_sns_from_shape<s+1>( other_dimensions... );
        d.set_stride( dimension, d.shape(dimension + 1) * d.stride(dimension + 1) );
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
    template< int s, int other_s,
        typename ShapeDerived, typename StrideDerived,
        typename ... Dimensions, typename = EnableIf<(s>=0 && other_s>=0)> >
    void init_sns_reshape_tensor(
            const Shape<ShapeDerived>& other_shape,
            const Stride<StrideDerived>& other_stride,
            int current_total_size,
            int other_current_total_size,
            Dimensions ... dimensions )
    {
        static_assert(
                Traits<ShapeDerived>::dim == dim && Traits<StrideDerived>::dim == dim,
                "Invalid stride/shape dimension" );

        int new_total_size = current_total_size*nth_of_pack<s>(dimensions...);
        int other_new_total_size = other_current_total_size*other_shape[other_s];

        Derived& d = derived();
        d.set_shape( s, nth_of_pack<s>(dimensions...) );

        if ( new_total_size == other_new_total_size )
        {
            d.set_stride( s, other_stride[other_s] );

            if ( s > 0 && other_s > 0)
            {
                init_sns_reshape_tensor<s-1,other_s-1,ShapeDerived,StrideDerived>(
                        other_shape, other_stride,
                        new_total_size,
                        other_new_total_size,
                        dimensions... );
            }
            else if ( s > 0 )
            {
                init_sns_reshape_tensor<s-1,other_s,ShapeDerived,StrideDerived>(
                        other_shape, other_stride,
                        new_total_size,
                        other_current_total_size,
                        dimensions... );
            }
            else if ( other_s > 0 )
            {
                init_sns_reshape_tensor<s,other_s-1,ShapeDerived,StrideDerived>(
                        other_shape, other_stride,
                        current_total_size,
                        other_new_total_size,
                        dimensions... );
            }
        }
        else if ( new_total_size > other_new_total_size )
        {
            // Split other dimension. The strides must be compatible
            assert( other_s > 0
                    && other_shape[other_s]*other_stride[other_s] == other_stride[other_s-1]
                    && "Incompatible stride/shape" );

            init_sns_reshape_tensor<s,other_s-1,ShapeDerived,StrideDerived>(
                    other_shape, other_stride,
                    current_total_size,
                    other_new_total_size,
                    dimensions... );
        }
        else // new_total_size < other_new_total_size
        {
            // Split this dimension. The strides have no constraint
            d.set_stride( s, (s == dim-1)
                             ? other_stride.innerStride()
                             : d.stride(s+1) * d.shape(s+1) );

            init_sns_reshape_tensor<s-1,other_s,ShapeDerived,StrideDerived>(
                    other_shape, other_stride,
                    new_total_size,
                    other_current_total_size,
                    dimensions... );
        }
    }

    // Fortunately we can overload the base case of this function by using 'const'
    template< int s, int other_s,
        typename ShapeDerived, typename StrideDerived,
        typename ... Dimensions, typename = EnableIf<(s<0||other_s<0)> >
    inline void init_sns_reshape_tensor(
            const Shape<ShapeDerived>& other_shape,
            const Stride<StrideDerived>& other_stride,
            int current_total_size,
            int other_current_total_size,
            Dimensions ... dimensions ) const
    {}

public:
    //TODO put it on the top
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
    template< typename ... Dimensions, typename Subst = ScalType >
    inline TensorMap< ScalType, sizeof...(Dimensions) >
    reshape( Dimensions ... dimensions, EnableIf< !IsConst<Subst>() > = {}  )
    { return reshape< sizeof...(Dimensions), Dimensions... >( dimensions... ); }

    template< typename ... Dimensions >
    inline TensorMap< Const<ScalType>, sizeof...(Dimensions) >
    reshape( Dimensions ... dimensions ) const
    { return reshape< sizeof...(Dimensions), Dimensions... >( dimensions... ); }

protected:
    /*
    template< typename OtherType >
    void copy_move_constructor( OtherType other )
    {
        data_ = other.data_;
        std::copy(other.stride_, other.stride_ + dim, stride_);
        std::copy(other.shape_, other.shape_ + dim, shape_);
    }

    template< int SliceDim, typename SuperType >
    void slice_constructor( const Slice<SliceDim>& slice, SuperType super )
    {
        static_assert(SliceDim <= dim, "Slice used on invalid dimension");
        assert(slice.idx < super.shape_[SliceDim] && "Index out of shape");
        data_ = super.data_ + slice.idx * super.stride_[SliceDim];
        std::copy(super.stride_, super.stride_ + SliceDim, stride_);
        std::copy(super.shape_, super.shape_ + SliceDim, shape_);
        std::copy(super.stride_ + SliceDim + 1, super.stride_ + dim + 1, stride_ + SliceDim);
        std::copy(super.shape_ + SliceDim + 1, super.shape_ + dim + 1, shape_ + SliceDim);
    }
    */

    template< int contract_dim, typename ShapeDerived, typename StrideDerived,
        typename = EnableIf< (contract_dim<dim) > >
    void init_sns_from_contraction(
            const Shape<ShapeDerived>& initial_shape,
            const Stride<StrideDerived>& initial_stride )
    {
        static_assert( ShapeDerived::dim == dim+1 && StrideDerived::dim == dim+1,
                "Invalid shape/stride" );

        assert( initial_stride[contract_dim] ==
                initial_stride[contract_dim+1] * initial_shape[contract_dim+1]
                && "Dimension cannot be contracted" );

        Derived& d = derived();

        for ( int s = 0 ; s < contract_dim ; ++s )
            d.set_stride( s, initial_stride[s] );
        for ( int s = contract_dim ; s < dim ; ++s )
            d.set_stride( s, initial_stride[s+1] );

        for ( int s = 0 ; s < contract_dim ; ++s )
            d.set_shape( s, initial_shape[s] );
        d.set_shape( contract_dim, initial_shape[contract_dim] * initial_shape[contract_dim+1] );
        for ( int s = contract_dim+1 ; s < dim ; ++s )
            d.set_shape( s, initial_shape[s+1] );
    }

public:
    /*
    // Returns a slice, along SliceDim
    template<int SliceDim>
    TensorMap<Const<ScalType>,dim-1> slice(int idx) const
    { return TensorMap_Dim<Const<ScalType>,dim-1,0>(Slice<SliceDim>(idx), *this); }
    template<int SliceDim>
    TensorMap<ScalType,dim-1> slice(int idx)
    { return TensorMap_Dim<ScalType,dim-1,0>(Slice<SliceDim>(idx), *this); }

    // Equivalent to tensor block
    template< int SliceDim>
    TensorMap_Dim<Const<ScalType>,dim,0> middleSlices( int begin, int sz ) const
    {
        static_assert( SliceDim < dim, "Invalid slice dimension" );
        assert( begin >= 0 && begin+sz <= shape_[SliceDim]
                && "Invalid indices" );
        TensorMap_Dim<Const<ScalType>,dim,0> res = *this;
        res.data_ += begin * stride_[SliceDim];
        res.shape_[SliceDim] = sz;
        return res;
    }
    template< int SliceDim>
    TensorMap_Dim<ScalType,dim,0> middleSlices( int begin, int sz )
    {
        static_assert( SliceDim < dim, "Invalid slice dimension" );
        assert( begin >= 0 && begin+sz <= shape_[SliceDim]
                && "Invalid indices" );
        TensorMap_Dim<ScalType,dim,0> res = *this;
        res.data_ += begin * stride_[SliceDim];
        res.shape_[SliceDim] = sz;
        return res;
    }*/


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
    //TODO Check it, and put it on the top
    /*
    template< int contract_dim >
    TensorMap<ScalType,dim-1>
    contract()
    {
        assert( stride_[contract_dim] == stride_[contract_dim+1] * shape_[contract_dim+1]
                && "Cannot be trivially contracted" );
        return TensorMap_Dim<ScalType,dim-1,0>( Contraction<contract_dim>(), *this );
    }
    template< int contract_dim >
    TensorMap<Const<ScalType>,dim-1>
    contract() const
    {
        assert( stride_[contract_dim] == stride_[contract_dim+1] * shape_[contract_dim+1]
                && "Cannot be trivially contracted" );
        return TensorMap_Dim<Const<ScalType>,dim-1,0>( Contraction<contract_dim>(), *this );
    }

    TensorMap<ScalType,dim-1> contractFirst()
    { return contract<0>(); }
    TensorMap<Const<ScalType>,dim-1> contractFirst() const
    { return contract<0>(); }

    TensorMap<ScalType,dim-1> contractLast()
    { return contract<dim-2>(); }
    TensorMap<Const<ScalType>,dim-1> contractLast() const
    { return contract<dim-2>(); }
    */
};

// ----------------------------------------------------------------------------------------
/*
// This is the final class, that implements operator() and provides
// Eigen operations on dimensions 1 and 2
template< typename ScalType, int dim, int current_dim >
class TensorMap_Dim : public TensorMapBase<ScalType,dim>
{
    static_assert( current_dim <= dim, "You probably called operator() too many times" );

public:
    // Perfect forwarding.
    // This enables calling this constructor on TensorMapBase types,
    // which 'using TensoMapBase::TensorMapBase' does not
    template< typename ... Args >
    TensorMap_Dim( Args&& ... args )
     : TensorMapBase<ScalType,dim>( std::forward<Args>(args)... )
    {}

    TensorMap_Dim< ScalType, dim, current_dim+1 >
    operator()( void )
    { return TensorMap_Dim< ScalType, dim, current_dim+1 >( *this ); }
    TensorMap_Dim< Const<ScalType>, dim, current_dim+1 >
    operator()( void ) const
    { return TensorMap_Dim< Const<ScalType>, dim, current_dim+1 >( *this ); }

    TensorMap_Dim< ScalType, dim-1, current_dim >
    operator()( int i )
    {
        assert( i >= 0 && i < this->shape_[current_dim] && "Index out of range" );
        return TensorMap_Dim< ScalType, dim-1, current_dim >( Slice<current_dim>(i), *this );
    }
    TensorMap_Dim< Const<ScalType>, dim-1, current_dim >
    operator()( int i ) const
    {
        assert( i >= 0 && i < this->shape_[current_dim] && "Index out of range" );
        return TensorMap_Dim< Const<ScalType>, dim-1, current_dim >( Slice<current_dim>(i), *this );
    }

    // template< typename ... OtherIndices >
    // typename std::result_of< TensorMap_Dim<ScalType, dim-1, current_dim >(OtherIndices...) >::type
    // operator()( int i, OtherIndices ... indices )
    // { return this->operator()(i)(indices...); }
    // template< typename ... OtherIndices >
    // typename std::result_of< TensorMap_Dim<Const<ScalType>, dim-1, current_dim >(OtherIndices...) >::type
    // operator()( int i, OtherIndices ... indices ) const
    // { return this->operator()(i)(indices...); }
};

template< typename ScalType >
class TensorMap_Dim<ScalType,1,0> :
        public TensorMapBase<ScalType,1>,
        public Eigen::Map< ConstAs<ScalType,Vector<NonConst<ScalType>>>, Eigen::Unaligned, DynInnerStride >
{
    typedef Eigen::Map< ConstAs<ScalType,Vector<NonConst<ScalType>>>, Eigen::Unaligned, DynInnerStride > EigenBase;
    typedef Eigen::Ref< ConstAs<ScalType,Vector<NonConst<ScalType>>>, Eigen::Unaligned, DynInnerStride > EigenRef;

public:
    using EigenBase::operator=;
    using EigenBase::operator+=;
    using EigenBase::operator-=;
    using TensorMapBase<ScalType,1>::size;
    using TensorMapBase<ScalType,1>::data;

    // Perfect forwarding.
    template< typename ... Args >
    TensorMap_Dim( Args&& ... args )
     : TensorMapBase<ScalType,1>( std::forward<Args>(args)... ),
       EigenBase( this->data_, this->shape_[0],
                  DynInnerStride(this->stride_[0]) )
    {}

    // This specific constructor allows implicit conversion from any Vector
    TensorMap_Dim( EigenRef&& vec )
     : TensorMapBase<ScalType,1>( std::move(vec) ),
       EigenBase( this->data_, this->shape_[0],
                  DynInnerStride(this->stride_[0]) )
    {}


    // Use these functions to explicitly get an Eigen::Map
    template< CONDITIONAL_ENABLE_IF_TYPE(ScalType) >
    Eigen::Map< Vector<ScalType>, Eigen::Unaligned, DynInnerStride >
    map( ENABLE_IF( !std::is_const<EnableIfType>::value ) )
    {
        return Eigen::Map< Vector<ScalType>, Eigen::Unaligned, DynInnerStride >(
                this->data_, this->shape_[0], DynInnerStride(this->stride_[0]) );
    }

    Eigen::Map< const Vector< NonConst<ScalType> >, Eigen::Unaligned, DynInnerStride >
    const_map() const
    {
        return Eigen::Map< const Vector< NonConst<ScalType> >, Eigen::Unaligned, DynInnerStride >(
                this->data_, this->shape_[0], DynInnerStride(this->stride_[0]) );
    }

    // Same with contiguity guarantee
    template< CONDITIONAL_ENABLE_IF_TYPE(ScalType) >
    Eigen::Map< Vector<ScalType> >
    ref( ENABLE_IF( !std::is_const<EnableIfType>::value ) )
    {
        assert( this->stride_[0] == 1 && "This map is not contiguous" );
        return Eigen::Map< Vector<ScalType> >(
                this->data_, this->shape_[0] );
    }

    Eigen::Map< const Vector< NonConst<ScalType> > >
    const_ref() const
    {
        assert( this->stride_[0] == 1 && "This map is not contiguous" );
        return Eigen::Map< const Vector< NonConst<ScalType> > >(
                this->data_, this->shape_[0] );
    }


    TensorMap_Dim< ScalType, 1, 1 >
    operator()( void )
    { return TensorMap_Dim< ScalType, 1, 1 >( *this ); }
    TensorMap_Dim< Const<ScalType>, 1, 1 >
    operator()( void ) const
    { return TensorMap_Dim< Const<ScalType>, 1, 1 >( *this ); }

    typename std::conditional< std::is_const<ScalType>::value, NonConst<ScalType>, ScalType& >::type
    operator()( int i )
    {
        assert( i >= 0 && i < this->shape_[0] && "Index out of range" );
        return this->EigenBase::operator()(i);
    }
    ScalType
    operator()( int i ) const
    {
        assert( i >= 0 && i < this->shape_[0] && "Index out of range" );
        return this->EigenBase::operator()(i);
    }
};

template< typename ScalType >
class TensorMap_Dim<ScalType,1,1> :
        public TensorMapBase<ScalType,1>,
        public Eigen::Map< ConstAs<ScalType,Vector<NonConst<ScalType>>>, Eigen::Unaligned, DynInnerStride >
{
    typedef Eigen::Map< ConstAs<ScalType,Vector<NonConst<ScalType>>>, Eigen::Unaligned, DynInnerStride > EigenBase;

public:
    using EigenBase::operator=;
    using EigenBase::operator+=;
    using EigenBase::operator-=;
    using TensorMapBase<ScalType,1>::size;
    using TensorMapBase<ScalType,1>::data;

    // Perfect forwarding.
    template< typename ... Args >
    TensorMap_Dim( Args&& ... args )
     : TensorMapBase<ScalType,1>( std::forward<Args>(args)... ),
       EigenBase( this->data_, this->shape_[0],
                  DynInnerStride(this->stride_[0]) )
    {}

    // Use these functions to explicitly get an Eigen::Map
    template< CONDITIONAL_ENABLE_IF_TYPE(ScalType) >
    Eigen::Map< Vector<ScalType>, Eigen::Unaligned, DynInnerStride >
    map( ENABLE_IF( !std::is_const<EnableIfType>::value ) )
    {
        return Eigen::Map< Vector<ScalType>, Eigen::Unaligned, DynInnerStride >(
                this->data_, this->shape_[0], DynInnerStride(this->stride_[0]) );
    }

    Eigen::Map< const Vector< NonConst<ScalType> >, Eigen::Unaligned, DynInnerStride >
    const_map() const
    {
        return Eigen::Map< const Vector< NonConst<ScalType> >, Eigen::Unaligned, DynInnerStride >(
                this->data_, this->shape_[0], DynInnerStride(this->stride_[0]) );
    }

    // Same with contiguity guarantee
    template< CONDITIONAL_ENABLE_IF_TYPE(ScalType) >
    Eigen::Map< Vector<ScalType> >
    ref( ENABLE_IF( !std::is_const<EnableIfType>::value ) )
    {
        assert( this->stride_[0] == 1 && "This map is not contiguous" );
        return Eigen::Map< Vector<ScalType> >(
                this->data_, this->shape_[0] );
    }

    Eigen::Map< const Vector< NonConst<ScalType> > >
    const_ref() const
    {
        assert( this->stride_[0] == 1 && "This map is not contiguous" );
        return Eigen::Map< const Vector< NonConst<ScalType> > >(
                this->data_, this->shape_[0] );
    }
};

template< typename ScalType >
class TensorMap_Dim<ScalType,2,0> :
        public TensorMapBase<ScalType,2>,
        public Eigen::Map< ConstAs<ScalType,MatrixRM<NonConst<ScalType>>>, Eigen::Unaligned, DynStride >
{
    typedef Eigen::Map< ConstAs<ScalType,MatrixRM<NonConst<ScalType>>>, Eigen::Unaligned, DynStride > EigenBase;
    typedef Eigen::Ref< ConstAs<ScalType,MatrixRM<NonConst<ScalType>>>, Eigen::Unaligned, DynStride > EigenRef;

public:
    using EigenBase::operator=;
    using EigenBase::operator+=;
    using EigenBase::operator-=;
    using TensorMapBase<ScalType,2>::size;
    using TensorMapBase<ScalType,2>::data;

    // Perfect forwarding.
    template< typename ... Args >
    TensorMap_Dim( Args&& ... args )
     : TensorMapBase<ScalType,2>( std::forward<Args>(args)... ),
       EigenBase( this->data_, this->shape_[0], this->shape_[1],
                  DynStride(this->stride_[0], this->stride_[1]) )
    {}

    // This specific constructor allows implicit conversion from any MatrixRM
    TensorMap_Dim( EigenRef&& mat )
     : TensorMapBase<ScalType,2>( std::move(mat), mat.rows(), mat.cols() ),
       EigenBase( this->data_, this->shape_[0], this->shape_[1],
                  DynStride(this->stride_[0], this->stride_[1]) )
    {}

    // Use these functions to explicitly get an Eigen::Map
    template< CONDITIONAL_ENABLE_IF_TYPE(ScalType) >
    Eigen::Map< MatrixRM<ScalType>, Eigen::Unaligned, DynStride >
    map( ENABLE_IF( !std::is_const<EnableIfType>::value ) )
    {
        return Eigen::Map< MatrixRM<ScalType>, Eigen::Unaligned, DynStride >(
                this->data_, this->shape_[0], this->shape_[1],
                DynStride(this->stride_[0],this->stride_[1]) );
    }

    Eigen::Map< const MatrixRM< NonConst<ScalType> >, Eigen::Unaligned, DynStride >
    const_map() const
    {
        return Eigen::Map< const MatrixRM< NonConst<ScalType> >, Eigen::Unaligned, DynStride >(
                this->data_, this->shape_[0], this->shape_[1],
                DynStride(this->stride_[0], this->stride_[1]) );
    }


    // Same with contiguity guarantee
    template< CONDITIONAL_ENABLE_IF_TYPE(ScalType) >
    Eigen::Map< MatrixRM<ScalType>, Eigen::Unaligned, Eigen::OuterStride<> >
    ref( ENABLE_IF( !std::is_const<EnableIfType>::value ) )
    {
        assert( this->stride_[1] == 1 && "This map is not contiguous" );
        return Eigen::Map< MatrixRM<ScalType>, Eigen::Unaligned, Eigen::OuterStride<> >(
                this->data_, this->shape_[0], this->shape_[1],
                Eigen::OuterStride<>(this->stride_[0]) );
    }

    Eigen::Map< const MatrixRM< NonConst<ScalType> >, Eigen::Unaligned, Eigen::OuterStride<> >
    const_ref() const
    {
        assert( this->stride_[1] == 1 && "This map is not contiguous" );
        return Eigen::Map< const MatrixRM< NonConst<ScalType> >, Eigen::Unaligned, Eigen::OuterStride<> >(
                this->data_, this->shape_[0], this->shape_[1],
                Eigen::OuterStride<>(this->stride_[0]) );
    }

    TensorMap_Dim< ScalType, 2, 1 >
    operator()( void )
    { return TensorMap_Dim< ScalType, 2, 1 >( *this ); }
    TensorMap_Dim< Const<ScalType>, 2, 1 >
    operator()( void ) const
    { return TensorMap_Dim< Const<ScalType>, 2, 1 >( *this ); }

    TensorMap_Dim< ScalType, 1, 0 >
    operator()( int i )
    {
        assert( i >= 0 && i < this->shape_[0] && "Index out of range" );
        return TensorMap_Dim< ScalType, 1, 0 >( Slice<0>(i), *this );
    }
    TensorMap_Dim< Const<ScalType>, 1, 0 >
    operator()( int i ) const
    {
        assert( i >= 0 && i < this->shape_[0] && "Index out of range" );
        return TensorMap_Dim< Const<ScalType>, 1, 0 >( Slice<0>(i), *this );
    }

    typename std::conditional< std::is_const<ScalType>::value, NonConst<ScalType>, ScalType& >::type
    operator()( int i, int j )
    { return this->EigenBase::operator()( i, j ); }
    ScalType
    operator()( int i, int j ) const
    { return this->EigenBase::operator()( i, j ); }
};

template< typename ScalType >
class TensorMap_Dim<ScalType,2,1> :
        public TensorMapBase<ScalType,2>,
        public Eigen::Map< ConstAs<ScalType,MatrixRM<NonConst<ScalType>>>, Eigen::Unaligned, DynStride >
{
    typedef Eigen::Map< ConstAs<ScalType,MatrixRM<NonConst<ScalType>>>, Eigen::Unaligned, DynStride > EigenBase;

public:
    using EigenBase::operator=;
    using EigenBase::operator+=;
    using EigenBase::operator-=;
    using TensorMapBase<ScalType,2>::size;
    using TensorMapBase<ScalType,2>::data;

    // Perfect forwarding.
    template< typename ... Args >
    TensorMap_Dim( Args&& ... args )
     : TensorMapBase<ScalType,2>( std::forward<Args>(args)... ),
       EigenBase( this->data_, this->shape_[0], this->shape_[1],
                  DynStride(this->stride_[0], this->stride_[1]) )
    {}

    // Use these functions to explicitly get an Eigen::Map
    template< CONDITIONAL_ENABLE_IF_TYPE(ScalType) >
    Eigen::Map< MatrixRM<ScalType>, Eigen::Unaligned, DynStride >
    map( ENABLE_IF( !std::is_const<EnableIfType>::value ) )
    {
        return Eigen::Map< MatrixRM<ScalType>, Eigen::Unaligned, DynStride >(
                this->data_, this->shape_[0], this->shape_[1],
                DynStride(this->stride_[0],this->stride_[1]) );
    }

    Eigen::Map< const MatrixRM< NonConst<ScalType> >, Eigen::Unaligned, DynStride >
    const_map() const
    {
        return Eigen::Map< const MatrixRM< NonConst<ScalType> >, Eigen::Unaligned, DynStride >(
                this->data_, this->shape_[0], this->shape_[1],
                DynStride(this->stride_[0], this->stride_[1]) );
    }

    // Same with contiguity guarantee
    template< CONDITIONAL_ENABLE_IF_TYPE(ScalType) >
    Eigen::Map< MatrixRM<ScalType>, Eigen::Unaligned, Eigen::OuterStride<> >
    ref( ENABLE_IF( !std::is_const<EnableIfType>::value ) )
    {
        assert( this->stride_[1] == 1 && "This map is not contiguous" );
        return Eigen::Map< MatrixRM<ScalType>, Eigen::Unaligned, Eigen::OuterStride<> >(
                this->data_, this->shape_[0], this->shape_[1],
                Eigen::OuterStride<>(this->stride_[0]) );
    }

    Eigen::Map< const MatrixRM< NonConst<ScalType> >, Eigen::Unaligned, Eigen::OuterStride<> >
    const_ref() const
    {
        assert( this->stride_[1] == 1 && "This map is not contiguous" );
        return Eigen::Map< const MatrixRM< NonConst<ScalType> >, Eigen::Unaligned, Eigen::OuterStride<> >(
                this->data_, this->shape_[0], this->shape_[1],
                Eigen::OuterStride<>(this->stride_[0]) );
    }

    TensorMap_Dim< ScalType, 2, 2 >
    operator()( void )
    { return TensorMap_Dim< ScalType, 2, 2 >( *this ); }
    TensorMap_Dim< Const<ScalType>, 2, 2 >
    operator()( void ) const
    { return TensorMap_Dim< Const<ScalType>, 2, 2 >( *this ); }

    TensorMap_Dim< ScalType, 1, 1 >
    operator()( int i )
    {
        assert( i >= 0 && i < this->shape_[1] && "Index out of range" );
        return TensorMap_Dim< ScalType, 1, 1 >( Slice<1>(i), *this );
    }
    TensorMap_Dim< Const<ScalType>, 1, 1 >
    operator()( int i ) const
    {
        assert( i >= 0 && i < this->shape_[1] && "Index out of range" );
        return TensorMap_Dim< Const<ScalType>, 1, 1 >( Slice<1>(i), *this );
    }
};

template< typename ScalType >
class TensorMap_Dim<ScalType,2,2> :
        public TensorMapBase<ScalType,2>,
        public Eigen::Map< ConstAs<ScalType,MatrixRM<NonConst<ScalType>>>, Eigen::Unaligned, DynStride >
{
    typedef Eigen::Map< ConstAs<ScalType,MatrixRM<NonConst<ScalType>>>, Eigen::Unaligned, DynStride > EigenBase;

public:
    using EigenBase::operator=;
    using EigenBase::operator+=;
    using EigenBase::operator-=;
    using TensorMapBase<ScalType,2>::size;
    using TensorMapBase<ScalType,2>::data;

    // Perfect forwarding.
    template< typename ... Args >
    TensorMap_Dim( Args&& ... args )
     : TensorMapBase<ScalType,2>( std::forward<Args>(args)... ),
       EigenBase( this->data_, this->shape_[0], this->shape_[1],
                  DynStride(this->stride_[0], this->stride_[1]) )
    {}

    // Use these functions to explicitly get an Eigen::Map
    template< CONDITIONAL_ENABLE_IF_TYPE(ScalType) >
    Eigen::Map< MatrixRM<ScalType>, Eigen::Unaligned, DynStride >
    map( ENABLE_IF( !std::is_const<EnableIfType>::value ) )
    {
        return Eigen::Map< MatrixRM<ScalType>, Eigen::Unaligned, DynStride >(
                this->data_, this->shape_[0], this->shape_[1],
                DynStride(this->stride_[0],this->stride_[1]) );
    }

    Eigen::Map< const MatrixRM< NonConst<ScalType> >, Eigen::Unaligned, DynStride >
    const_map() const
    {
        return Eigen::Map< const MatrixRM< NonConst<ScalType> >, Eigen::Unaligned, DynStride >(
                this->data_, this->shape_[0], this->shape_[1],
                DynStride(this->stride_[0], this->stride_[1]) );
    }

    // Same with contiguity guarantee
    template< CONDITIONAL_ENABLE_IF_TYPE(ScalType) >
    Eigen::Map< MatrixRM<ScalType>, Eigen::Unaligned, Eigen::OuterStride<> >
    ref( ENABLE_IF( !std::is_const<EnableIfType>::value ) )
    {
        assert( this->stride_[1] == 1 && "This map is not contiguous" );
        return Eigen::Map< MatrixRM<ScalType>, Eigen::Unaligned, Eigen::OuterStride<> >(
                this->data_, this->shape_[0], this->shape_[1],
                Eigen::OuterStride<>(this->stride_[0]) );
    }

    Eigen::Map< const MatrixRM< NonConst<ScalType> >, Eigen::Unaligned, Eigen::OuterStride<> >
    const_ref() const
    {
        assert( this->stride_[1] == 1 && "This map is not contiguous" );
        return Eigen::Map< const MatrixRM< NonConst<ScalType> >, Eigen::Unaligned, Eigen::OuterStride<> >(
                this->data_, this->shape_[0], this->shape_[1],
                Eigen::OuterStride<>(this->stride_[0]) );
    }
};
*/

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
    { return { m_shape }; }

    inline StrideMap<dim,int> stride() const
    { return { m_stride }; }

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
    using AttrBase::shape;
    using AttrBase::stride;

protected:
    using AttrBase::set_data;
    using AttrBase::set_shape;
    using AttrBase::set_stride;
};

// ----- TensorDim<1> -----

//TODO

// ----- TensorOperator -----

template< typename Derived, int _dim = Traits<Derived>::dim, int _current_dim = 0 >
class TensorOperator :
    public TensorOperator< Derived, _dim, _current_dim+1 >
{
    static_assert( _current_dim >= 0 && _current_dim <= _dim,
           "Wrong usage of TensorOperator" );

public:
    typedef TensorOperator< Derived, _dim, _current_dim+1 > Base;
    friend Base;
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

protected:
    using Base::set_data;
    using Base::set_shape;
    using Base::set_stride;
};

// ----- TensorMapBase -----

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
        std::cout << sizeof...(Dimensions) << " == " << dim << std::endl;
        std::cout << "Inner stride = " << inner_stride.inner << std::endl;
        derived().set_data( nullptr );
        derived().set_stride( dim-1, inner_stride.inner );
        Base::template init_sns_from_shape<0>(dimensions...);
    }

    // ... or let it to 1 in the default case
    template< typename ... Dimensions, typename = EnableIf<sizeof...(Dimensions)==dim> >
    inline TensorMapBase( ScalType* data, Dimensions ... dimensions )
     : TensorMapBase( data, InnerStride(1), dimensions... )
    {
        std::cout << "Default inner stride" << std::endl;
    }

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
        ConstCompatible< typename Traits<OtherDerived>::ScalType, ScalType >()
        && Traits<Derived>::dim == dim > >
    TensorMapBase( ConstAs< ScalType,TensorBase< OtherDerived > >& other )
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

    // This can take inner and outer strides into account (like blocks)
    template<typename ... Dimensions,
        typename = EnableIf< sizeof...(Dimensions) == dim > >
    TensorMapBase( MatrixRef&& mat, Dimensions ... dimensions )
    {
        typedef typename MatrixRef::Index Index;

        derived().set_data( mat.data() );
        Base::template init_sns_reshape_tensor<dim-1,1,2,Index,Index>(
                ShapeOwn<2,Index>( mat.rows(), mat.cols() ),
                StrideOwn<2,Index>( mat.outerStride(), mat.innerStride() ),
                dimensions... );
    }
};

// ----- TensorMap -----

template< typename _ScalType, int _dim >
class TensorMap :
    public TensorMapBase< TensorMap<_ScalType,_dim> >
{
public:
    typedef TensorMapBase<TensorMap> Base;
    friend Base;

    typedef TensorBase<TensorMap> OtherBase;
    friend OtherBase;

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

protected:
    using Base::set_data;
    using Base::set_shape;
    using Base::set_stride;
};

// ----- Method implementations -----

template< typename Derived >
template< int new_dim, typename ... Dimensions, typename >
TensorMap< typename TensorBase<Derived>::ScalType, new_dim >
TensorBase<Derived>::reshape( Dimensions ... dimensions )
{
    const Derived& d = derived();
    TensorMap< ScalType, new_dim > new_tensor( EmptyConstructor() );
    new_tensor.data() = d.data();
    new_tensor.template init_sns_reshape_tensor<new_dim-1,dim-1,dim,int,int>(
            d.shape(), d.stride(),
            1, 1,
            dimensions... );
    return new_tensor;
}

template< typename Derived >
template< int new_dim, typename ... Dimensions, typename >
TensorMap< Const<typename TensorBase<Derived>::ScalType>, new_dim >
TensorBase<Derived>::reshape( Dimensions ... dimensions ) const
{
    const Derived& d = derived();
    TensorMap< Const<ScalType>, new_dim > new_tensor( EmptyConstructor() );
    new_tensor.data() = d.data();
    new_tensor.template init_sns_reshape_tensor<new_dim-1,dim-1,dim,int,int>(
            d.shape(), d.stride(),
            1, 1,
            dimensions... );
    return new_tensor;
}

} // namespace TensorMapTools

template< typename ScalType, int dim >
using TensorMap = TensorMapTools::TensorMap<ScalType,dim>;
