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

template< typename ScalType >
constexpr bool IsConst = std::is_const<AsThis>::value;

template< typename AsThis, typename Type >
using ConstAs = typename std::conditional< IsConst<AsThis>, Const<Type>, NonConst<Type> >::type;


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

// ----- Stride and shape tool structures -----

// These little structs are used to keep easy-to-read constructors
// of TensorMapBase

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

    inline Integer operator[]( int i ) const
    {
        assert( i < dim );
        return shape[i];
    }
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

    inline Integer operator[]( int i ) const
    {
        assert( i < dim );
        return stride[i];
    }
};

// Eigen dynamic strides
typedef Eigen::InnerStride<Eigen::Dynamic> DynInnerStride;
typedef Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> DynStride;

// ----- Forward declarations -----

template< typename ScalType, int dim, int current_dim >
class TensorDim;

template< typename ScalType, int dim >
class TensorOwn;

template< typename Derived >
class TensorBase;

template< typename Derived >
class TensorMapBase;

template< typename Derived >
class TensorOwnBase;

template< typename Derived >
class TensorBase_MatrixLike;

template< typename Derived >
class TensorBase_VectorLike;

// ----- Traits -----

template< typename Type >
struct Traits;

template< typename _ScalType, int _dim, int _current_dim >
struct Traits< TensorDim<_ScalType,_dim,_current_dim> >
{
    typedef _ScalType ScalType;
    static constexpr int dim = _dim;
    static constexpr int current_dim = _current_dim;
    static constexpr bool owns = false;
};

template< typename _ScalType, int _dim >
struct Traits< TensorOwn<_ScalType,_dim> >
{
    typedef _ScalType ScalType;
    static constexpr int dim = _dim;
    static constexpr int current_dim = 0;
    static constexpr bool owns = true;
};

// ----- TensorBase -----

template< typename Derived >
class TensorBase
{
public:
    typedef typename Traits<Derived>::ScalType CScalType;
    typedef NonConst<CScalType> ScalType;
    constexpr static int dim = Traits<Derived>::dim;
    constexpr static int current_dim = Traits<Derived>::current_dim;
    constexpr static bool owns = Traits<Derived>::owns;

    typedef Eigen::Ref<ConstAs<ScalType, MatrixRM<NonConst<ScalType>>>,
            Eigen::Unaligned, DynStride>
            EigenRef;
    static_assert(dim > 0, "You are a weird person");

public:
    Derived& derived() { return *static_cast<Derived*>(this); }
    const Derived& derived() const { return *static_cast<const Derived*>(this); }

    TensorBase()
     : data_(NULL)
    {
        for ( int& s : stride_ )
            s = 0;
        for ( int& s : shape_ )
            s = 0;
    }


    Stride<dim,int> stride() const
    { return { stride_ }; }

    int stride( int i ) const
    { return stride_[i]; }

    Shape<dim,int> shape() const
    { return { shape_ }; }

    int shape( int i ) const
    { return shape_[i]; }

    template< typename = EnableIf< !IsConst<ScalType> > >
    ScalType* data()
    { return data_; }

    Const<ScalType>* data() const
    { return data_; }

protected:
    TensorBase( EmptyConstructor ) {}

    ScalType *data_;
    int shape_[dim];
    int stride_[dim];

    // Used to initialize strides and shapes in constructors
    // (inner stride must be specified before calling this)
    template< int d, typename ... OtherDimensions >
    inline void init_sns_from_shape( int s, OtherDimensions ... other_dimensions )
    {
        shape_[d] = s;
        init_from_shape<d+1>( other_dimensions... );
        stride_[d] = shape_[d + 1] * stride_[d + 1];
    }
    template< int s >
    inline void init_sns_from_shape( int d )
    {
        shape_[s] = d;
    }

    // Give the total shape of the tensor
    template< typename ... OtherDimensions >
    inline static int total_size( int s, OtherDimensions ... other_dimensions )
    {
        return s*total_size( other_dimensions... );
    }
    inline static int total_size()
    { return 1; }

    // A bit tricky : from another Tensor (including strange strides)
    // It can be a pybind array, an Eigen matrix, another Tensor, or whatever
    // It has to be recursive to access statically the members of Dimensions...
    template< int s, int other_s,
        int other_dim, typename ShapeType, typename StrideType,
        typename ... Dimensions, typename = EnableIf< (s>=0&&other_s>=0) > >
    inline void init_sns_reshape_tensor( const Shape<other_dim,ShapeType>& other_shape,
                              const Stride<other_dim,StrideType>& other_stride,
                              int current_total_size,
                              int other_current_total_size,
                              Dimensions ... dimensions )
    {
        int new_total_size = current_total_size*nth_of_pack<s>(dimensions...);
        int other_new_total_size = other_current_total_size*other_shape[other_s];

        shape_[s] = nth_of_pack<s>(dimensions...);

        if ( new_total_size == other_new_total_size )
        {
            stride_[s] = other_stride[other_s];

            if ( s > 0 && other_s > 0)
            {
                init_sns_reshape_tensor<s-1,other_s-1,other_dim,ShapeType,StrideType>(
                        other_shape, other_stride,
                        new_total_size,
                        other_new_total_size,
                        dimensions... );
            }
            else if ( s > 0 )
            {
                init_sns_reshape_tensor<s-1,other_s,other_dim,ShapeType,StrideType>(
                        other_shape, other_stride,
                        new_total_size,
                        other_current_total_size,
                        dimensions... );
            }
            else if ( other_s > 0 )
            {
                init_sns_reshape_tensor<s,other_s-1,other_dim,ShapeType,StrideType>(
                        other_shape, other_stride,
                        current_total_size,
                        other_new_total_size,
                        dimensions... );
            }
        }
        else if ( new_total_size > other_new_total_size )
        {
            // Split other dimension. The strides must be compatible
            assert( other_dim > 0
                    && other_shape[other_s]*other_stride[other_s] == other_stride[other_s-1]
                    && "Incompatible stride/shape" );

            init_sns_reshape_tensor<s,other_s-1,other_dim,ShapeType,StrideType>(
                    other_shape, other_stride,
                    current_total_size,
                    other_new_total_size,
                    dimensions... );
        }
        else // new_total_size < other_new_total_size
        {
            // Split this dimension. The strides have no constraint
            stride_[s] = (s == dim-1)
                         ? other_stride.innerStride()
                         : stride_[s+1] * shape_[s+1];

            init_reshape_tensor<s-1,other_s,other_dim,ShapeType,StrideType>(
                    other_shape, other_stride,
                    new_total_size,
                    other_current_total_size,
                    dimensions... );
        }
    }
    
    template< int s, int other_s, int other_dim,
        typename ShapeType, typename StrideType,
        typename ... Dimensions, typename = EnableIf< (s<0||other_s<0) > >
    inline void init_sns_reshape_tensor( const Shape<other_dim,ShapeType>& other_shape,
                              const Stride<other_dim,StrideType>& other_stride,
                              int current_total_size,
                              int other_current_total_size,
                              Dimensions ... dimensions ) const
    {}

public:
    // Reshape the tensor
    template< int new_dim, Dimensions ... dimensions,
        typename = EnableIf< sizeof...(Dimensions)==new_dim && !IsConst<ScalType> > >
    TensorMap< ScalType, new_dim >
    reshape( Dimensions ... dimensions )
    {
        TensorMap< ScalType, new_dim > t( EmptyConstructor() );
        t.data_ = data_;
        t.init_sns_reshape_tensor<new_dim-1,dim-1,dim,int,int>(
                shape(), stride(),
                1, 1,
                dimensions... );
        return t;
    }

    template< int new_dim, Dimensions ... dimensions,
        typename = EnableIf< sizeof...(Dimensions)==new_dim > >
    TensorMap< Const<ScalType>, new_dim >
    reshape( Dimensions ... dimensions ) const
    {
        TensorMap< Const<ScalType>, new_dim > t( EmptyConstructor() );
        t.data_ = data_;
        t.init_sns_reshape_tensor<new_dim-1,dim-1,dim,int,int>(
                shape(), stride(),
                1, 1,
                dimensions... );
        return t;
    }

    // You can omit the new dimension
    template< Dimensions ... dimensions,
        typename = EnableIf< !IsConst<ScalType> > >
    inline TensorMap< ScalType, sizeof...(Dimensions) >
    reshape( Dimensions ... dimensions )
    { return reshape< sizeof...(Dimensions), Dimensions... >( dimensions... ); }

    template< Dimensions ... dimensions >
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

    template< int contract_dim, typename ShapeType, typename StrideType,
        typename = EnableIf< (contract_dim<dim) > >
    void init_sns_from_contraction( const Shape<dim+1>& shape, const Stride<dim+1>& stride )
    {
        assert( stride[contract_dim] == stride[contract_dim+1] * shape[contract_dim+1]
                && "Dimension cannot be contracted" );

        std::copy(stride.stride, stride.stride + contract_dim, stride_);
        std::copy(stride.stride + contract_dim + 1, stride.stride + dim + 1, stride_ + contract_dim);

        std::copy(shape.shape, shape.shape + contract_dim, shape_);
        shape_[contract_dim] = shape.shape[contract_dim] * shape.shape[contract_dim+1];
        std::copy(shape.shape + contract_dim + 2, shape.shape + dim + 1, shape_ + contract_dim+1);
    }

public:
    // Returns a slice, along SliceDim
    template<int SliceDim>
    TensorMap_Dim<Const<ScalType>,dim-1,0> slice(int idx) const
    { return TensorMap_Dim<Const<ScalType>,dim-1,0>(Slice<SliceDim>(idx), *this); }
    template<int SliceDim>
    TensorMap_Dim<ScalType,dim-1,0> slice(int idx)
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
    }


    // --- Utility methods ---

    bool ravelable() const
    {
        for (int i = 0; i < dim - 1; ++i)
        {
            if (stride_[i] != shape_[i + 1] * stride_[i + 1])
                return false;
        }
        return true;
    }

    int size() const
    {
        int s = 1;
        for (int i = 0; i < dim; ++i)
            s *= shape_[i];
        return s;
    }

    // Returns the pointer to the data after the contained buffer
    // Note: this is not the data()+size() pointer, it takes the outer
    // stride into account
    ScalType* next_data()
    {
        int outer = outerDim();
        return data_ + stride_[outer]*shape_[outer];
    }

    Const<ScalType>* next_data() const
    {
        int outer = outerDim();
        return data_ + stride_[outer]*shape_[outer];
    }

    // Returns the dimension whose stride is the biggest
    int outerDim() const
    {
        int biggestStride = 0;
        int res = 0;
        for ( int i = 0 ; i < dim ; ++i )
        {
            if ( stride_[i] > biggestStride )
            {
                biggestStride = stride_[i];
                res = i;
            }
        }

        return res;
    }

    // Returns a pointer to the beginning of the mapped data
    ScalType* data()
    { return data_; }

    Const<ScalType>* data() const
    { return data_; }


    bool empty() const
    { return data_ == NULL; }


    // Returns a Map of the corresponding col vector
    Eigen::Map< ConstAs<ScalType,Vector<NonConst<ScalType>>>, Eigen::Unaligned, DynInnerStride >
    dyn_ravel()
    {
        assert( ravelable() && "Cannot be raveled" );
        return Eigen::Map<ConstAs<ScalType,Vector<NonConst<ScalType>>>,
                Eigen::Unaligned, DynInnerStride>(data_, size(), DynInnerStride(stride_[dim - 1]));
    }
    Eigen::Map< const Vector<NonConst<ScalType>>, Eigen::Unaligned, DynInnerStride >
    dyn_ravel() const
    {
        assert(ravelable() && "Cannot be raveled");
        return Eigen::Map<const Vector<NonConst<ScalType>>,
                Eigen::Unaligned, DynInnerStride>(data_, size(), DynInnerStride(stride_[dim - 1]));
    }

    // Returns a Map of the corresponding col vector, with an inner stride of 1
    Eigen::Map< ConstAs<ScalType,Vector<NonConst<ScalType>>>, Eigen::Unaligned >
    ravel()
    {
        assert( stride_[dim-1] == 1 && ravelable() && "Cannot be raveled" );
        return Eigen::Map<ConstAs<ScalType,Vector<NonConst<ScalType>>>,
                Eigen::Unaligned>(data_, size());
    }
    Eigen::Map< const Vector<NonConst<ScalType>>, Eigen::Unaligned >
    ravel() const
    {
        assert( stride_[dim-1] == 1 && ravelable() && "Cannot be raveled" );
        return Eigen::Map<const Vector<NonConst<ScalType>>,
                Eigen::Unaligned>(data_, size());
    }

    // Returns the shape and stride for a certain dimension
    int shape(int i) const
    {
        assert(i < dim && "Shape index out of dim");
        return shape_[i];
    }
    int stride(int i) const
    {
        assert(i < dim && "Stride index out of dim");
        return stride_[i];
    }

    // Contracts ContractDim with ContractDim+1 dimensions
    template< int ContractDim >
    TensorMap<ScalType,dim-1,0>
    contract()
    {
        assert( stride_[ContractDim] == stride_[ContractDim+1] * shape_[ContractDim+1]
                && "Cannot be trivially contracted" );
        return TensorMap_Dim<ScalType,dim-1,0>( Contraction<ContractDim>(), *this );
    }
    template< int ContractDim >
    TensorMap<Const<ScalType>,dim-1,0>
    contract() const
    {
        assert( stride_[ContractDim] == stride_[ContractDim+1] * shape_[ContractDim+1]
                && "Cannot be trivially contracted" );
        return TensorMap_Dim<Const<ScalType>,dim-1,0>( Contraction<ContractDim>(), *this );
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

// ----- TensorMapBase -----
//
template< typename Derived >
class TensorMapBase : public TensorBase<Derived>
{
    typedef TensorBase<Derived> Base;

    using Base::Base;

public:
    template< typename ... Dimensions, typename = EnableIf<sizeof...(Dimensions)==dim> >
    TensorMapBase( ScalType *data, const DynInnerStride &inner_stride, Dimensions ... dimensions )
     : data_(data)
    {
        stride_[dim-1] = inner_stride.inner();
        init_from_shape<0>(dimensions...);
    }

    template< typename ... Dimensions, typename = EnableIf<sizeof...(Dimensions) == dim> >
    inline TensorMapBase(ScalType *data, Dimensions ... dimensions)
     : TensorMapBase( data, DynInnerStride(1), dimensions... )
    {}

    template< typename ShapeType, typename StrideType >
    TensorMapBase( ScalType* data,
                   const Shape<dim,ShapeType>& shape,
                   const Stride<dim,StrideShape>& stride )
    {
        data_ = data;
        for ( int i = 0 ; i < dim ; ++i )
        {
            shape_[i] = shape[i];
            stride_[i] = stride[i];
        }
    }

    // This is used to allocate the tensor-map through a BufMap
    template< typename ... Dimensions,
        typename = EnableIf< sizeof...(Dimensions) == dim > >
    TensorMapBase( DxyzUtils::BufMap< NonConst<ScalType> >& bufmap, Dimensions ... dimensions )
     : TensorMapBase( bufmap.ptr(total_size(dimensions...)), dimensions... )
    {}

    // This can take inner and outer strides into account (like blocks)
    template<typename ... Dimensions,
        typename = EnableIf< sizeof...(Dimensions) == dim > >
    TensorMapBase(EigenRef&& mat, Dimensions ... dimensions)
    {
        typedef typename EigenRef::Index Index;

        data_ = mat.data();
        Index shape[] = { mat.rows(), mat.cols() };
        Index stride[] = { mat.outerStride(), mat.innerStride() };
        init_reshape_tensor<dim-1,1,2,Index,Index>(
                Shape<2,Index>( shape ),
                Stride<2,Index>( stride ),
                dimensions... );
    }
};

// ----------------------------------------------------------------------------------------

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

} // namespace TensorMapTools
