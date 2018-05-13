#include <iostream>

struct Dimension
{
    Dimension() = delete;
    Dimension( int i ) : i(i) {}
    int i;
};

struct Nothing {};

void f( void )
{
    std::cout << "Done." << std::endl;
}

template< typename ... OtherDims >
void f( Nothing, OtherDims ... dims )
{
    std::cout << "No dim." << std::endl;
    f( dims... );
}

template< typename ... OtherDims >
void f( Dimension d, OtherDims ... dims )
{
    std::cout << "Got dim " << d.i << std::endl;
    f( dims... );
}

int main()
{
    f( 0 );
    f( {} );
    //f( 0, 1, {}, 2 );

    return 0;
}
