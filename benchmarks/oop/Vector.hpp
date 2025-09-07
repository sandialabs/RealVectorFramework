#pragma once

#include <memory>

// OOP Example base class for comparison with concepts + CPO + ADL approach

template<class Real>
class Vector { 
public:
  virtual ~Vector() = default;
  // Required Overrides
  virtual void add_in_place( const Vector& ) = 0;
  virtual std::unique_ptr<Vector> clone() const = 0;
  virtual Real inner_product( const Vector& ) const = 0;
  virtual void scale_in_place( Real ) = 0;
  // Optional overrides
  virtual void assign( const Vector& x ) {
    scale_in_place(0);
    add_in_place(x);
  }
  virtual void axpy_in_place( Real alpha, const Vector& x ) {
    auto tmp = clone();
    tmp->assign(x);
    tmp->scale_in_place(alpha);
    add_in_place(*tmp);
  }
};

template<class Real>
class VectorMap {
public:
  virtual void operator() (Vector<Real>&, const Vector<Real>&) const = 0;
};


                         
