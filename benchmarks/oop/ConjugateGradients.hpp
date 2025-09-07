#pragma once

#include "Vector.hpp"
template<class Real>
void conjugate_gradients(
    Vector<Real>&       x,
    const VectorMap<Real>& applyA,
    const Vector<Real>& b,
    Real                tol,
    int                 max_iter = 1000) {

  auto r_ptr  = x.clone(); Vector<Real>& r  = *r_ptr;
  auto Ap_ptr = x.clone(); Vector<Real>& Ap = *Ap_ptr;
  auto p_ptr  = x.clone(); Vector<Real>& p  = *p_ptr;

  applyA(Ap, x);
  r.assign(b);
  r.axpy_in_place(-1, Ap);

  Real rr = r.inner_product(r);
  Real norm0 = std::sqrt(rr);

  for(int iter = 0; iter < max_iter; ++iter) {
    applyA(Ap, p);
    Real pAp = p.inner_product(Ap);
    Real alpha = rr / pAp;

    x.axpy_in_place(alpha, p);
    r.axpy_in_place(-alpha, Ap);

    Real rr_new = r.inner_product(r);
    if(std::sqrt(rr_new) < tol*norm0) break;

    Real beta = rr_new / rr;
    p.scale_in_place(beta);
    p.add_in_place(r);

    rr = rr_new;
  }
  // x now holds the solution
}
