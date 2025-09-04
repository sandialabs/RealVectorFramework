/**
RealVectorFramework - A Generic Library for Vector Operations and Algorithms

Copyright (c) National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

Questions? Contact Greg von Winckel (gvonwin@sandia.gov)
*/

#pragma once

namespace rvf {

/**
 * @brief Concept for operators that map vectors to vectors of the same type
 * 
 * A self_map is any callable that takes two vector arguments and performs
 * an operation of the form: A(y, x) computes y = A*x and returns void.
 * 
 * This concept is intentionally generic and does not enforce mathematical
 * properties like linearity, symmetry, or positive definiteness, as these
 * cannot be verified at compile time.
 * 
 * Usage notes for algorithms:
 * - When used as a linear operator: Algorithms expect A to satisfy linearity:
 *   A(αx + βy) = αA(x) + βA(y) for all scalars α, β and vectors x, y
 * - When used as a matrix: Algorithms may expect additional properties like
 *   symmetry or positive definiteness depending on the mathematical context
 * - When used as a preconditioner: Algorithms expect A to approximate the
 *   inverse of some other operator
 * 
 * @tparam A The operator type
 * @tparam V The vector type
 */
template<typename A, typename V>
concept self_map_c = requires(const A& op, V& y, const V& x) {
  { op(y, x) } -> std::same_as<void>;
};

} // namespace rvf