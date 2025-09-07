/**
RealVectorFramework - A Generic Library for Vector Operations and Algorithms

Copyright (c) National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

Questions? Contact Greg von Winckel (gvonwin@sandia.gov)
*/

#pragma once

#include <concepts>
#include <type_traits>
#include <compare>

#include "core/cmath/abs.hpp"
#include "core/cmath/fmax.hpp"
#include "core/cmath/fmin.hpp"

namespace rvf {

template<typename S>
concept real_scalar_c = std::totally_ordered<S>         // C++20 <=> ordering
                     && std::constructible_from<S, int> // can be built from an integer
                     && !std::integral<S>               // but must *not* itself be an integer type
                     && requires(S a, S b) {            // these operations must compile…
  
    // Supports binary arithmetic operations
    { a + b } -> std::same_as<S>;
    { a - b } -> std::same_as<S>;
    { a * b } -> std::same_as<S>;
    { a / b } -> std::same_as<S>;

    // 1/2 runs and isn’t zero
    { S{ 1 } / S{ 2 } } -> std::same_as<S>;
    requires (S{ 1 } / S{ 2 }) != S{ 0 };

    {abs(a)    } -> std::same_as<S>;
    {fmax(a, b)} -> std::same_as<S>;
    {fmin(a, b)} -> std::same_as<S>;
};

template<typename S>
concept small_real_scalar_c = (sizeof(S) <= 3 * sizeof(void*)) && real_scalar_c<S>;

template<typename S>
concept large_real_scalar_c = (sizeof(S) > 3 * sizeof(void*)) && real_scalar_c<S>;

} // namespace rvf

