/**
RealVectorFramework - A Generic Library for Vector Operations and Algorithms

Copyright (c) National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

Questions? Contact Greg von Winckel (gvonwin@sandia.gov)
*/

#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "operations/clone.hpp"
#include "operations/deref_if_needed.hpp"

// Provide a specialization that returns weak_ptr to demonstrate that this
// is intentionally not supported by RVF's deref_if_needed pattern.
namespace rvf {

inline auto tag_invoke(clone_ftor, const std::vector<double>& x)
    -> std::weak_ptr<std::vector<double>> {
  return std::weak_ptr<std::vector<double>>{std::make_shared<std::vector<double>>(x)};
}

} // namespace rvf

namespace {

// This test only asserts the type-level expectation that weak_ptr is not
// treated as a nullable pointer by RVF; attempting to deref_if_needed a
// weak_ptr would not compile by design.
TEST(CloneTests, WeakPtrIsNotSupportedForDereference) {
  using Vec = std::vector<double>;
  using weak_vec = std::weak_ptr<Vec>;

  static_assert(!rvf::nullable_pointer_c<weak_vec>,
                "weak_ptr must not satisfy nullable_pointer_c");

  // Show that clone_return_t resolves to weak_ptr in this TU
  using clone_t = rvf::clone_return_t<Vec>;
  static_assert(std::is_same_v<clone_t, weak_vec>,
                "tag_invoke in this TU returns weak_ptr for clone");

  // We intentionally do NOT call deref_if_needed on the weak_ptr here,
  // as that usage is unsupported by design.
  SUCCEED();
}

} // namespace

