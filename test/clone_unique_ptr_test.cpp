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

#include "operations/core/clone.hpp"

// Provide a specialization of clone for std::vector<double> in this TU that
// returns a std::unique_ptr<std::vector<double>>. This is more specialized than
// the generic std::ranges overload and will be preferred by overload resolution
// within this translation unit only.
namespace rvf {

inline auto tag_invoke(clone_ftor, const std::vector<double>& x)
    -> std::unique_ptr<std::vector<double>> {
  return std::make_unique<std::vector<double>>(x);
}

} // namespace rvf

namespace {

TEST(CloneTests, CloneReturnsUniquePtr) {
  std::vector<double> x{5.0, 6.0, 7.0};

  // Uses the specialized tag_invoke above in this TU
  auto cl = rvf::clone(x); auto& y = rvf::deref_if_needed(cl);

  ASSERT_EQ(y.size(), x.size());
  EXPECT_EQ(y, x);

  y[1] = -1.0;
  EXPECT_NE(y[1], x[1]);  // distinct storage (deep clone)
}

} // namespace
