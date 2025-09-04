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
// returns a std::shared_ptr<std::vector<double>>.
namespace rvf {

inline auto tag_invoke(clone_ftor, const std::vector<double>& x)
    -> std::shared_ptr<std::vector<double>> {
  return std::make_shared<std::vector<double>>(x);
}

} // namespace rvf

namespace {

TEST(CloneTests, CloneReturnsSharedPtr) {
  std::vector<double> x{1.0, 2.0, 3.0};

  auto cl = rvf::clone(x); auto& y = rvf::deref_if_needed(cl);

  ASSERT_EQ(y.size(), x.size());
  EXPECT_EQ(y, x);

  y[0] = 9.0;
  EXPECT_NE(y[0], x[0]);  // deep copy
}

} // namespace

