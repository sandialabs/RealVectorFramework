/**
RealVectorFramework - A Generic Library for Vector Operations and Algorithms

Copyright (c) National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

Questions? Contact Greg von Winckel (gvonwin@sandia.gov)
*/

#include <gtest/gtest.h>
#include <vector>

#include "operations.hpp"

namespace {

TEST(CloneTests, CloneReturnsByValue) {
  std::vector<double> x{1.0, 2.0, 3.0};

  // Default std::ranges support returns by value for std::vector
  auto cl = rvf::clone(x); auto& y = rvf::deref_if_needed(cl);

  ASSERT_EQ(y.size(), x.size());
  EXPECT_EQ(y, x);

  // Modifying y does not affect x
  y[0] = 42.0;
  EXPECT_NE(y[0], x[0]);
}

} // namespace
