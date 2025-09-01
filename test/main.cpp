/**
RealVectorFramework - A Generic Library for Vector Operations and Algorithms

Copyright (c) National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

Questions? Contact Greg von Winckel (gvonwin@sandia.gov)
*/

#include <gtest/gtest.h>
#include <tincup/tincup.hpp>
#include <vector>

TEST(BasicInclusionTest, TInCuPIncluded) {
    // Just test that TInCuP headers can be included
    EXPECT_TRUE(true);
}

// Simple test for header inclusion
TEST(BasicInclusionTest, CanIncludeHeaders) {
    // Just verify the build system works
    EXPECT_TRUE(true);
}
