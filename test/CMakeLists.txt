# RealVectorFramework - A Generic Library for Vector Operations and Algorithms
#
# Copyright (c) National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# Questions? Contact Greg von Winckel (gvonwin@sandia.gov)

add_executable(tests
  main.cpp
  clone_by_value_test.cpp
  clone_unique_ptr_test.cpp
)

target_link_libraries(tests PRIVATE RealVectorFramework::RealVectorFramework GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(tests)
