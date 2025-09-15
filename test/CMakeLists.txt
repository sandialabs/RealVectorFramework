# RealVectorFramework - A Generic Library for Vector Operations and Algorithms
#
# Copyright (c) National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# Questions? Contact Greg von Winckel (gvonwin@sandia.gov)

# Create separate executables for tests that have ODR conflicts
add_executable(basic_tests
  main.cpp
  clone_by_value_test.cpp
  clone_weak_ptr_negative.cpp
)

add_executable(unique_ptr_tests
  main.cpp
  clone_unique_ptr_test.cpp
)

add_executable(shared_ptr_tests
  main.cpp
  clone_shared_ptr_test.cpp
)

target_link_libraries(basic_tests PRIVATE RealVectorFramework::RealVectorFramework GTest::gtest_main)
target_link_libraries(unique_ptr_tests PRIVATE RealVectorFramework::RealVectorFramework GTest::gtest_main)
target_link_libraries(shared_ptr_tests PRIVATE RealVectorFramework::RealVectorFramework GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(basic_tests)
gtest_discover_tests(unique_ptr_tests)
gtest_discover_tests(shared_ptr_tests)
