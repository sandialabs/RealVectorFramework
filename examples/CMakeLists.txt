# RealVectorFramework - A Generic Library for Vector Operations and Algorithms
#
# Copyright (c) National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# Questions? Contact Greg von Winckel (gvonwin@sandia.gov)

add_executable(example main.cpp)

target_link_libraries(example PRIVATE RealVectorFramework::RealVectorFramework)

add_executable(example_traits traits_vs_ranges.cpp)

target_link_libraries(example_traits PRIVATE RealVectorFramework::RealVectorFramework)

add_executable(hs1_gradient_descent hs1_gradient_descent.cpp)

target_link_libraries(hs1_gradient_descent PRIVATE RealVectorFramework::RealVectorFramework)

add_executable(hs1_advanced hs1_advanced.cpp)

target_link_libraries(hs1_advanced PRIVATE RealVectorFramework::RealVectorFramework)
