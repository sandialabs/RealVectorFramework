# RealVectorFramework - A Generic Library for Vector Operations and Algorithms
#
# Copyright (c) National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# Questions? Contact Greg von Winckel (gvonwin@sandia.gov)

add_executable(arena_benchmarks arena_benchmarks.cpp)

target_link_libraries(arena_benchmarks 
  PRIVATE 
    RealVectorFramework::RealVectorFramework
    benchmark::benchmark
    benchmark::benchmark_main
)

# Enable optimizations for benchmarking
target_compile_options(arena_benchmarks PRIVATE 
  $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -DNDEBUG>
  $<$<CXX_COMPILER_ID:MSVC>:/O2 /DNDEBUG>
)

add_executable(stack_vs_heap_benchmark stack_vs_heap_benchmark.cpp)

target_link_libraries(stack_vs_heap_benchmark 
  PRIVATE 
    RealVectorFramework::RealVectorFramework
    benchmark::benchmark
    benchmark::benchmark_main
)

# Enable optimizations for benchmarking
target_compile_options(stack_vs_heap_benchmark PRIVATE 
  $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -DNDEBUG>
  $<$<CXX_COMPILER_ID:MSVC>:/O2 /DNDEBUG>
)