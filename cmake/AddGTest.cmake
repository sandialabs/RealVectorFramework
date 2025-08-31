# TInCuP - A library for generating and validating C++ customization point objects that use `tag_invoke`
#
# Copyright (c) National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# Questions? Contact Greg von Winckel (gvonwin@sandia.gov)

# File: ${CMAKE_PROJECT_SOURCE_DIR}/cmake/AddGTest.cmake
function(add_gtest TEST_NAME)
  # The first argument is the test name, the rest are source files
  set(SOURCES ${ARGN})

  # Explicitly add the .cpp extension to source files if not present
  set(PROCESSED_SOURCES)
  foreach(SOURCE ${SOURCES})
    get_filename_component(SOURCE_EXT ${SOURCE} EXT)
    if(SOURCE_EXT STREQUAL "")
      list(APPEND PROCESSED_SOURCES "${SOURCE}.cpp")
    else()
      list(APPEND PROCESSED_SOURCES ${SOURCE})
    endif()
  endforeach()

  add_executable(${TEST_NAME} ${PROCESSED_SOURCES})

  # Use PRIVATE to prevent transitive propagation of gtest libraries
  target_link_libraries(${TEST_NAME} PRIVATE gtest_main real_vector_framework)

  # Suppress GTest warnings based on compiler
  if(MSVC)
    # MSVC-specific warning suppression
    target_compile_options(${TEST_NAME} PRIVATE
      /experimental:external
      /external:anglebrackets
      /external:W0)
  else()
    # GCC/Clang warning suppression
    target_compile_options(${TEST_NAME} PRIVATE
      -Wno-error=unused-parameter
      $<$<CXX_COMPILER_ID:Clang>:-Wno-global-constructors>
      $<$<CXX_COMPILER_ID:GNU>:-Wno-global-constructors>
      $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:
        -isystem
        ${CMAKE_BINARY_DIR}/_deps/gtest-src/googletest/src
        ${CMAKE_BINARY_DIR}/_deps/gtest-src/googletest/include>)
  endif()

  add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
  if( ENABLE_GTEST_COLORED_OUTPUT )
    set_tests_properties(${TEST_NAME} PROPERTIES
        ENVIRONMENT "GTEST_COLOR=yes")
  endif()
endfunction()