# TInCuP - A library for generating and validating C++ customization point objects that use `tag_invoke`
#
# Copyright (c) National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# Questions? Contact Greg von Winckel (gvonwin@sandia.gov)

cmake_minimum_required( VERSION 3.22 )

cmake_policy( SET CMP0115 NEW )

# Enable Hot Reload for MSVC compilers if supported.
if( POLICY CMP0141 )
  cmake_policy( SET CMP0141 NEW )
  set( CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>" )
endif()

project( "RealVectorFramework" VERSION 1.0 )

set( CMAKE_CXX_STANDARD 20 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )

if(APPLE)
  set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Build architectures for Mac OS X" FORCE)
endif()

list( APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" )


option( ENABLE_EXAMPLES "Enable Build Examples" ON )

if( ENABLE_EXAMPLES )
  add_subdirectory(examples)
endif()


include(FetchContent)

FetchContent_Declare(tincup
  GIT_REPOSITORY https://github.com/sandialabs/TInCuP.git
  GIT_TAG main  
)
FetchContent_MakeAvailable(tincup)


option( ENABLE_TESTING "Enable testing with CTest" ON )
option( ENABLE_GTEST_COLORED_OUTPUT "Enable colored output for GTest" ON )

add_library(RealVectorFramework INTERFACE)

add_library(RealVectorFramework::RealVectorFramework ALIAS RealVectorFramework)

target_include_directories(RealVectorFramework INTERFACE include)

target_link_libraries(RealVectorFramework INTERFACE tincup::tincup)

if( ENABLE_TESTING)
  set(GOOGLETEST_VERSION v1.14.0)

  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/${GOOGLETEST_VERSION}.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )

  FetchContent_MakeAvailable(googletest)

  enable_testing()
  add_subdirectory(test)
endif()
