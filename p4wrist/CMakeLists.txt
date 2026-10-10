cmake_minimum_required(VERSION 3.16)
project(p4wrist VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)

# One shared library: C++ classes for C++ users, the C API (p4wrist.h) for
# everything else (python/p4wrist.py loads it with ctypes).
add_library(p4wrist SHARED
  src/c_api.cpp
  src/canopen.cpp
  src/clock.cpp
  src/socketcan.cpp
  src/wrist.cpp)
target_include_directories(p4wrist PUBLIC include)
target_link_libraries(p4wrist PUBLIC Threads::Threads)
target_compile_options(p4wrist PRIVATE -Wall -Wextra -Wpedantic)

enable_testing()
add_executable(p4wrist_tests tests/test_core.cpp)
target_link_libraries(p4wrist_tests PRIVATE p4wrist)
target_compile_options(p4wrist_tests PRIVATE -Wall -Wextra)
add_test(NAME p4wrist_tests COMMAND p4wrist_tests)
