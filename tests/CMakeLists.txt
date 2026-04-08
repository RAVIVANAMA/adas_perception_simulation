cmake_minimum_required(VERSION 3.18)

# Gather all test sources
file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_executable(adas_tests ${TEST_SOURCES})

target_link_libraries(adas_tests
  PRIVATE
    adas_core
    GTest::gtest_main
    GTest::gmock
)

target_include_directories(adas_tests PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)

# Register with CTest
include(GoogleTest)
gtest_discover_tests(adas_tests)
