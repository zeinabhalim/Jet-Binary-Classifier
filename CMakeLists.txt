cmake_minimum_required(VERSION 3.10)

project(QuarkGluonJets VERSION 1.0 LANGUAGES CXX)

# -------------------------------------------------
# C++ Standard
# -------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# -------------------------------------------------
# Find PYTHIA8
# -------------------------------------------------
find_program(PYTHIA8_CONFIG pythia8-config)

if(NOT PYTHIA8_CONFIG)
    message(FATAL_ERROR "pythia8-config not found. Install PYTHIA8 and ensure it is in PATH.")
endif()

execute_process(
    COMMAND ${PYTHIA8_CONFIG} --cxxflags
    OUTPUT_VARIABLE PYTHIA8_CXXFLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND ${PYTHIA8_CONFIG} --libs
    OUTPUT_VARIABLE PYTHIA8_LIBS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# -------------------------------------------------
# Find FastJet
# -------------------------------------------------
find_program(FASTJET_CONFIG fastjet-config HINTS $ENV{FASTJET}/bin)

if(NOT FASTJET_CONFIG)
    message(FATAL_ERROR "fastjet-config not found. Install FastJet correctly.")
endif()

execute_process(
    COMMAND ${FASTJET_CONFIG} --cxxflags
    OUTPUT_VARIABLE FASTJET_CXXFLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND ${FASTJET_CONFIG} --libs
    OUTPUT_VARIABLE FASTJET_LIBS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# -------------------------------------------------
# Executable
# -------------------------------------------------
add_executable(generate
    src/generate_pythia_jets.cc
)

target_compile_options(generate PRIVATE
    ${PYTHIA8_CXXFLAGS}
    ${FASTJET_CXXFLAGS}
)

target_link_libraries(generate
    ${PYTHIA8_LIBS}
    ${FASTJET_LIBS}
    Nsubjettiness
)

message(STATUS "PYTHIA8 flags: ${PYTHIA8_CXXFLAGS}")
message(STATUS "FastJet flags: ${FASTJET_CXXFLAGS}")
