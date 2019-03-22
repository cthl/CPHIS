cmake_minimum_required (VERSION 2.8) 
project (cphis_tpetra CXX)

# Include Trilinos configuration.
INCLUDE (${TRILINOS_DIR}/lib/cmake/Trilinos/TrilinosConfig.cmake)

set (CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER})

add_library (cphistpetra STATIC "cphis_tpetra.cc")

# Set include directories.
include_directories ("." "../../include")
# The SYSTEM flag supresses warnings from these sources.
include_directories (SYSTEM "${Trilinos_INCLUDE_DIRS}")
if (Trilinos_TPL_INCLUDE_DIRS)
  include_directories (SYSTEM "${Trilinos_TPL_INCLUDE_DIRS}")
endif (Trilinos_TPL_INCLUDE_DIRS)

# Set linker flags.
set (LDFLAGS "${LDFLAGS} ${Trilinos_EXTRA_LD_FLAGS} ") 
set (CMAKE_EXE_LINKER_FLAGS " ${LDFLAGS} ")

# Set compiler flags.
set (CXXFLAGS "${CXXFLAGS} ${Trilinos_CXX_COMPILER_FLAGS} ")
set (CMAKE_CXX_FLAGS " ${CXXFLAGS} ")
