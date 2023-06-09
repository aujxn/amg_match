cmake_minimum_required(VERSION 3.0)
project(SPE10Example)

set(MFEM_DIR /home/austen/school/s2023/mth653/project/mfem/install/lib/cmake/mfem)
find_package(MFEM REQUIRED)

include_directories(${MFEM_INCLUDE_DIRS})
add_executable(spe10 spe10.cpp spe10.hpp)
target_link_libraries(spe10 PRIVATE ${MFEM_LIBRARIES})
