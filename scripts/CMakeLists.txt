cmake_minimum_required(VERSION 3.0)
project(SPE10Example)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")

set(MFEM_DIR /home/austen/school/s2023/mth653/project/mfem/install/lib/cmake/mfem)
find_package(MFEM REQUIRED)

include_directories(${MFEM_INCLUDE_DIRS})
add_executable(spe10 spe10.cpp spe10.hpp)
add_executable(anisotropic anisotropic.cpp)
add_executable(laplace simple_laplace_hierarchy.cpp)
target_link_libraries(spe10 PRIVATE ${MFEM_LIBRARIES})
target_link_libraries(anisotropic PRIVATE ${MFEM_LIBRARIES})
target_link_libraries(laplace PRIVATE ${MFEM_LIBRARIES})
