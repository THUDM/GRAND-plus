# ============================================================================
# Add targets add their dependencies
# ============================================================================
set(SFMT_params
  SFMT-params.h
  SFMT-params607.h
  SFMT-params1279.h
  SFMT-params2281.h
  SFMT-params4253.h
  SFMT-params11213.h
  SFMT-params19937.h
  SFMT-params44497.h
  SFMT-params86243.h
  SFMT-params132049.h
  SFMT-params216091.h)

add_library(sfmt
  SFMT.h
  SFMT.c
  jump/SFMT-jump.h
  jump/SFMT-jump.cpp
  SFMT-common.h
  ${SFMT_params})


# ============================================================================
# Use GSL and zlib
# ============================================================================
target_link_libraries(sfmt PUBLIC ${ZLIB_LIBRARY})


# ============================================================================
# Make STDC MACROS available (for fixed width integers)
# ============================================================================
target_compile_definitions(sfmt
  PUBLIC __STDC_FORMAT_MACROS
  PUBLIC __STDC_CONSTANT_MACROS)


# ============================================================================
# Silence -Wunused-const-variable warning
# ============================================================================
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wno-unused-const-variable")
