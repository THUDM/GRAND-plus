############################################
#                 Variables                #
############################################

AUTOMAKE_OPTIONS = subdir-objects

########################
# Set library to build #
########################
noinst_LIBRARIES = libsfmt.a


AM_CPPFLAGS = -I./

############################################
# C99 exact-width integer specific support #
############################################
# __STDC_FORMAT_MACROS allows us to use exact-width integer format specifiers e.g. PRId32 (for printf etc)
# __STDC_CONSTANT_MACROS allows us to define exact-width integer macros with e.g. INT32_C( <value> )
# __STDC_LIMIT_MACROS allows us to use exact-width integer limit macros with e.g. INT32_MAX
AM_CPPFLAGS += -D__STDC_FORMAT_MACROS -D__STDC_CONSTANT_MACROS -D__STDC_LIMIT_MACROS



SFMT_params = SFMT-params.h SFMT-params607.h SFMT-params1279.h SFMT-params2281.h SFMT-params4253.h SFMT-params11213.h SFMT-params19937.h SFMT-params44497.h SFMT-params86243.h SFMT-params132049.h  SFMT-params216091.h

libsfmt_adir = ./
libsfmt_a_CPPFLAGS = $(AM_CPPFLAGS)
#libsfmt_a_HEADERS = SFMT.h SFMT-common.h $(SFMT_params) jump/SFMT-jump.h
libsfmt_a_SOURCES = SFMT.c jump/SFMT-jump.cpp
libsfmt_a_SOURCES += SFMT.h SFMT-common.h $(SFMT_params) jump/SFMT-jump.h
