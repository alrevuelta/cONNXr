
VARIABLE+=TRACE_LEVEL
HELP_TRACE_LEVEL=trace verbosity
# TRACE_LEVEL?=0

VARIABLE+=BUILDDIR
HELP_BUILDDIR=build directory
BUILDDIR?=build

VARIABLE+=PROFILINGDIR
HELP_PROFILINGDIR=profiling directory
PROFILINGDIR?=profiling

VARIABLE+=FORMAT
HELP_FORMAT=which files to format (git wildcards)
ifndef FORMAT
FORMAT+=**/*.h
FORMAT+=**/*.c
FORMAT+=!**/protobuf/**/*
FORMAT+=!**/third_party/**/*
endif

VARIABLE+=ONNX_CUSTOM
HELP_ONNX_CUSTOM=use custom onnx installation
ONNX_CUSTOM=third_party/onnx/onnx.build

VARIABLE+=ONNX_INCLUDE
HELP_ONNX_INCLUDE=which schemas to include
ifndef ONNX_INCLUDE
ONNX_INCLUDE+="^Add$$"
ONNX_INCLUDE+="^ArgMax$$"
ONNX_INCLUDE+="^BatchNormalization$$"
ONNX_INCLUDE+="^Clip$$"
ONNX_INCLUDE+="^Constant$$"
ONNX_INCLUDE+="^Conv$$"
ONNX_INCLUDE+="^GlobalAveragePool$$"
ONNX_INCLUDE+="^LeakyRelu$$"
ONNX_INCLUDE+="^MatMul$$"
ONNX_INCLUDE+="^MaxPool$$"
ONNX_INCLUDE+="^Mul$$"
ONNX_INCLUDE+="^Relu$$"
ONNX_INCLUDE+="^Reshape$$"
ONNX_INCLUDE+="^Sigmoid$$"
ONNX_INCLUDE+="^Softmax$$"
ONNX_INCLUDE+="^Transpose$$"
ONNX_INCLUDE+="^Elu$$"
ONNX_INCLUDE+="^Identity$$"
endif

VARIABLE+=ONNX_VERSION
HELP_ONNX_VERSION=which onnx version to use
ONNX_VERSION=latest

VARIABLE+=ONNX_DOMAINS
HELP_ONNX_DOMAINS=which onnx domains to use
ONNX_DOMAINS=ai.onnx

VARIABLE+=ONNX_EXCLUDE
HELP_ONNX_EXCLUDE=which schemas to exclude
ONNX_EXCLUDE=

CC=gcc
CFLAGS+=-std=c99
CFLAGS+=-Wall
CFLAGS+=-g3 -gdwarf -O2
# CFLAGS+=-Werror # CI jobs run with flag enabled
ifdef TRACE_LEVEL
CPPFLAGS+=-D "TRACE_LEVEL=$(TRACE_LEVEL)"
endif

LDFLAGS+=-g
LDLIBS+=-lcunit
LDLIBS+=-lm

INCDIR+=include
INCDIR+=protobuf
CPPFLAGS+=$(foreach DIR, $(INCDIR),-I $(DIR) )

SRCDIR+=src/operators
SRCDIR+=protobuf
SRCS+=$(foreach DIR, $(SRCDIR), $(shell find $(DIR) -type f -name '*.c'))
SRCS+=src/inference.c
SRCS+=src/trace.c
SRCS+=src/utils.c
SRCS+=src/test/test_utils.c
OBJS=$(SRCS:%.c=$(BUILDDIR)/%.o)

$(BUILDDIR)/%.o:%.c
	@mkdir -p $(dir $@)
	$(CC) -o $@ -c $(CFLAGS) $(CPPFLAGS) $^

$(BINARY): $(OBJS)

DEFAULT=help

# TODO: Define new objects that are compiled with -fic?
.phony: sharedlib
HELP_sharedlib=build sharedlib binary
ALL+=sharedlib
TARGET+=sharedlib
sharedlib: CFLAGS += -fpic
sharedlib: $(BUILDDIR)/sharedlib
$(BUILDDIR)/sharedlib: $(OBJS)
	$(CC) -shared -o $(BUILDDIR)/libconnxr.so -fpic $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) $(LDLIBS) `find build/ -iname '*.o' -type f`

.phony: clean_build
CLEAN+=clean_build
clean_build:
	rm -rf $(BUILDDIR)

# C unit tests, not related to models and operators
.phony: unittests
HELP_unittests=Build and run unit tests that are not related to models or operators
ALL+=unittests
TARGET+=unittests
unittests: $(BUILDDIR)/unittests
$(BUILDDIR)/unittests: $(OBJS)
	$(CC) -o $@ src/test/tests.c $^ $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) $(LDLIBS)
	$(BUILDDIR)/unittests 

# Operator tests
.phony:test_operators
HELP_test_operators=run onnx backend operator tests
TARGET_test+=test_operators
test_operators: sharedlib
	python3 tests/test_operators.py

# Model tests
.phony:test_models
HELP_test_models=run model tests
TARGET_test+=test_models
test_models: sharedlib
	python3 tests/test_models.py

.phony: test
HELP_test=run tests
TARGET+=test
test: $(TARGET_test)

.phony:benchmark
HELP_benchmark=run benchmarks of all MODELS
TARGET+=benchmark
benchmark: sharedlib
	python3 tests/benchmarking.py

.phony:connxr
HELP_connxr=build connxr binary
TARGET+=connxr
ALL+=connxr
connxr: $(BUILDDIR)/connxr
$(BUILDDIR)/connxr: $(OBJS)
	$(CC) -o $@ src/connxr.c $^ $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) $(LDLIBS)

.phony:format
HELP_format=run uncrustify to format code
TARGET+=format
format:
	git ls-files -ico $(foreach PATTERN, $(FORMAT),-x '$(PATTERN)' ) | uncrustify -c ./.uncrustify.cfg -F - --no-backup --replace

.phony:format-check
HELP_format-check=check if code needs formatting and show diffs
TARGET+=format-check
format-check:
	git ls-files -ico $(foreach PATTERN, $(FORMAT),-x '$(PATTERN)' ) \
	| xargs -I % sh -c \
	"uncrustify -c ./.uncrustify.cfg -f % \
	 | git -c 'color.diff.new=normal 22' -c 'color.diff.old=normal 88' diff --exit-code --no-index --color --word-diff=color % -"

.phony:onnx_generator
HELP_onnx_generator=generate various onnx sources and headers
TARGET+=onnx_generator
onnx_generator:
	python3 -m venv venv
	. venv/bin/activate; pip install -r requirements.txt
	cd scripts; python3 -m onnx_generator \
	$(if $(ONNX_INCLUDE), --include $(ONNX_INCLUDE)) \
	$(if $(ONNX_EXCLUDE), --exclude $(ONNX_EXCLUDE)) \
	$(if $(ONNX_VERSION), --version $(ONNX_VERSION)) \
	$(if $(ONNX_DOMAINS), --domains $(ONNX_DOMAINS)) \
	-vv \
	--force-resolve \
	--force-sets \
	--force-info \
	$(shell git rev-parse --show-toplevel)

.phony: distclean_venv
DISTCLEAN+=distclean_venv
distclean_venv:
	rm -rf venv

.phony:generate_custom_tests
HELP_generate_custom_tests=generate the custom test models using the py scripts
TARGET+=generate_custom_tests
generate_custom_tests:
	python3 test_data/generate_custom_tests.py generate-data -o test_data

include .Makefile.template
