
VARIABLE+=TRACE_LEVEL
HELP_TRACE_LEVEL=trace verbosity
TRACE_LEVEL?=0

VARIABLE+=BUILDDIR
HELP_BUILDDIR=build directory
BUILDDIR?=build

VARIABLE+=BENCHMARKDIR
HELP_BENCHMARKDIR=benchmark directory
BENCHMARKDIR?=benchmarks

VARIABLE+=PROFILINGDIR
HELP_PROFILINGDIR=profiling directory
PROFILINGDIR?=profiling

VARIABLE+=MODELS
HELP_MODELS=existing models
ifndef MODELS
MODELS+=mnist
MODELS+=tinyyolov2
MODELS+=super_resolution
endif

VARIABLE+=OPERATORS
HELP_OPERATORS=operators to test (all if empty)

VARIABLE+=REPEAT
HELP_REPEAT=default repetition count if not otherwise specified by REPEAT_<modelname>
REPEAT=1

VARIABLE+=FORMAT
HELP_FORMAT=which files to format (git wildcards)
ifndef FORMAT
FORMAT+=**/*.h
FORMAT+=**/*.c
FORMAT+=!**/pb/**/*
FORMAT+=!**/third_party/**/*
endif

VARIABLE+=ONNX_CUSTOM
HELP_ONNX_CUSTOM=use custom onnx installation
ONNX_CUSTOM=third_party/onnx/onnx.build

VARIABLE+=ONNX_INCLUDE
HELP_ONNX_INCLUDE=which schemas to include
ifndef ONNX_INCLUDE
ONNX_INCLUDE+="^Conv$$"
ONNX_INCLUDE+="^Add$$"
ONNX_INCLUDE+="^Relu$$"
ONNX_INCLUDE+="^MaxPool$$"
ONNX_INCLUDE+="^Reshape$$"
ONNX_INCLUDE+="^MatMul$$"
ONNX_INCLUDE+="^ArgMax$$"
ONNX_INCLUDE+="^BatchNormalization$$"
ONNX_INCLUDE+="^Sigmoid$$"
ONNX_INCLUDE+="^Softmax$$"
ONNX_INCLUDE+="^Mul$$"
ONNX_INCLUDE+="^LeakyRelu$$"
ONNX_INCLUDE+="^Constant$$"
ONNX_INCLUDE+="^Transpose$$"
endif

VARIABLE+=ONNX_VERSION
HELP_ONNX_VERSION=which onnx version to use
ONNX_VERSION=latest

VARIABLE+=ONNX_DOMAINS
HELP_ONNX_DOMAINS=which onnx domains to use
ONNX_DOMAINS=onnx

VARIABLE+=ONNX_EXCLUDE
HELP_ONNX_EXCLUDE=which schemas to exclude
ONNX_EXCLUDE=

$(foreach MODEL, $(MODELS), $(eval REPEAT_$(MODEL)=$(REPEAT)))
REPEAT_tinyyolov2=1
REPEAT_super_resolution=1
REPEAT_mnist=5

CC=gcc
CFLAGS+=-std=c99
CFLAGS+=-Wall
CFLAGS+=-g
# CFLAGS+=-Werror # CI jobs run with flag enabled
CPPFLAGS+=-D TRACE_LEVEL=$(TRACE_LEVEL)

LDFLAGS+=-g
LDLIBS+=-lcunit
LDLIBS+=-lm

INCDIR+=include
INCDIR+=src
INCDIR+=src/pb
CPPFLAGS+=$(foreach DIR, $(INCDIR),-I $(DIR) )

SRCDIR+=src/operators
SRCDIR+=src/operators/info/onnx
SRCDIR+=src/operators/resolve/onnx
SRCDIR+=src/operators/implementation/onnx
SRCDIR+=src/pb
SRCS+=$(foreach DIR, $(SRCDIR), $(wildcard $(DIR)/*.c))
SRCS+=src/inference.c
SRCS+=src/trace.c
SRCS+=src/utils.c
OBJS=$(SRCS:%.c=$(BUILDDIR)/%.o)

$(BUILDDIR)/%.o:%.c
	@mkdir -p $(dir $@)
	$(CC) -o $@ -c $(CFLAGS) $(CPPFLAGS) $^

$(BINARY): $(OBJS)

DEFAULT=help

.phony: runtest
HELP_runtest=build runtest binary
ALL+=runtest
TARGET+=runtest
runtest: $(BUILDDIR)/runtest
$(BUILDDIR)/runtest: $(OBJS)
	$(CC) -o $@ src/test/tests.c $^ $(CPPFLAGS) $(LDFLAGS) $(LDLIBS)

.phony: clean_build
CLEAN+=clean_build
clean_build:
	rm -rf $(BUILDDIR)

.phony:test_operators
HELP_test_operators=run onnx backend test for each operator in OPERATORS (all if empty)
TARGET_test+=test_operators
test_operators: runtest
ifeq (,$(OPERATORS))
	$(BUILDDIR)/runtest onnxBackendSuite
else
	for OPERATOR in $(OPERATORS); do \
		$(BUILDDIR)/runtest onnxBackendSuite test_$$OPERATOR; \
	done
endif

.phony:test_models
HELP_test_models=run model test for each model in MODELS (all if empty)
TARGET_test+=test_models
test_models: runtest
ifeq (,$(MODELS))
	$(BUILDDIR)/runtest modelsTestSuite
else
	for MODEL in $(MODELS); do \
		$(BUILDDIR)/runtest modelsTestSuite test_model_$$MODEL; \
	done
endif

.phony: test
HELP_test=run tests
TARGET+=test
test: $(TARGET_test)

define BENCHMARK_MODEL
HELP_benchmark_$(1)=run $(1) benchmark
TARGET_benchmark+=benchmark_$(1)
benchmark_$(1): $(BENCHMARKDIR)/$(1).txt
#Dont trace to run faster
TRACE_LEVEL=-1
$(BENCHMARKDIR)/$(1).txt: runtest
	rm -f $(BENCHMARKDIR)/$(1).txt
	mkdir -p $(BENCHMARKDIR)
	for number in $$$$(seq $(REPEAT_$(1))) ; do \
		echo "Benchmarking iteration "$$$$number ; \
		$(BUILDDIR)/runtest modelsTestSuite test_model_$(1) >> $(BENCHMARKDIR)/$(1).txt ; \
  done
endef

$(foreach MODEL, $(MODELS), $(eval $(call BENCHMARK_MODEL,$(MODEL))))

.phony:benchmark
HELP_benchmark=run benchmarks of all MODELS
TARGET+=benchmark
benchmark: $(TARGET_benchmark)
	rm -f $@
	mkdir -p $(dir $@)
	# Run some postprocessing on the benchmarking results
	python scripts/parse_output_benchmarking.py

.phony:clean_benchmark
CLEAN+=clean_benchmark
clean_benchmark:
	rm -rf $(BENCHMARKDIR)

.phony:connxr
HELP_connxr=build connxr binary
TARGET+=connxr
ALL+=connxr
connxr: $(BUILDDIR)/connxr
$(BUILDDIR)/connxr: $(OBJS)
	$(CC) -o $@ src/connxr.c $^ $(CPPFLAGS) $(LDFLAGS) $(LDLIBS)

define PROFILING_MODEL
HELP_profiling_$(1)=run $(1) profiling
TARGET_profiling+=profiling_$(1)
profiling_$(1): $(PROFILINGDIR)/$(1).txt
$(PROFILINGDIR)/$(1).txt: runtest
	mkdir -p $(PROFILINGDIR)
	valgrind --tool=callgrind --callgrind-out-file=$(PROFILINGDIR)/$(1).txt ./$(BUILDDIR)/runtest modelsTestSuite test_model_$(1)
endef

$(foreach MODEL, $(MODELS), $(eval $(call PROFILING_MODEL,$(MODEL))))

.phony:profiling
HELP_profiling=run profiling of all MODELS
TARGET+=profiling
profiling: $(TARGET_profiling)

.phony:clean_profiling
CLEAN+=clean_profiling
clean_profiling:
	rm -rf $(PROFILINGDIR)

#memory leak stuff TODO:

#gprof:
#	rm -f gprof
#	gcc -std=c99 -D xxx -pg ../src/operators/*.c ../src/trace.c ../src/utils.c ../src/inference.c ../src/pb/onnx.pb-c.c -o gprof tests.c -I/usr/local/include -L/usr/local/lib -lcunit -lprotobuf-c
#	./gprof $(ts) $(tc)

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
	cd scripts; python -m onnx_generator \
	$(if $(ONNX_INCLUDE), --include $(ONNX_INCLUDE)) \
	$(if $(ONNX_EXCLUDE), --exclude $(ONNX_EXCLUDE)) \
	$(if $(ONNX_VERSION), --version $(ONNX_VERSION)) \
	$(if $(ONNX_DOMAINS), --domains $(ONNX_DOMAINS)) \
	-vv \
	--force-header \
	--force-resolve \
	--force-sets \
	--force-info \
	$(shell git rev-parse --show-toplevel)

include .Makefile.template
