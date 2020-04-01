
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
endif

VARIABLE+=OPERATORS
HELP_OPERATORS=operators to test (all if empty)

VARIABLE+=REPEAT
HELP_REPEAT=default repetition count if not otherwise specified by REPEAT_<modelname>
REPEAT=10

$(foreach MODEL, $(MODELS), $(eval REPEAT_$(MODEL)=$(REPEAT)))
REPEAT_tinyyolov2=1

CC=gcc
CFLAGS+=-std=c99
CFLAGS+=-Wall
# CFLAGS+=-Werror # CI jobs run with flag enabled
CPPFLAGS+=-D TRACE_LEVEL=$(TRACE_LEVEL)

#LDFLAGS+=
LDLIBS+=-lcunit
LDLIBS+=-lm

SRCDIR+=src/operators
SRCDIR+=src/pb
SRCS+=$(foreach DIR, $(SRCDIR), $(wildcard $(DIR)/*.c))
SRCS+=src/inference.c
SRCS+=src/trace.c
SRCS+=src/utils.c
OBJS=$(SRCS:%.c=$(BUILDDIR)/obj/%.o)

$(BUILDDIR)/obj/%.o:%.c
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
	$(CC) -o $@ test/tests.c $^ $(LDFLAGS) $(LDLIBS)

.phony: clean_runtest
CLEAN+=clean_runtest
clean_runtest:
	rm -f $(BUILDDIR)/runtest
	-rmdir $(BUILDDIR)

.phony: clean_objs
CLEAN+=clean_objs
clean_objs:
	rm -rf $(BUILDDIR)/obj
	-rmdir $(BUILDDIR)

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
$(BENCHMARKDIR)/$(1).txt: runtest
	# TODO Benchmarking should run without many logging crap to avoid performance loss
	# All runs will be average later on in the post processing phase
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
benchmark: $(BENCHMARKDIR)/result.txt
$(BENCHMARKDIR)/result.txt: $(TARGET_benchmark)
	rm -f $@
	mkdir -p $(dir $@)
	cat $(BENCHMARKDIR)/*.txt > $@
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
	$(CC) -o $@ src/connxr.c $^ $(LDFLAGS) $(LDLIBS)

.phony:clean_connxr
CLEAN+=clean_connxr
clean_connxr:
	rm -f $(BUILDDIR)/connxr
	-rmdir $(BUILDDIR)

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

#nanopb:
#	rm -f prototest
#	gcc -std=c99 -Wall -D xxx -o prototest test_nanopb.c ../src/pb/nanopb/onnx.pb.c ../src/pb/nanopb/pb_common.c ../src/pb/nanopb/pb_decode.c ../src/pb/nanopb/pb_encode.c -I/usr/local/include -L/usr/local/lib -lcunit
#	./prototest $(ts) $(tc)

#gprof:
#	rm -f gprof
#	gcc -std=c99 -D xxx -pg ../src/operators/*.c ../src/trace.c ../src/utils.c ../src/inference.c ../src/pb/onnx.pb-c.c -o gprof tests.c -I/usr/local/include -L/usr/local/lib -lcunit -lprotobuf-c
#	./gprof $(ts) $(tc)

include .Makefile.template
