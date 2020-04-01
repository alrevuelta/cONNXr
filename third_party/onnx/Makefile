
.PHONY: build
TARGET += build
DEFAULT += build
ALL += build
build: | ./onnx.build
./onnx.build: | check-onnx
	mkdir -p onnx.build;
	cd onnx.build; \
	cmake \
	  -j $(shell grep -c 'processor' /proc/cpuinfo)\
	  -DONNX_ML=1 \
		-DBUILD_ONNX_PYTHON=y \
	  ../onnx; \
	cmake \
	  --build . \
	  -j $(shell grep -c 'processor' /proc/cpuinfo)\
	  --target onnx_cpp2py_export

.PHONY: clean-onnx
CLEAN += clean-onnx
clean-onnx:
	git -C onnx clean -dfx

.PHONY: distclean-onnx
DISTCLEAN += distclean-onnx
distclean-onnx:
	rm -rf onnx.build
	git submodule deinit --force onnx 

.PHONY: check-onnx
CHECK += check-onnx
check-onnx:
	git submodule update --init onnx
	cd onnx;\
	git submodule update --init --recursive

include ../../.Makefile.template