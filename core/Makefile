CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Wpedantic -Werror

build/libtrackfuse.so: ekf.cpp capi.cpp ekf.hpp mat.hpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) -shared -fPIC -o $@ ekf.cpp capi.cpp

build/test_core: ekf.cpp test_core.cpp ekf.hpp mat.hpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) -fsanitize=address,undefined -g -o $@ ekf.cpp test_core.cpp

.PHONY: test clean
test: build/test_core
	./build/test_core

clean:
	rm -rf build
