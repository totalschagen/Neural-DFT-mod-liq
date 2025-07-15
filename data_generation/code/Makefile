CXX = g++
CXXFLAGS = -Wall -O3 -fopenmp -std=c++17 -I./src/modular
LDFLAGS = -lm -lstdc++fs

## directories
SRCDIR = src
BUILDDIR = build
BINDIR = $(BUILDDIR)/exe

CORE_SRCS = $(wildcard $(SRCDIR)/modular/*.cpp)
MAIN_SRCS = $(wildcard $(SRCDIR)/mains/*.cpp)

CORE_OBJS = $(patsubst $(SRCDIR)/modular/%.cpp, $(BUILDDIR)/modular/%.o,$(CORE_SRCS))

EXECUTABLES = $(patsubst $(SRCDIR)/mains/%.cpp, $(BINDIR)/%,$(MAIN_SRCS))

# build all executables
.PHONY: all
all :  directories $(EXECUTABLES)

$(BINDIR)/%: $(BUILDDIR)/mains/%.o $(CORE_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)


# compile corefiles
$(BUILDDIR)/modular/%.o: $(SRCDIR)/modular/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

#compile mainfiles
$(BUILDDIR)/mains/%.o: $(SRCDIR)/mains/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# directory creation rule
.PHONY: directories
directories:
	@mkdir -p $(BINDIR) $(BUILDDIR)/modular $(BUILDDIR)/mains

#clean
.PHONY: clean
clean:
	rm -rf $(BUILDDIR)/*
