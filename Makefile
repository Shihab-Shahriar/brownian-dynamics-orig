# Compiler
CXX = nvcc 

# Compiler Flags
CXXFLAGS = -Xcompiler -Wall -O3 -std=c++17 -arch=sm_70

# Linker Flags
LDFLAGS = -lcurand

# Executable name
TARGET = appl

# Source files
SOURCES = brownian.cu

# Object files
OBJECTS = $(SOURCES:.cu=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

%.o: %.cu
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET) out.txt a.out
