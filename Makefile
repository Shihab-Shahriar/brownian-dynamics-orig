# Compiler
CXX = nvcc

# Compiler Flags
CXXFLAGS = -Xcompiler -Wall -O3 -std=c++17

INCLUDES = 

LIBS = 

# SFML Linker Flags
LDFLAGS = -lsfml-graphics -lsfml-window -lsfml-system

# Executable name
TARGET = appl


# Source files
SOURCES = brownian.cu

# Object files
OBJECTS = $(SOURCES:.cu=.o)

all: $(TARGET)


$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJECTS) -o $(TARGET) $(LIBS) $(LDFLAGS)

%.o: %.cu
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET) out.txt a.out