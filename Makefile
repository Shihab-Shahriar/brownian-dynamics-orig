# Compiler
CXX = nvcc

# Compiler Flags
CXXFLAGS = -Xcompiler -Wall -O3 -std=c++17

# SFML include directory
INCLUDES = 
#-I/path/to/SFML/include

# SFML library directory
LIBS = 
#-L/path/to/SFML/lib

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
