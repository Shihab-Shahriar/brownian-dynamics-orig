# Compiler
CXX = g++

# Compiler Flags
CXXFLAGS = -Wall -O1 -g

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
SOURCES = brownian.cpp

# Object files
OBJECTS = $(SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJECTS) -o $(TARGET) $(LIBS) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET) out.txt
