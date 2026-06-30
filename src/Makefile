CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -msse4.1 -pthread -fopenmp
DEPFLAGS = -MMD -MP

TARGET = run_test
BASELINE = run_baseline

SRCS_TEST = main.cpp IterativeMLCore.cpp SkylineSet.cpp DynamicMLCore.cpp
SRCS_BASELINE = run_baseline.cpp IterativeMLCore.cpp SkylineSet.cpp DynamicMLCore.cpp
SRCS_ORIGINAL = ../MlcDec/Graphs/Graph.cpp \
                ../MlcDec/Graphs/MultilayerGraph.cpp \
                ../MlcDec/Core/MLCore.cpp \
                ../MlcDec/Core/KC.cpp \
                ../MlcDec/Core/MLCTree.cpp \
                ../MlcDec/Core/PMLCTreeBuilder.cpp \
                ../MlcDec/Core/MLCTreeBuilder_simp.cpp \
                ../MlcDec/Core/MLCHashTable.cpp

OBJS_TEST = $(SRCS_TEST:.cpp=.o) $(SRCS_ORIGINAL:.cpp=.o)
OBJS_BASELINE = $(SRCS_BASELINE:.cpp=.o) $(SRCS_ORIGINAL:.cpp=.o)
DEPS = $(sort $(OBJS_TEST:.o=.d) $(OBJS_BASELINE:.o=.d))

all: $(TARGET) $(BASELINE)

$(TARGET): $(OBJS_TEST)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS_TEST)
	@echo "=================================================="
	@echo "Build successful! Executable is ready: ./$(TARGET)"
	@echo "=================================================="

$(BASELINE): $(OBJS_BASELINE)
	$(CXX) $(CXXFLAGS) -o $(BASELINE) $(OBJS_BASELINE)
	@echo "=================================================="
	@echo "Build successful! Executable is ready: ./$(BASELINE)"
	@echo "=================================================="

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS_TEST) $(OBJS_BASELINE) $(DEPS) $(TARGET) $(BASELINE)

-include $(DEPS)
