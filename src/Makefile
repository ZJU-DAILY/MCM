.DEFAULT_GOAL := all

CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -msse4.1 -pthread -fopenmp
DEPFLAGS = -MMD -MP
BUILD = .build

COMMON = IterativeMLCore DynamicMLCore SkylineSet
MLCDEC = Graphs/Graph Graphs/MultilayerGraph Core/MLCore Core/KC \
         Core/MLCTree Core/PMLCTreeBuilder Core/MLCTreeBuilder_simp Core/MLCHashTable

OBJECTS = $(addprefix $(BUILD)/,$(addsuffix .o,$(COMMON))) \
          $(addprefix $(BUILD)/MlcDec/,$(addsuffix .o,$(MLCDEC)))
DEPS = $(OBJECTS:.o=.d) $(BUILD)/main.d $(BUILD)/run_baseline.d

.PHONY: all clean
all: run_test run_baseline

run_test: $(BUILD)/main.o $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

run_baseline: $(BUILD)/run_baseline.o $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(BUILD)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@

$(BUILD)/MlcDec/%.o: ../MlcDec/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD)
	rm -f run_test run_baseline

-include $(DEPS)
