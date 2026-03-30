# ============================
# Compiler settings
# ============================
GCC      = g++
NVCC     = nvcc

# Use Intel MPI with GCC fallback if Intel compiler not installed
# Uncomment below line if Intel MPI is installed and icpc missing
export I_MPI_CXX=g++

# ============================
# Flags for OpenCV
# ============================
OPENCV = `pkg-config --cflags --libs opencv4`

# ============================
# Directories
# ============================
SRC_DIR      = src
IMAGE_DIR    = Data/images

# ============================
# VTune Profiling
# ============================
VTUNE = vtune
VTUNE_OPTS = -collect hotspots    # Change to `mpi-hotspots` for MPI profiling
VTUNE_RESULT_DIR = vtune_results

# ============================
# Source files
# ============================
SEQ_SRC           = $(SRC_DIR)/sequential/Sequential_OPENCV.cpp
CUDA_SRC          = $(SRC_DIR)/cuda/parallel_CUDA_OPENCV.cu #src/tests/parallel_CUDA_OPENCV.cu

# ============================
# Executables
# ============================
SEQ_EXEC           = build/serial
CUDA_EXEC          = build/cuda_parallel

# ============================
# Configurable variables
# ============================
CORE  ?= 8
IMAGE ?= $(IMAGE_DIR)/72x72.png

# ============================
# Default target
# ============================
.DEFAULT_GOAL := help

# ============================
# Build all
# ============================
all: prepare $(SEQ_EXEC) $(CUDA_EXEC)

# Create build directory if missing
prepare:
	@mkdir -p build

# ============================
# Build targets
# ============================

$(SEQ_EXEC): $(SEQ_SRC)
	$(GCC)  -o $(SEQ_EXEC) $(SEQ_SRC) $(OPENCV)

$(CUDA_EXEC): $(CUDA_SRC)
	$(NVCC)  -arch=sm_60 -o $(CUDA_EXEC) $(CUDA_SRC) $(OPENCV)

# ============================
# Run targets
# ============================

run-serial: $(SEQ_EXEC)
	./$(SEQ_EXEC) $(IMAGE)

run-cuda: $(CUDA_EXEC)
	./$(CUDA_EXEC) $(IMAGE)


# ============================
# Clean target
# ============================
clean:
	rm -rf build
	@mkdir -p build


profile-hybrid: $(HYBRID_EXEC)
	@mkdir -p $(VTUNE_RESULT_DIR)
	$(VTUNE) $(VTUNE_OPTS) -r $(VTUNE_RESULT_DIR)/hybrid_profile mpirun -np $(CORE) ./$(HYBRID_EXEC) $(IMAGE)

vtune-report:
	$(VTUNE) -report hotspots -r $(VTUNE_RESULT_DIR)/mpi_profile -format html -report-output $(VTUNE_RESULT_DIR)/report.html



# ============================
# Help target
# ============================
help:
	@echo "================== Available targets =================="
	@echo "Build Targets:"
	@echo "  all                     - Build all executables"
	@echo "  $(MPI_EXEC)             - Build MPI + OpenCV version"
	@echo "  $(MPI_OPT_EXEC)         - Build MPI + OpenMP + OpenCV version"
	@echo "  $(SEQ_EXEC)             - Build Sequential OpenCV version"
	@echo "  $(CUDA_EXEC)            - Build CUDA + OpenCV version"
	@echo "  $(CUDA_SM_EXEC)         - Build CUDA Shared Memory + OpenCV version"
	@echo "  $(HYBRID_EXEC)          - Build Hybrid MPI + OpenMP + CUDA + OpenCV version"
	@echo "  $(TEXTON_HYBRID_EXEC)   - Build MPI + OpenMP + OpenCV (Texton) version"
	@echo ""
	@echo "Run Targets:"
	@echo "  run-mpi                 - Run MPI version (np=$(CORE), image=$(IMAGE))"
	@echo "  run-mpi-optimized       - Run MPI + OpenMP optimized version"
	@echo "  run-serial              - Run Sequential version"
	@echo "  run-cuda                - Run CUDA version"
	@echo "  run-cuda-shared         - Run CUDA Shared Memory version"
	@echo "  run-hybrid              - Run Hybrid MPI+OpenMP+CUDA version"
	@echo "  run-texton-hybrid       - Run MPI+OpenMP+OpenCV Texton version"
	@echo ""
	@echo "Profiling Targets (VTune):"
	@echo "  profile-mpi             - Profile MPI version using VTune (hotspots)"
	@echo "  profile-mpi-optimized   - Profile MPI + OpenMP version using VTune"
	@echo "  profile-hybrid          - Profile Hybrid MPI+OpenMP+CUDA version using VTune"
	@echo "  vtune-report            - Generate VTune HTML report for MPI profiling results"
	@echo ""
	@echo "Utilities:"
	@echo "  clean                   - Remove all build artifacts and recreate build directory"
	@echo ""
	@echo "Configurable Variables (override at runtime, e.g., make CORE=4 IMAGE=Data/images/img.jpg):"
	@echo "  CORE                    - Number of MPI processes (default: 8)"
	@echo "  IMAGE                   - Path to input image (default: $(IMAGE_DIR)/real1.jpeg)"
	@echo "  VTUNE_OPTS              - Options for VTune collection (default: -collect hotspots)"
	@echo "========================================================"

# ============================
# Phony targets
# ============================
.PHONY: all clean help run-mpi run-mpi-optimized run-serial run-cuda run-cuda-shared run-hybrid run-texton-hybrid prepare
