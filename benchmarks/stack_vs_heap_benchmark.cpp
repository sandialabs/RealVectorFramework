/**
 * @file stack_vs_heap_benchmark.cpp
 * @brief Benchmark comparing stack allocation (std::array) vs heap allocation vs arena
 */

#include <core/rvf.hpp>
#include <operations/memory/memory_arena.hpp>
#include <benchmark/benchmark.h>
#include <vector>
#include <array>
#include <memory>

using namespace rvf;

//=============================================================================
// Compile-time sized vector using std::array (stack allocation)
//=============================================================================

template<std::size_t N>
class StackVector {
private:
  std::array<double, N> data_;
  
public:
  constexpr StackVector() : data_{} {}
  explicit constexpr StackVector(double fill_value) {
    data_.fill(fill_value);
  }
  StackVector(const StackVector&) = default;
  StackVector(StackVector&&) = default;
  StackVector& operator=(const StackVector&) = default;
  StackVector& operator=(StackVector&&) = default;
  
  constexpr std::size_t size() const { return N; }
  double& operator[](std::size_t i) { return data_[i]; }
  const double& operator[](std::size_t i) const { return data_[i]; }
  
  auto begin() { return data_.begin(); }
  auto end() { return data_.end(); }
  auto begin() const { return data_.begin(); }
  auto end() const { return data_.end(); }
  
  constexpr static std::size_t static_size() { return N; }
};

//=============================================================================
// CPO implementations for StackVector
//=============================================================================

template<std::size_t N>
StackVector<N> tag_invoke(clone_ftor, const StackVector<N>& v) {
  return StackVector<N>(v);
}

template<std::size_t N>
std::size_t tag_invoke(dimension_ftor, const StackVector<N>& v) {
  return N;
}

template<std::size_t N>
void tag_invoke(add_in_place_ftor, StackVector<N>& x, const StackVector<N>& y) {
  for (std::size_t i = 0; i < N; ++i) {
    x[i] += y[i];
  }
}

template<std::size_t N>
void tag_invoke(scale_in_place_ftor, StackVector<N>& x, double alpha) {
  for (auto& val : x) {
    val *= alpha;
  }
}

template<std::size_t N>
double tag_invoke(inner_product_ftor, const StackVector<N>& x, const StackVector<N>& y) {
  double result = 0.0;
  for (std::size_t i = 0; i < N; ++i) {
    result += x[i] * y[i];
  }
  return result;
}

//=============================================================================
// Runtime sized vector using std::vector (heap allocation)
//=============================================================================

class HeapVector {
private:
  std::vector<double> data_;
  
public:
  explicit HeapVector(std::size_t size) : data_(size, 0.0) {}
  HeapVector(const HeapVector&) = default;
  HeapVector(HeapVector&&) = default;
  HeapVector& operator=(const HeapVector&) = default;
  HeapVector& operator=(HeapVector&&) = default;
  
  std::size_t size() const { return data_.size(); }
  double& operator[](std::size_t i) { return data_[i]; }
  const double& operator[](std::size_t i) const { return data_[i]; }
  
  auto begin() { return data_.begin(); }
  auto end() { return data_.end(); }
  auto begin() const { return data_.begin(); }
  auto end() const { return data_.end(); }
};

//=============================================================================
// CPO implementations for HeapVector
//=============================================================================

HeapVector tag_invoke(clone_ftor, const HeapVector& v) {
  return HeapVector(v);
}

std::size_t tag_invoke(dimension_ftor, const HeapVector& v) {
  return v.size();
}

void tag_invoke(add_in_place_ftor, HeapVector& x, const HeapVector& y) {
  for (std::size_t i = 0; i < x.size() && i < y.size(); ++i) {
    x[i] += y[i];
  }
}

void tag_invoke(scale_in_place_ftor, HeapVector& x, double alpha) {
  for (auto& val : x) {
    val *= alpha;
  }
}

double tag_invoke(inner_product_ftor, const HeapVector& x, const HeapVector& y) {
  double result = 0.0;
  for (std::size_t i = 0; i < x.size() && i < y.size(); ++i) {
    result += x[i] * y[i];
  }
  return result;
}

//=============================================================================
// Preallocated workspace for stack vectors
//=============================================================================

template<std::size_t N, std::size_t PoolSize = 10>
class StackVectorWorkspace {
private:
  std::array<StackVector<N>, PoolSize> workspace_;
  std::array<bool, PoolSize> in_use_;
  
public:
  StackVectorWorkspace() : in_use_{} {}
  
  class WorkspaceVector {
  private:
    StackVectorWorkspace* workspace_;
    std::size_t index_;
    
  public:
    WorkspaceVector(StackVectorWorkspace* ws, std::size_t idx) 
      : workspace_(ws), index_(idx) {}
    
    ~WorkspaceVector() {
      if (workspace_) {
        workspace_->release(index_);
      }
    }
    
    // Move-only semantics
    WorkspaceVector(const WorkspaceVector&) = delete;
    WorkspaceVector& operator=(const WorkspaceVector&) = delete;
    
    WorkspaceVector(WorkspaceVector&& other) noexcept 
      : workspace_(other.workspace_), index_(other.index_) {
      other.workspace_ = nullptr;
    }
    
    WorkspaceVector& operator=(WorkspaceVector&& other) noexcept {
      if (this != &other) {
        if (workspace_) {
          workspace_->release(index_);
        }
        workspace_ = other.workspace_;
        index_ = other.index_;
        other.workspace_ = nullptr;
      }
      return *this;
    }
    
    StackVector<N>& operator*() { return workspace_->workspace_[index_]; }
    const StackVector<N>& operator*() const { return workspace_->workspace_[index_]; }
    
    StackVector<N>* operator->() { return &workspace_->workspace_[index_]; }
    const StackVector<N>* operator->() const { return &workspace_->workspace_[index_]; }
  };
  
  std::optional<WorkspaceVector> acquire() {
    for (std::size_t i = 0; i < PoolSize; ++i) {
      if (!in_use_[i]) {
        in_use_[i] = true;
        return WorkspaceVector(this, i);
      }
    }
    return std::nullopt;  // No available workspace
  }
  
private:
  void release(std::size_t index) {
    if (index < PoolSize) {
      in_use_[index] = false;
    }
  }
  
  friend class WorkspaceVector;
};

//=============================================================================
// Benchmarks for different allocation strategies
//=============================================================================

// Stack allocation benchmarks
template<std::size_t N>
static void BM_StackVector_Clone(benchmark::State& state) {
  StackVector<N> prototype;
  
  for (auto _ : state) {
    auto vec = rvf::clone(prototype);
    benchmark::DoNotOptimize(vec);
  }
  
  state.SetItemsProcessed(state.iterations());
  state.SetLabel(std::to_string(N) + " elements (stack)");
}

// Heap allocation benchmarks
static void BM_HeapVector_Clone(benchmark::State& state) {
  const std::size_t N = state.range(0);
  HeapVector prototype(N);
  
  for (auto _ : state) {
    auto vec = rvf::clone(prototype);
    benchmark::DoNotOptimize(vec);
  }
  
  state.SetItemsProcessed(state.iterations());
  state.SetLabel(std::to_string(N) + " elements (heap)");
}

// Arena allocation benchmarks
static void BM_HeapVector_Arena(benchmark::State& state) {
  const std::size_t N = state.range(0);
  HeapVector prototype(N);
  
  thread_local memory_arena<HeapVector> arena;
  
  for (auto _ : state) {
    auto vec = arena.allocate(prototype);
    benchmark::DoNotOptimize(vec);
  }
  
  state.SetItemsProcessed(state.iterations());
  state.SetLabel(std::to_string(N) + " elements (arena)");
}

// Preallocated stack workspace benchmarks
template<std::size_t N>
static void BM_StackVector_Workspace(benchmark::State& state) {
  thread_local StackVectorWorkspace<N> workspace;
  
  for (auto _ : state) {
    auto vec = workspace.acquire();
    if (vec) {
      benchmark::DoNotOptimize(*vec);
    } else {
      // Fallback to regular allocation if workspace is full
      auto fallback = rvf::clone(StackVector<N>{});
      benchmark::DoNotOptimize(fallback);
    }
  }
  
  state.SetItemsProcessed(state.iterations());
  state.SetLabel(std::to_string(N) + " elements (workspace)");
}

//=============================================================================
// Algorithm pattern comparisons
//=============================================================================

template<std::size_t N>
static void BM_StackVector_AlgorithmPattern(benchmark::State& state) {
  StackVector<N> prototype;
  
  for (auto _ : state) {
    for (int iter = 0; iter < 10; ++iter) {
      auto temp1 = rvf::clone(prototype);
      auto temp2 = rvf::clone(prototype);
      auto temp3 = rvf::clone(prototype);
      
      // Simulate work
      temp1[0] = iter;
      temp2[0] = iter * 2;
      temp3[0] = iter * 3;
      
      benchmark::DoNotOptimize(temp1);
      benchmark::DoNotOptimize(temp2);
      benchmark::DoNotOptimize(temp3);
    }
  }
  
  state.SetItemsProcessed(state.iterations() * 10 * 3);
  state.SetLabel(std::to_string(N) + " elements (stack)");
}

static void BM_HeapVector_AlgorithmPattern(benchmark::State& state) {
  const std::size_t N = state.range(0);
  HeapVector prototype(N);
  
  for (auto _ : state) {
    for (int iter = 0; iter < 10; ++iter) {
      auto temp1 = rvf::clone(prototype);
      auto temp2 = rvf::clone(prototype);
      auto temp3 = rvf::clone(prototype);
      
      // Simulate work
      temp1[0] = iter;
      temp2[0] = iter * 2;
      temp3[0] = iter * 3;
      
      benchmark::DoNotOptimize(temp1);
      benchmark::DoNotOptimize(temp2);
      benchmark::DoNotOptimize(temp3);
    }
  }
  
  state.SetItemsProcessed(state.iterations() * 10 * 3);
  state.SetLabel(std::to_string(N) + " elements (heap)");
}

static void BM_HeapVector_Arena_AlgorithmPattern(benchmark::State& state) {
  const std::size_t N = state.range(0);
  HeapVector prototype(N);
  
  thread_local memory_arena<HeapVector> arena;
  
  for (auto _ : state) {
    for (int iter = 0; iter < 10; ++iter) {
      auto temp1 = arena.allocate(prototype);
      auto temp2 = arena.allocate(prototype);
      auto temp3 = arena.allocate(prototype);
      
      // Simulate work
      (*temp1)[0] = iter;
      (*temp2)[0] = iter * 2;
      (*temp3)[0] = iter * 3;
      
      benchmark::DoNotOptimize(temp1);
      benchmark::DoNotOptimize(temp2);
      benchmark::DoNotOptimize(temp3);
    }
  }
  
  state.SetItemsProcessed(state.iterations() * 10 * 3);
  state.SetLabel(std::to_string(N) + " elements (arena)");
}

template<std::size_t N>
static void BM_StackVector_Workspace_AlgorithmPattern(benchmark::State& state) {
  thread_local StackVectorWorkspace<N> workspace;
  
  for (auto _ : state) {
    for (int iter = 0; iter < 10; ++iter) {
      auto temp1 = workspace.acquire();
      auto temp2 = workspace.acquire();
      auto temp3 = workspace.acquire();
      
      if (temp1 && temp2 && temp3) {
        // Simulate work
        (**temp1)[0] = iter;
        (**temp2)[0] = iter * 2;
        (**temp3)[0] = iter * 3;
        
        benchmark::DoNotOptimize(*temp1);
        benchmark::DoNotOptimize(*temp2);
        benchmark::DoNotOptimize(*temp3);
      }
    }
  }
  
  state.SetItemsProcessed(state.iterations() * 10 * 3);
  state.SetLabel(std::to_string(N) + " elements (workspace)");
}

//=============================================================================
// Register benchmarks
//=============================================================================

// Single allocation benchmarks - small sizes
BENCHMARK(BM_StackVector_Clone<100>);
BENCHMARK(BM_HeapVector_Clone)->Arg(100);
BENCHMARK(BM_HeapVector_Arena)->Arg(100);
BENCHMARK(BM_StackVector_Workspace<100>);

BENCHMARK(BM_StackVector_Clone<1000>);
BENCHMARK(BM_HeapVector_Clone)->Arg(1000);
BENCHMARK(BM_HeapVector_Arena)->Arg(1000);
BENCHMARK(BM_StackVector_Workspace<1000>);

// Medium sizes - only heap and arena (stack would be too large)
BENCHMARK(BM_HeapVector_Clone)->Arg(10000);
BENCHMARK(BM_HeapVector_Arena)->Arg(10000);

BENCHMARK(BM_HeapVector_Clone)->Arg(100000);
BENCHMARK(BM_HeapVector_Arena)->Arg(100000);

// Algorithm pattern benchmarks
BENCHMARK(BM_StackVector_AlgorithmPattern<100>);
BENCHMARK(BM_HeapVector_AlgorithmPattern)->Arg(100);
BENCHMARK(BM_HeapVector_Arena_AlgorithmPattern)->Arg(100);
BENCHMARK(BM_StackVector_Workspace_AlgorithmPattern<100>);

BENCHMARK(BM_StackVector_AlgorithmPattern<1000>);
BENCHMARK(BM_HeapVector_AlgorithmPattern)->Arg(1000);
BENCHMARK(BM_HeapVector_Arena_AlgorithmPattern)->Arg(1000);
BENCHMARK(BM_StackVector_Workspace_AlgorithmPattern<1000>);

BENCHMARK(BM_HeapVector_AlgorithmPattern)->Arg(10000);
BENCHMARK(BM_HeapVector_Arena_AlgorithmPattern)->Arg(10000);

BENCHMARK(BM_HeapVector_AlgorithmPattern)->Arg(100000);
BENCHMARK(BM_HeapVector_Arena_AlgorithmPattern)->Arg(100000);

BENCHMARK_MAIN();
