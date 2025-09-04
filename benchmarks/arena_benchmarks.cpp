/**
 * @file arena_benchmarks.cpp
 * @brief Google Benchmark tests for RVF memory arena performance
 */

#include <core/rvf.hpp>
#include <operations/memory/memory_arena.hpp>
#include <operations/memory/arena_observers.hpp>
#include <benchmark/benchmark.h>
#include <vector>
#include <memory>

using namespace rvf;

// Test vector class - simple std::vector<double> wrapper
class BenchVector {
private:
  std::vector<double> data_;
  
public:
  explicit BenchVector(std::size_t size) : data_(size, 0.0) {}
  BenchVector(const BenchVector&) = default;
  BenchVector(BenchVector&&) = default;
  BenchVector& operator=(const BenchVector&) = default;
  BenchVector& operator=(BenchVector&&) = default;
  
  std::size_t size() const { return data_.size(); }
  double& operator[](std::size_t i) { return data_[i]; }
  const double& operator[](std::size_t i) const { return data_[i]; }
  
  auto begin() { return data_.begin(); }
  auto end() { return data_.end(); }
  auto begin() const { return data_.begin(); }
  auto end() const { return data_.end(); }
};

// Implement required CPO overloads for BenchVector
namespace {
  
BenchVector tag_invoke(clone_ftor, const BenchVector& v) {
  return BenchVector(v);
}

std::size_t tag_invoke(dimension_ftor, const BenchVector& v) {
  return v.size();
}

void tag_invoke(add_in_place_ftor, BenchVector& x, const BenchVector& y) {
  for (std::size_t i = 0; i < x.size() && i < y.size(); ++i) {
    x[i] += y[i];
  }
}

void tag_invoke(scale_in_place_ftor, BenchVector& x, double alpha) {
  for (auto& val : x) {
    val *= alpha;
  }
}

double tag_invoke(inner_product_ftor, const BenchVector& x, const BenchVector& y) {
  double result = 0.0;
  for (std::size_t i = 0; i < x.size() && i < y.size(); ++i) {
    result += x[i] * y[i];
  }
  return result;
}

} // anonymous namespace

// Global arena for benchmarking
thread_local memory_arena<BenchVector> bench_arena;

//=============================================================================
// Single allocation benchmarks
//=============================================================================

static void BM_RegularClone_Small(benchmark::State& state) {
  BenchVector prototype(100);
  
  for (auto _ : state) {
    auto vec = rvf::clone(prototype);
    benchmark::DoNotOptimize(vec);
  }
  
  state.SetItemsProcessed(state.iterations());
  state.SetLabel("100 elements");
}

static void BM_ArenaClone_Small(benchmark::State& state) {
  BenchVector prototype(100);
  
  for (auto _ : state) {
    auto vec = bench_arena.allocate(prototype);
    benchmark::DoNotOptimize(vec);
  }
  
  state.SetItemsProcessed(state.iterations());
  state.SetLabel("100 elements");
}

static void BM_RegularClone_Medium(benchmark::State& state) {
  BenchVector prototype(10000);
  
  for (auto _ : state) {
    auto vec = rvf::clone(prototype);
    benchmark::DoNotOptimize(vec);
  }
  
  state.SetItemsProcessed(state.iterations());
  state.SetLabel("10K elements");
}

static void BM_ArenaClone_Medium(benchmark::State& state) {
  BenchVector prototype(10000);
  
  for (auto _ : state) {
    auto vec = bench_arena.allocate(prototype);
    benchmark::DoNotOptimize(vec);
  }
  
  state.SetItemsProcessed(state.iterations());
  state.SetLabel("10K elements");
}

static void BM_RegularClone_Large(benchmark::State& state) {
  BenchVector prototype(1000000);
  
  for (auto _ : state) {
    auto vec = rvf::clone(prototype);
    benchmark::DoNotOptimize(vec);
  }
  
  state.SetItemsProcessed(state.iterations());
  state.SetLabel("1M elements");
}

static void BM_ArenaClone_Large(benchmark::State& state) {
  BenchVector prototype(1000000);
  
  for (auto _ : state) {
    auto vec = bench_arena.allocate(prototype);
    benchmark::DoNotOptimize(vec);
  }
  
  state.SetItemsProcessed(state.iterations());
  state.SetLabel("1M elements");
}

//=============================================================================
// Batch allocation benchmarks
//=============================================================================

static void BM_RegularClone_Batch(benchmark::State& state) {
  const std::size_t vector_size = state.range(0);
  const std::size_t batch_size = state.range(1);
  
  BenchVector prototype(vector_size);
  
  for (auto _ : state) {
    std::vector<clone_return_t<BenchVector>> vectors;
    vectors.reserve(batch_size);
    
    for (std::size_t i = 0; i < batch_size; ++i) {
      vectors.push_back(rvf::clone(prototype));
    }
    
    benchmark::DoNotOptimize(vectors);
  }
  
  state.SetItemsProcessed(state.iterations() * batch_size);
  state.SetLabel(std::to_string(vector_size) + " elem, batch " + std::to_string(batch_size));
}

static void BM_ArenaClone_Batch(benchmark::State& state) {
  const std::size_t vector_size = state.range(0);
  const std::size_t batch_size = state.range(1);
  
  BenchVector prototype(vector_size);
  
  for (auto _ : state) {
    std::vector<arena_vector<BenchVector>> vectors;
    vectors.reserve(batch_size);
    
    for (std::size_t i = 0; i < batch_size; ++i) {
      vectors.push_back(bench_arena.allocate(prototype));
    }
    
    benchmark::DoNotOptimize(vectors);
  }
  
  state.SetItemsProcessed(state.iterations() * batch_size);
  state.SetLabel(std::to_string(vector_size) + " elem, batch " + std::to_string(batch_size));
}

//=============================================================================
// Reuse pattern benchmarks (realistic algorithm scenarios)
//=============================================================================

static void BM_RegularClone_AlgorithmPattern(benchmark::State& state) {
  const std::size_t vector_size = state.range(0);
  BenchVector prototype(vector_size);
  
  for (auto _ : state) {
    // Simulate algorithm needing temporary vectors
    for (int iter = 0; iter < 10; ++iter) {
      auto temp1 = rvf::clone(prototype);
      auto temp2 = rvf::clone(prototype);
      auto temp3 = rvf::clone(prototype);
      
      // Simulate some work
      temp1[0] = iter;
      temp2[0] = iter * 2;
      temp3[0] = iter * 3;
      
      benchmark::DoNotOptimize(temp1);
      benchmark::DoNotOptimize(temp2);
      benchmark::DoNotOptimize(temp3);
    }
  }
  
  state.SetItemsProcessed(state.iterations() * 10 * 3);
  state.SetLabel(std::to_string(vector_size) + " elements");
}

static void BM_ArenaClone_AlgorithmPattern(benchmark::State& state) {
  const std::size_t vector_size = state.range(0);
  BenchVector prototype(vector_size);
  
  for (auto _ : state) {
    // Simulate algorithm needing temporary vectors
    for (int iter = 0; iter < 10; ++iter) {
      auto temp1 = bench_arena.allocate(prototype);
      auto temp2 = bench_arena.allocate(prototype);
      auto temp3 = bench_arena.allocate(prototype);
      
      // Simulate some work
      (*temp1)[0] = iter;
      (*temp2)[0] = iter * 2;
      (*temp3)[0] = iter * 3;
      
      benchmark::DoNotOptimize(temp1);
      benchmark::DoNotOptimize(temp2);
      benchmark::DoNotOptimize(temp3);
    }
  }
  
  state.SetItemsProcessed(state.iterations() * 10 * 3);
  state.SetLabel(std::to_string(vector_size) + " elements");
}

//=============================================================================
// Arena overhead benchmarks
//=============================================================================

static void BM_Arena_Statistics(benchmark::State& state) {
  for (auto _ : state) {
    auto stats = bench_arena.get_statistics();
    benchmark::DoNotOptimize(stats);
  }
}

static void BM_Arena_PoolStats(benchmark::State& state) {
  for (auto _ : state) {
    auto pool_stats = bench_arena.get_pool_statistics();
    benchmark::DoNotOptimize(pool_stats);
  }
}

//=============================================================================
// Observer overhead benchmarks
//=============================================================================

static void BM_ArenaWithObservers_Clone(benchmark::State& state) {
  memory_arena<BenchVector> arena_with_observers;
  
  // Add observers
  auto stats_obs = add_statistics_observer(arena_with_observers);
  auto compaction_obs = add_compaction_observer(arena_with_observers, 1000);
  
  BenchVector prototype(1000);
  
  for (auto _ : state) {
    auto vec = arena_with_observers.allocate(prototype);
    benchmark::DoNotOptimize(vec);
  }
  
  state.SetItemsProcessed(state.iterations());
}

//=============================================================================
// Memory pressure benchmarks
//=============================================================================

static void BM_Arena_MemoryPressure(benchmark::State& state) {
  const std::size_t concurrent_vectors = state.range(0);
  BenchVector prototype(10000);
  
  for (auto _ : state) {
    std::vector<arena_vector<BenchVector>> vectors;
    vectors.reserve(concurrent_vectors);
    
    // Allocate many vectors simultaneously
    for (std::size_t i = 0; i < concurrent_vectors; ++i) {
      vectors.push_back(bench_arena.allocate(prototype));
    }
    
    // Access them all to ensure they stay alive
    for (auto& vec : vectors) {
      (*vec)[0] = 42.0;
      benchmark::DoNotOptimize(vec);
    }
  }
  
  state.SetItemsProcessed(state.iterations() * concurrent_vectors);
}

//=============================================================================
// Register benchmarks
//=============================================================================

// Single allocation benchmarks
BENCHMARK(BM_RegularClone_Small);
BENCHMARK(BM_ArenaClone_Small);
BENCHMARK(BM_RegularClone_Medium);
BENCHMARK(BM_ArenaClone_Medium);
BENCHMARK(BM_RegularClone_Large);
BENCHMARK(BM_ArenaClone_Large);

// Batch allocation benchmarks
BENCHMARK(BM_RegularClone_Batch)
  ->Args({1000, 10})
  ->Args({1000, 100})
  ->Args({1000, 1000})
  ->Args({10000, 10})
  ->Args({10000, 100})
  ->Args({100000, 10});

BENCHMARK(BM_ArenaClone_Batch)
  ->Args({1000, 10})
  ->Args({1000, 100})
  ->Args({1000, 1000})
  ->Args({10000, 10})
  ->Args({10000, 100})
  ->Args({100000, 10});

// Algorithm pattern benchmarks
BENCHMARK(BM_RegularClone_AlgorithmPattern)
  ->Arg(1000)
  ->Arg(10000)
  ->Arg(100000);

BENCHMARK(BM_ArenaClone_AlgorithmPattern)
  ->Arg(1000)
  ->Arg(10000)
  ->Arg(100000);

// Arena overhead benchmarks
BENCHMARK(BM_Arena_Statistics);
BENCHMARK(BM_Arena_PoolStats);
BENCHMARK(BM_ArenaWithObservers_Clone);

// Memory pressure benchmarks
BENCHMARK(BM_Arena_MemoryPressure)
  ->Arg(10)
  ->Arg(100)
  ->Arg(1000)
  ->Arg(10000);

// Configure benchmark settings
BENCHMARK_MAIN();
