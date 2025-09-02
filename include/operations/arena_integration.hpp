/**
RealVectorFramework - A Generic Library for Vector Operations and Algorithms

Copyright (c) National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

Questions? Contact Greg von Winckel (gvonwin@sandia.gov)
*/

#pragma once

#include "memory_arena.hpp"
#include "arena_observers.hpp"
#include "clone.hpp"
#include <memory>
#include <utility>

namespace rvf {

/**
 * @brief RAII class for automatic arena configuration
 */
template<typename V>
class arena_manager {
private:
  memory_arena<V>* arena_;
  std::shared_ptr<statistics_observer<V>> stats_observer_;
  std::shared_ptr<compaction_observer<V>> compaction_observer_;
  std::shared_ptr<logging_observer<V>> logging_observer_;
  
public:
  struct config {
    bool enable_statistics = true;
    bool enable_compaction = true;
    bool enable_logging = false;
    std::size_t compaction_threshold = 1000;
    std::chrono::steady_clock::duration compaction_interval = std::chrono::minutes(5);
    std::ostream* log_stream = &std::cout;
  };
  
  explicit arena_manager(memory_arena<V>& arena, const config& cfg = {})
    : arena_(&arena) {
    
    if (cfg.enable_statistics) {
      stats_observer_ = add_statistics_observer(*arena_);
    }
    
    if (cfg.enable_compaction) {
      compaction_observer_ = add_compaction_observer(*arena_, cfg.compaction_threshold);
    }
    
    if (cfg.enable_logging) {
      logging_observer_ = add_logging_observer(*arena_, *cfg.log_stream);
    }
  }
  
  ~arena_manager() = default;
  
  // Non-copyable, movable
  arena_manager(const arena_manager&) = delete;
  arena_manager& operator=(const arena_manager&) = delete;
  arena_manager(arena_manager&&) = default;
  arena_manager& operator=(arena_manager&&) = default;
  
  memory_arena<V>& get_arena() { return *arena_; }
  const memory_arena<V>& get_arena() const { return *arena_; }
  
  auto get_statistics() const -> decltype(stats_observer_->get_statistics()) {
    if (!stats_observer_) {
      throw std::runtime_error("Statistics not enabled for this arena manager");
    }
    return stats_observer_->get_statistics();
  }
  
  void reset_statistics() {
    if (stats_observer_) {
      stats_observer_->reset();
    }
  }
};

/**
 * @brief Scoped arena allocator that automatically returns vectors
 */
template<typename V>
class scoped_arena_allocator {
private:
  memory_arena<V>* arena_;
  std::vector<arena_vector<V>> allocated_vectors_;
  
public:
  explicit scoped_arena_allocator(memory_arena<V>& arena) : arena_(&arena) {}
  
  ~scoped_arena_allocator() {
    // Vectors are automatically returned when arena_vector destructors are called
    allocated_vectors_.clear();
  }
  
  // Non-copyable, movable
  scoped_arena_allocator(const scoped_arena_allocator&) = delete;
  scoped_arena_allocator& operator=(const scoped_arena_allocator&) = delete;
  scoped_arena_allocator(scoped_arena_allocator&&) = default;
  scoped_arena_allocator& operator=(scoped_arena_allocator&&) = default;
  
  template<typename VectorType>
    requires std::same_as<std::decay_t<VectorType>, V>
  V& allocate(const VectorType& prototype) {
    allocated_vectors_.push_back(arena_->allocate(prototype));
    return *allocated_vectors_.back();
  }
  
  std::size_t allocated_count() const {
    return allocated_vectors_.size();
  }
  
  void clear() {
    allocated_vectors_.clear();
  }
};

/**
 * @brief Integration with existing clone CPO using type trait dispatch
 */
namespace detail {
  
template<typename V>
struct use_arena : std::false_type {};

// Specialization to enable arena for specific types
// Users can specialize this for their types
template<>
struct use_arena<std::vector<double>> : std::true_type {};

} // namespace detail

/**
 * @brief Enhanced clone that can optionally use arena allocation
 */
inline constexpr struct enhanced_clone_ftor final : tincup::cpo_base<enhanced_clone_ftor> {
  TINCUP_CPO_TAG("enhanced_clone")
  inline static constexpr bool is_variadic = false;
  
  template<typename V>
    requires clone_invocable_c<V>
  constexpr auto operator()(const V& x) const
    noexcept(clone_nothrow_invocable_c<V>) {
    
    if constexpr (detail::use_arena<V>::value) {
      // Use arena allocation
      return arena_clone(x);
    } else {
      // Use regular clone
      return clone(x);
    }
  }
} enhanced_clone;

/**
 * @brief Context class for algorithm implementations that need temporary vectors
 */
template<typename V>
class algorithm_context {
private:
  scoped_arena_allocator<V> allocator_;
  
public:
  explicit algorithm_context(memory_arena<V>& arena) : allocator_(arena) {}
  
  template<typename VectorType>
    requires std::same_as<std::decay_t<VectorType>, V>
  V& get_temporary(const VectorType& prototype) {
    return allocator_.allocate(prototype);
  }
  
  std::size_t temporary_count() const {
    return allocator_.allocated_count();
  }
};

/**
 * @brief Factory function to create a fully configured arena
 */
template<typename V>
auto create_configured_arena(const typename arena_manager<V>::config& cfg = {}) {
  auto arena = std::make_unique<memory_arena<V>>();
  auto manager = std::make_unique<arena_manager<V>>(*arena, cfg);
  
  return std::make_pair(std::move(arena), std::move(manager));
}

/**
 * @brief Thread-local arena accessor for high-performance scenarios
 */
template<typename V>
memory_arena<V>& get_thread_local_arena() {
  thread_local memory_arena<V> arena;
  return arena;
}

/**
 * @brief Utility to benchmark arena vs regular allocation
 */
template<typename V>
struct allocation_benchmark_result {
  std::chrono::nanoseconds regular_clone_time;
  std::chrono::nanoseconds arena_clone_time;
  double speedup_factor;
  std::size_t iterations;
};

template<typename V>
allocation_benchmark_result<V> benchmark_allocation(const V& prototype, 
                           std::size_t iterations = 10000) {
  using clock = std::chrono::high_resolution_clock;
  
  // Benchmark regular clone
  auto start = clock::now();
  {
    std::vector<clone_return_t<V>> vectors;
    vectors.reserve(iterations);
    
    for (std::size_t i = 0; i < iterations; ++i) {
      vectors.push_back(clone(prototype));
    }
  }
  auto regular_time = clock::now() - start;
  
  // Benchmark arena allocation
  start = clock::now();
  {
    auto& arena = get_global_arena<V>();
    std::vector<arena_vector<V>> vectors;
    vectors.reserve(iterations);
    
    for (std::size_t i = 0; i < iterations; ++i) {
      vectors.push_back(arena.allocate(prototype));
    }
  }
  auto arena_time = clock::now() - start;
  
  return {
    std::chrono::duration_cast<std::chrono::nanoseconds>(regular_time),
    std::chrono::duration_cast<std::chrono::nanoseconds>(arena_time),
    static_cast<double>(regular_time.count()) / arena_time.count(),
    iterations
  };
}

/**
 * @brief Utility to print detailed arena status
 */
template<typename V>
void print_arena_status(const memory_arena<V>& arena, std::ostream& os = std::cout) {
  auto stats = arena.get_statistics();
  auto pool_stats = arena.get_pool_statistics();
  
  os << "=== Arena Status ===\n";
  os << "Uptime: " << std::chrono::duration<double>(stats.uptime()).count() << " seconds\n";
  os << "Total allocations: " << stats.total_vectors_allocated << "\n";
  os << "Current in use: " << stats.total_vectors_in_use << "\n";
  os << "Peak usage: " << stats.peak_memory_usage << "\n";
  os << "Number of pools: " << stats.total_pools << "\n";
  
  os << "\n--- Pool Details ---\n";
  for (const auto& pool : pool_stats) {
    os << "Pool (type=" << pool.key.type_id.name() 
       << ", dim=" << pool.key.dimension 
       << ", align=" << pool.key.alignment << "):\n";
    os << "  Available: " << pool.available_count << "\n";
    os << "  Total allocated: " << pool.total_allocated << "\n";
    os << "  Peak size: " << pool.peak_size << "\n";
  }
  os << "==================\n";
}

} // namespace rvf
