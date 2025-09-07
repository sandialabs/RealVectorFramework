/**
RealVectorFramework - A Generic Library for Vector Operations and Algorithms

Copyright (c) National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

Questions? Contact Greg von Winckel (gvonwin@sandia.gov)
*/

/**
 * @file memory_arena_example.cpp
 * @brief Example usage of the RVF memory arena system
 */

#include <core/rvf.hpp>
#include <operations/memory/memory_arena.hpp>
#include <operations/memory/arena_observers.hpp>
#include <vector>
#include <iostream>
#include <chrono>
#include <thread>

using namespace rvf;

// Example vector type - simple std::vector<double> wrapper
class SimpleVector {
private:
  std::vector<double> data_;
  
public:
  explicit SimpleVector(std::size_t size) : data_(size, 0.0) {}
  SimpleVector(const SimpleVector&) = default;
  SimpleVector(SimpleVector&&) = default;
  SimpleVector& operator=(const SimpleVector&) = default;
  SimpleVector& operator=(SimpleVector&&) = default;
  
  std::size_t size() const { return data_.size(); }
  double& operator[](std::size_t i) { return data_[i]; }
  const double& operator[](std::size_t i) const { return data_[i]; }
  
  auto begin() { return data_.begin(); }
  auto end() { return data_.end(); }
  auto begin() const { return data_.begin(); }
  auto end() const { return data_.end(); }
};

// Implement required CPO overloads for SimpleVector
namespace {
  
// Clone support
SimpleVector tag_invoke(clone_ftor, const SimpleVector& v) {
  return SimpleVector(v);
}

// Dimension support  
std::size_t tag_invoke(dimension_ftor, const SimpleVector& v) {
  return v.size();
}

// Add in place support
void tag_invoke(add_in_place_ftor, SimpleVector& x, const SimpleVector& y) {
  for (std::size_t i = 0; i < x.size() && i < y.size(); ++i) {
    x[i] += y[i];
  }
}

// Scale in place support
void tag_invoke(scale_in_place_ftor, SimpleVector& x, double alpha) {
  for (auto& val : x) {
    val *= alpha;
  }
}

// Inner product support
double tag_invoke(inner_product_ftor, const SimpleVector& x, const SimpleVector& y) {
  double result = 0.0;
  for (std::size_t i = 0; i < x.size() && i < y.size(); ++i) {
    result += x[i] * y[i];
  }
  return result;
}

} // anonymous namespace

void basic_arena_usage() {
  std::cout << "\n=== Basic Arena Usage ===\n";
  
  // Get the global arena for SimpleVector
  auto& arena = get_global_arena<SimpleVector>();
  
  // Create a prototype vector
  SimpleVector prototype(1000);
  for (std::size_t i = 0; i < prototype.size(); ++i) {
    prototype[i] = static_cast<double>(i);
  }
  
  // Allocate vectors from arena
  {
    auto vec1 = arena.allocate(prototype);
    auto vec2 = arena.allocate(prototype);
    auto vec3 = arena.allocate(prototype);
    
    std::cout << "Allocated 3 vectors from arena\n";
    std::cout << "Arena stats: " << arena.get_statistics().total_vectors_in_use 
          << " vectors in use\n";
    
    // Use the vectors
    (*vec1)[0] = 100.0;
    (*vec2)[0] = 200.0;
    (*vec3)[0] = 300.0;
    
    std::cout << "Vector values: " << (*vec1)[0] << ", " << (*vec2)[0] 
          << ", " << (*vec3)[0] << "\n";
    
  } // vectors automatically returned to arena when destructed
  
  std::cout << "After vectors go out of scope, in use: " 
        << arena.get_statistics().total_vectors_in_use << "\n";
}

void observer_pattern_demo() {
  std::cout << "\n=== Observer Pattern Demo ===\n";
  
  auto& arena = get_global_arena<SimpleVector>();
  
  // Add various observers
  auto logger = add_logging_observer(arena);
  auto stats_observer = add_statistics_observer(arena);
  auto leak_detector = add_leak_detector(arena, 5); // Low threshold for demo
  
  SimpleVector prototype(500);
  
  std::cout << "Allocating vectors with observers active...\n";
  
  // Allocate some vectors
  std::vector<arena_vector<SimpleVector>> vectors;
  for (int i = 0; i < 10; ++i) {
    vectors.push_back(arena.allocate(prototype));
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  
  // Get statistics
  auto stats = stats_observer->get_statistics();
  std::cout << "\nObserver Statistics:\n";
  std::cout << "  Total allocations: " << stats.total_allocations << "\n";
  std::cout << "  Current allocations: " << stats.current_allocations << "\n";
  std::cout << "  Peak allocations: " << stats.peak_allocations << "\n";
  std::cout << "  Allocation rate: " << stats.allocation_rate() << " allocs/sec\n";
  std::cout << "  Unreturned vectors: " << leak_detector->get_unreturned_count() << "\n";
}

void pool_management_demo() {
  std::cout << "\n=== Pool Management Demo ===\n";
  
  auto& arena = get_global_arena<SimpleVector>();
  
  // Create vectors of different sizes to create multiple pools
  SimpleVector small_vec(100);
  SimpleVector medium_vec(1000);
  SimpleVector large_vec(10000);
  
  {
    auto v1 = arena.allocate(small_vec);
    auto v2 = arena.allocate(medium_vec);
    auto v3 = arena.allocate(large_vec);
    auto v4 = arena.allocate(small_vec);  // Reuse small pool
    
    std::cout << "Allocated vectors of different sizes\n";
    
    // Get pool statistics
    auto pool_stats = arena.get_pool_statistics();
    std::cout << "Number of pools: " << pool_stats.size() << "\n";
    
    for (const auto& stats : pool_stats) {
      std::cout << "  Pool dimension " << stats.key.dimension 
            << ": " << stats.available_count << " available, "
            << stats.total_allocated << " total allocated\n";
    }
  }
  
  // Compact arena to remove unused pools
  std::cout << "\nCompacting arena...\n";
  arena.compact();
  
  auto final_stats = arena.get_pool_statistics();
  std::cout << "Pools after compaction: " << final_stats.size() << "\n";
}

void performance_comparison() {
  std::cout << "\n=== Performance Comparison ===\n";
  
  const std::size_t num_iterations = 10000;
  const std::size_t vector_size = 1000;
  
  SimpleVector prototype(vector_size);
  
  // Test regular clone performance
  auto start = std::chrono::high_resolution_clock::now();
  {
    std::vector<SimpleVector> vectors;
    for (std::size_t i = 0; i < num_iterations; ++i) {
      vectors.push_back(rvf::clone(prototype));
    }
  }
  auto regular_time = std::chrono::high_resolution_clock::now() - start;
  
  // Test arena performance
  start = std::chrono::high_resolution_clock::now();
  {
    auto& arena = get_global_arena<SimpleVector>();
    std::vector<arena_vector<SimpleVector>> vectors;
    for (std::size_t i = 0; i < num_iterations; ++i) {
      vectors.push_back(arena.allocate(prototype));
    }
  }
  auto arena_time = std::chrono::high_resolution_clock::now() - start;
  
  std::cout << "Regular clone time: " 
        << std::chrono::duration<double, std::milli>(regular_time).count() 
        << " ms\n";
  std::cout << "Arena allocation time: " 
        << std::chrono::duration<double, std::milli>(arena_time).count() 
        << " ms\n";
  std::cout << "Speedup factor: " 
        << static_cast<double>(regular_time.count()) / arena_time.count() 
        << "x\n";
}

void arena_cpo_demo() {
  std::cout << "\n=== Arena CPO Demo ===\n";
  
  SimpleVector prototype(200);
  for (std::size_t i = 0; i < prototype.size(); ++i) {
    prototype[i] = static_cast<double>(i);
  }
  
  // Use the arena_clone CPO directly
  auto arena_vec = arena_clone(prototype);
  
  std::cout << "Created vector using arena_clone CPO\n";
  std::cout << "Vector size: " << arena_vec->size() << "\n";
  std::cout << "First few elements: ";
  for (std::size_t i = 0; i < 5; ++i) {
    std::cout << (*arena_vec)[i] << " ";
  }
  std::cout << "\n";
  
  // Modify the vector
  (*arena_vec)[0] = 999.0;
  std::cout << "After modification, first element: " << (*arena_vec)[0] << "\n";
  
  // Vector is automatically returned to arena when arena_vec goes out of scope
}

int main() {
  std::cout << "RVF Memory Arena Demo\n";
  std::cout << "====================\n";
  
  try {
    basic_arena_usage();
    observer_pattern_demo();
    pool_management_demo();
    performance_comparison();
    arena_cpo_demo();
    
    std::cout << "\n=== Final Arena Statistics ===\n";
    auto& arena = get_global_arena<SimpleVector>();
    auto stats = arena.get_statistics();
    std::cout << "Total vectors allocated: " << stats.total_vectors_allocated << "\n";
    std::cout << "Current vectors in use: " << stats.total_vectors_in_use << "\n";
    std::cout << "Total pools: " << stats.total_pools << "\n";
    std::cout << "Arena uptime: " 
          << std::chrono::duration<double>(stats.uptime()).count() 
          << " seconds\n";
    
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  
  return 0;
}
