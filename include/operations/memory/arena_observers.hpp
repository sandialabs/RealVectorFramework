/**
RealVectorFramework - A Generic Library for Vector Operations and Algorithms

Copyright (c) National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

Questions? Contact Greg von Winckel (gvonwin@sandia.gov)
*/

#pragma once

#include "memory_arena.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <unordered_set>

namespace rvf {

/**
 * @brief Observer that logs arena events to a stream
 */
template<typename V>
class logging_observer : public arena_observer<V> {
private:
  std::ostream& stream_;
  bool log_allocations_;
  bool log_returns_;
  bool log_pool_events_;
  
  std::string timestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    return ss.str();
  }
  
public:
  explicit logging_observer(std::ostream& stream = std::cout,
                            bool log_allocations = true,
                            bool log_returns = false,
                            bool log_pool_events = true)
    : stream_(stream), log_allocations_(log_allocations), 
      log_returns_(log_returns), log_pool_events_(log_pool_events) {}
  
  void on_event(arena_event event, const V* vector, std::size_t pool_size) override {
    switch (event) {
      case arena_event::vector_allocated:
        if (log_allocations_) {
          stream_ << "[" << timestamp() << "] ALLOC: Vector allocated at " 
               << vector << std::endl;
        }
        break;
      
      case arena_event::vector_returned:
        if (log_returns_) {
          stream_ << "[" << timestamp() << "] RETURN: Vector returned at " 
               << vector << std::endl;
        }
        break;
      
      case arena_event::pool_created:
        if (log_pool_events_) {
          stream_ << "[" << timestamp() << "] POOL_CREATE: New pool created" 
               << std::endl;
        }
        break;
      
      case arena_event::pool_destroyed:
        if (log_pool_events_) {
          stream_ << "[" << timestamp() << "] POOL_DESTROY: Pool destroyed" 
               << std::endl;
        }
        break;
      
      case arena_event::arena_cleared:
        stream_ << "[" << timestamp() << "] CLEAR: Arena cleared" << std::endl;
        break;
    }
  }
};

/**
 * @brief Observer that tracks memory usage statistics
 */
template<typename V>
class statistics_observer : public arena_observer<V> {
private:
  std::atomic<std::size_t> total_allocations_{0};
  std::atomic<std::size_t> current_allocations_{0};
  std::atomic<std::size_t> peak_allocations_{0};
  std::atomic<std::size_t> pool_creations_{0};
  std::atomic<std::size_t> pool_destructions_{0};
  
  std::chrono::steady_clock::time_point start_time_{std::chrono::steady_clock::now()};
  
public:
  struct statistics {
    std::size_t total_allocations;
    std::size_t current_allocations;
    std::size_t peak_allocations;
    std::size_t pool_creations;
    std::size_t pool_destructions;
    std::chrono::steady_clock::duration uptime;
    
    double allocation_rate() const {
      auto seconds = std::chrono::duration<double>(uptime).count();
      return seconds > 0 ? static_cast<double>(total_allocations) / seconds : 0.0;
    }
  };
  
  void on_event(arena_event event, const V* vector, std::size_t pool_size) override {
    switch (event) {
      case arena_event::vector_allocated: {
        total_allocations_.fetch_add(1, std::memory_order_relaxed);
        std::size_t current = current_allocations_.fetch_add(1, std::memory_order_relaxed) + 1;
        
        // Update peak if necessary
        std::size_t current_peak = peak_allocations_.load(std::memory_order_relaxed);
        while (current > current_peak && 
             !peak_allocations_.compare_exchange_weak(current_peak, current, 
                                 std::memory_order_relaxed)) {
        }
        break;
      }
      
      case arena_event::vector_returned:
        current_allocations_.fetch_sub(1, std::memory_order_relaxed);
        break;
      
      case arena_event::pool_created:
        pool_creations_.fetch_add(1, std::memory_order_relaxed);
        break;
      
      case arena_event::pool_destroyed:
        pool_destructions_.fetch_add(1, std::memory_order_relaxed);
        break;
      
      case arena_event::arena_cleared:
        current_allocations_.store(0, std::memory_order_relaxed);
        break;
    }
  }
  
  statistics get_statistics() const {
    return {
      total_allocations_.load(),
      current_allocations_.load(),
      peak_allocations_.load(),
      pool_creations_.load(),
      pool_destructions_.load(),
      std::chrono::steady_clock::now() - start_time_
    };
  }
  
  void reset() {
    total_allocations_.store(0);
    current_allocations_.store(0);
    peak_allocations_.store(0);
    pool_creations_.store(0);
    pool_destructions_.store(0);
    start_time_ = std::chrono::steady_clock::now();
  }
};

/**
 * @brief Observer that triggers memory compaction based on thresholds
 */
template<typename V>
class compaction_observer : public arena_observer<V> {
private:
  memory_arena<V>* arena_;
  std::size_t allocation_threshold_;
  std::size_t allocations_since_compaction_{0};
  std::chrono::steady_clock::time_point last_compaction_{std::chrono::steady_clock::now()};
  std::chrono::steady_clock::duration min_interval_;
  
public:
  compaction_observer(memory_arena<V>* arena, 
             std::size_t allocation_threshold = 1000,
             std::chrono::steady_clock::duration min_interval = std::chrono::minutes(5))
    : arena_(arena), allocation_threshold_(allocation_threshold), min_interval_(min_interval) {}
  
  void on_event(arena_event event, const V* vector, std::size_t pool_size) override {
    if (event == arena_event::vector_allocated) {
      ++allocations_since_compaction_;
      
      auto now = std::chrono::steady_clock::now();
      bool threshold_reached = allocations_since_compaction_ >= allocation_threshold_;
      bool interval_passed = (now - last_compaction_) >= min_interval_;
      
      if (threshold_reached && interval_passed && arena_) {
        arena_->compact();
        allocations_since_compaction_ = 0;
        last_compaction_ = now;
      }
    }
  }
};

/**
 * @brief Observer that detects potential memory leaks
 */
template<typename V>
class leak_detector : public arena_observer<V> {
private:
  std::unordered_set<const V*> allocated_vectors_;
  std::mutex vectors_mutex_;
  std::size_t leak_threshold_;
  
public:
  explicit leak_detector(std::size_t leak_threshold = 10000) 
    : leak_threshold_(leak_threshold) {}
  
  void on_event(arena_event event, const V* vector, std::size_t pool_size) override {
    std::lock_guard<std::mutex> lock(vectors_mutex_);
    
    switch (event) {
      case arena_event::vector_allocated:
        allocated_vectors_.insert(vector);
        if (allocated_vectors_.size() > leak_threshold_) {
          std::cerr << "WARNING: Potential memory leak detected. "
               << allocated_vectors_.size() << " vectors allocated without return."
               << std::endl;
        }
        break;
      
      case arena_event::vector_returned:
        allocated_vectors_.erase(vector);
        break;
      
      case arena_event::arena_cleared:
        allocated_vectors_.clear();
        break;
        
      default:
        break;
    }
  }
  
  std::size_t get_unreturned_count() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(vectors_mutex_));
    return allocated_vectors_.size();
  }
};

/**
 * @brief Convenience function to create and register a logging observer
 */
template<typename V>
std::shared_ptr<logging_observer<V>> add_logging_observer(memory_arena<V>& arena, 
                             std::ostream& stream = std::cout) {
  auto observer = std::make_shared<logging_observer<V>>(stream);
  arena.register_observer(observer);
  return observer;
}

/**
 * @brief Convenience function to create and register a statistics observer
 */
template<typename V>
std::shared_ptr<statistics_observer<V>> add_statistics_observer(memory_arena<V>& arena) {
  auto observer = std::make_shared<statistics_observer<V>>();
  arena.register_observer(observer);
  return observer;
}

/**
 * @brief Convenience function to create and register a compaction observer
 */
template<typename V>
std::shared_ptr<compaction_observer<V>> add_compaction_observer(memory_arena<V>& arena,
                                std::size_t threshold = 1000) {
  auto observer = std::make_shared<compaction_observer<V>>(&arena, threshold);
  arena.register_observer(observer);
  return observer;
}

/**
 * @brief Convenience function to create and register a leak detector
 */
template<typename V>
std::shared_ptr<leak_detector<V>> add_leak_detector(memory_arena<V>& arena,
                           std::size_t threshold = 10000) {
  auto observer = std::make_shared<leak_detector<V>>(threshold);
  arena.register_observer(observer);
  return observer;
}

} // namespace rvf
