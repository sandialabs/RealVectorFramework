/**
RealVectorFramework - A Generic Library for Vector Operations and Algorithms

Copyright (c) National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

Questions? Contact Greg von Winckel (gvonwin@sandia.gov)
*/

#pragma once

#include <tincup/tincup.hpp>
#include <memory>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <functional>
#include <typeindex>
#include <atomic>
#include <chrono>
#include "clone.hpp"
#include "dimension.hpp"

namespace rvf {

// Forward declarations
template<typename V> class vector_pool;
template<typename V> class arena_observer;
template<typename V> class memory_arena;

/**
 * @brief Event types for arena observer pattern
 */
enum class arena_event {
    vector_allocated,
    vector_returned,
    pool_created,
    pool_destroyed,
    arena_cleared
};

/**
 * @brief Statistics for memory arena monitoring
 */
struct arena_statistics {
    std::size_t total_vectors_allocated{0};
    std::size_t total_vectors_in_use{0};
    std::size_t total_pools{0};
    std::size_t peak_memory_usage{0};
    std::chrono::steady_clock::time_point creation_time{std::chrono::steady_clock::now()};
    
    auto uptime() const {
        return std::chrono::steady_clock::now() - creation_time;
    }
};

/**
 * @brief Observer interface for arena events
 */
template<typename V>
class arena_observer {
public:
    virtual ~arena_observer() = default;
    virtual void on_event(arena_event event, const V* vector = nullptr, 
                         std::size_t pool_size = 0) = 0;
};

/**
 * @brief Key for identifying compatible vector types
 */
struct vector_key {
    std::type_index type_id;
    std::size_t dimension;
    std::size_t alignment;
    
    vector_key(std::type_index tid, std::size_t dim, std::size_t align = alignof(std::max_align_t))
        : type_id(tid), dimension(dim), alignment(align) {}
    
    bool operator==(const vector_key& other) const {
        return type_id == other.type_id && 
               dimension == other.dimension && 
               alignment == other.alignment;
    }
};

} // namespace rvf

// Hash specialization for vector_key
template<>
struct std::hash<rvf::vector_key> {
    std::size_t operator()(const rvf::vector_key& k) const {
        std::size_t h1 = std::hash<std::type_index>{}(k.type_id);
        std::size_t h2 = std::hash<std::size_t>{}(k.dimension);
        std::size_t h3 = std::hash<std::size_t>{}(k.alignment);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

namespace rvf {

/**
 * @brief Thread-safe memory pool for a specific vector type and dimension
 */
template<typename V>
class vector_pool {
public:
    using vector_ptr = std::unique_ptr<V>;
    
private:
    mutable std::mutex mutex_;
    std::vector<vector_ptr> available_;
    std::atomic<std::size_t> total_allocated_{0};
    std::atomic<std::size_t> peak_size_{0};
    vector_key key_;
    
public:
    explicit vector_pool(const vector_key& key) : key_(key) {}
    
    ~vector_pool() = default;
    
    // Non-copyable, non-movable
    vector_pool(const vector_pool&) = delete;
    vector_pool& operator=(const vector_pool&) = delete;
    vector_pool(vector_pool&&) = delete;
    vector_pool& operator=(vector_pool&&) = delete;
    
    /**
     * @brief Acquire a vector from the pool or create a new one
     */
    template<typename VectorType>
        requires std::same_as<std::decay_t<VectorType>, V>
    vector_ptr acquire(const VectorType& prototype) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (!available_.empty()) {
            auto result = std::move(available_.back());
            available_.pop_back();
            return result;
        }
        
        // Create new vector using clone operation
        auto cloned = rvf::clone(prototype);
        auto ptr = std::make_unique<V>(std::move(rvf::deref_if_needed(cloned)));
        total_allocated_.fetch_add(1, std::memory_order_relaxed);
        
        return ptr;
    }
    
    /**
     * @brief Return a vector to the pool
     */
    void release(vector_ptr&& vec) {
        if (!vec) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        available_.push_back(std::move(vec));
        
        std::size_t current_size = available_.size();
        std::size_t current_peak = peak_size_.load(std::memory_order_relaxed);
        while (current_size > current_peak && 
               !peak_size_.compare_exchange_weak(current_peak, current_size, 
                                               std::memory_order_relaxed)) {
        }
    }
    
    /**
     * @brief Get pool statistics
     */
    struct pool_stats {
        std::size_t available_count;
        std::size_t total_allocated;
        std::size_t peak_size;
        vector_key key;
    };
    
    pool_stats get_stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return {
            available_.size(),
            total_allocated_.load(),
            peak_size_.load(),
            key_
        };
    }
    
    /**
     * @brief Clear all available vectors from pool
     */
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        available_.clear();
    }
    
    const vector_key& get_key() const { return key_; }
};

/**
 * @brief RAII wrapper for arena-managed vectors
 */
template<typename V>
class arena_vector {
private:
    std::unique_ptr<V> ptr_;
    std::shared_ptr<vector_pool<V>> pool_;
    std::function<void()> on_destroy_;
    
public:
    arena_vector(std::unique_ptr<V> ptr, std::shared_ptr<vector_pool<V>> pool,
                std::function<void()> on_destroy = nullptr)
        : ptr_(std::move(ptr)), pool_(std::move(pool)), on_destroy_(std::move(on_destroy)) {}
    
    ~arena_vector() {
        if (ptr_ && pool_) {
            pool_->release(std::move(ptr_));
        }
        if (on_destroy_) {
            on_destroy_();
        }
    }
    
    // Move-only semantics
    arena_vector(const arena_vector&) = delete;
    arena_vector& operator=(const arena_vector&) = delete;
    
    arena_vector(arena_vector&& other) noexcept 
        : ptr_(std::move(other.ptr_)), pool_(std::move(other.pool_)), on_destroy_(std::move(other.on_destroy_)) {}
    
    arena_vector& operator=(arena_vector&& other) noexcept {
        if (this != &other) {
            if (ptr_ && pool_) {
                pool_->release(std::move(ptr_));
            }
            if (on_destroy_) {
                on_destroy_();
            }
            ptr_ = std::move(other.ptr_);
            pool_ = std::move(other.pool_);
            on_destroy_ = std::move(other.on_destroy_);
        }
        return *this;
    }
    
    V& operator*() { return *ptr_; }
    const V& operator*() const { return *ptr_; }
    
    V* operator->() { return ptr_.get(); }
    const V* operator->() const { return ptr_.get(); }
    
    V* get() { return ptr_.get(); }
    const V* get() const { return ptr_.get(); }
    
    explicit operator bool() const { return ptr_ != nullptr; }
};

/**
 * @brief Thread-safe memory arena for vector allocation management
 */
template<typename V>
class memory_arena {
private:
    mutable std::mutex pools_mutex_;
    std::unordered_map<vector_key, std::shared_ptr<vector_pool<V>>> pools_;
    
    mutable std::mutex observers_mutex_;
    std::vector<std::weak_ptr<arena_observer<V>>> observers_;
    
    mutable std::mutex stats_mutex_;
    arena_statistics stats_;
    
    /**
     * @brief Notify all observers of an event
     */
    void notify_observers(arena_event event, const V* vector = nullptr, 
                         std::size_t pool_size = 0) {
        std::lock_guard<std::mutex> lock(observers_mutex_);
        
        // Clean up expired observers while notifying
        auto it = observers_.begin();
        while (it != observers_.end()) {
            if (auto observer = it->lock()) {
                observer->on_event(event, vector, pool_size);
                ++it;
            } else {
                it = observers_.erase(it);
            }
        }
    }
    
    /**
     * @brief Get or create a pool for the given vector type
     */
    template<typename VectorType>
    std::shared_ptr<vector_pool<V>> get_or_create_pool(const VectorType& prototype) {
        auto dim = rvf::dimension(prototype);
        vector_key key{std::type_index(typeid(VectorType)), 
                      static_cast<std::size_t>(dim)};
        
        std::lock_guard<std::mutex> lock(pools_mutex_);
        
        auto it = pools_.find(key);
        if (it != pools_.end()) {
            return it->second;
        }
        
        // Create new pool
        auto pool = std::make_shared<vector_pool<V>>(key);
        pools_[key] = pool;
        
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.total_pools++;
        }
        
        notify_observers(arena_event::pool_created, nullptr, 0);
        
        return pool;
    }
    
public:
    memory_arena() = default;
    ~memory_arena() = default;
    
    // Non-copyable, movable
    memory_arena(const memory_arena&) = delete;
    memory_arena& operator=(const memory_arena&) = delete;
    memory_arena(memory_arena&&) = default;
    memory_arena& operator=(memory_arena&&) = default;
    
    /**
     * @brief Allocate a vector from the arena
     */
    template<typename VectorType>
        requires std::same_as<std::decay_t<VectorType>, V>
    arena_vector<V> allocate(const VectorType& prototype) {
        auto pool = get_or_create_pool(prototype);
        auto ptr = pool->acquire(prototype);
        
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.total_vectors_allocated++;
            stats_.total_vectors_in_use++;
        }
        
        notify_observers(arena_event::vector_allocated, ptr.get());
        
        // Create callback to decrement in-use count when vector is destroyed
        auto decrement_callback = [this]() {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            if (stats_.total_vectors_in_use > 0) {
                stats_.total_vectors_in_use--;
            }
        };
        
        return arena_vector<V>(std::move(ptr), pool, decrement_callback);
    }
    
    /**
     * @brief Register an observer for arena events
     */
    void register_observer(std::shared_ptr<arena_observer<V>> observer) {
        std::lock_guard<std::mutex> lock(observers_mutex_);
        observers_.emplace_back(observer);
    }
    
    /**
     * @brief Get arena statistics
     */
    arena_statistics get_statistics() const {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        return stats_;
    }
    
    /**
     * @brief Get detailed pool information
     */
    std::vector<typename vector_pool<V>::pool_stats> get_pool_statistics() const {
        std::lock_guard<std::mutex> lock(pools_mutex_);
        std::vector<typename vector_pool<V>::pool_stats> result;
        result.reserve(pools_.size());
        
        for (const auto& [key, pool] : pools_) {
            result.push_back(pool->get_stats());
        }
        
        return result;
    }
    
    /**
     * @brief Clear all pools
     */
    void clear() {
        std::lock_guard<std::mutex> lock(pools_mutex_);
        
        for (auto& [key, pool] : pools_) {
            pool->clear();
        }
        
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.total_vectors_in_use = 0;
        }
        
        notify_observers(arena_event::arena_cleared);
    }
    
    /**
     * @brief Remove unused pools
     */
    void compact() {
        std::lock_guard<std::mutex> lock(pools_mutex_);
        
        auto it = pools_.begin();
        while (it != pools_.end()) {
            if (it->second.use_count() == 1) {  // Only arena holds reference
                notify_observers(arena_event::pool_destroyed, nullptr, 0);
                it = pools_.erase(it);
            } else {
                ++it;
            }
        }
        
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.total_pools = pools_.size();
        }
    }
};

/**
 * @brief Global arena instance access
 */
template<typename V>
memory_arena<V>& get_global_arena() {
    static memory_arena<V> arena;
    return arena;
}

/**
 * @brief CPO for arena-based allocation
 */
inline constexpr struct arena_clone_ftor final : tincup::cpo_base<arena_clone_ftor> {
    TINCUP_CPO_TAG("arena_clone")
    inline static constexpr bool is_variadic = false;
    
    template<typename V>
        requires clone_invocable_c<V>
    constexpr auto operator()(const V& x) const
        noexcept(false) // Arena operations may throw
        -> arena_vector<std::decay_t<decltype(rvf::deref_if_needed(rvf::clone(x)))>> {
        using vector_type = std::decay_t<decltype(rvf::deref_if_needed(rvf::clone(x)))>;
        return get_global_arena<vector_type>().allocate(x);
    }
} arena_clone;

} // namespace rvf