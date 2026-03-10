#pragma once

#include <cstdint>
#include <memory>
#include <typeindex>
#include <unordered_map>

namespace engine {

using Entity = uint32_t;

class Registry {
    struct PoolBase {
        virtual ~PoolBase() = default;
    };

    template <typename T>
    struct Pool : PoolBase {
        std::unordered_map<Entity, T> data;
    };

    uint32_t next_id_ = 1;
    std::unordered_map<std::type_index, std::unique_ptr<PoolBase>> pools_;

    template <typename T>
    Pool<T>& pool() {
        auto key = std::type_index(typeid(T));
        auto it = pools_.find(key);
        if (it == pools_.end()) {
            auto p = std::make_unique<Pool<T>>();
            auto& ref = *p;
            pools_.emplace(key, std::move(p));
            return ref;
        }
        return static_cast<Pool<T>&>(*it->second);
    }

public:
    Entity create() { return next_id_++; }

    template <typename T, typename... Args>
    T& emplace(Entity e, Args&&... args) {
        auto& p = pool<T>();
        auto [it, _] = p.data.emplace(e, T{std::forward<Args>(args)...});
        return it->second;
    }

    template <typename T>
    T& get(Entity e) {
        return pool<T>().data.at(e);
    }

    template <typename T>
    bool has(Entity e) {
        auto& p = pool<T>();
        return p.data.contains(e);
    }

    template <typename First, typename... Rest, typename Fn>
    void each(Fn&& fn) {
        auto& primary = pool<First>();
        for (auto& [entity, first_comp] : primary.data) {
            if ((has<Rest>(entity) && ...)) {
                fn(entity, first_comp, get<Rest>(entity)...);
            }
        }
    }
};

}  // namespace engine
