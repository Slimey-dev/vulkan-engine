#pragma once

#include <cstdio>
#include <format>
#include <string>

namespace engine {

template <typename... Args>
void logInfo(std::format_string<Args...> fmt, Args&&... args) {
    auto msg = std::format(fmt, std::forward<Args>(args)...);
    std::fprintf(stdout, "[INFO] %s\n", msg.c_str());
}

template <typename... Args>
void logError(std::format_string<Args...> fmt, Args&&... args) {
    auto msg = std::format(fmt, std::forward<Args>(args)...);
    std::fprintf(stderr, "[ERROR] %s\n", msg.c_str());
}

template <typename... Args>
void logWarn(std::format_string<Args...> fmt, Args&&... args) {
    auto msg = std::format(fmt, std::forward<Args>(args)...);
    std::fprintf(stderr, "[WARN] %s\n", msg.c_str());
}

}  // namespace engine
