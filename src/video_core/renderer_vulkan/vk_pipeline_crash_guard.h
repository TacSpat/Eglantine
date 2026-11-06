// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Driver crash guard for pipeline builds.
//
// Some host drivers (seen with RADV on GFX12) crash inside vkCreate*Pipelines on unusual
// translated shaders. The guest shader is already in the disk cache by then, so every later boot
// preloads it and crashes again. Before each driver build this writes a marker file naming the
// pipeline and removes it once the driver returns. A marker still present on the next boot means
// the process died inside the driver: that pipeline is added to blocked.txt and skipped from then
// on (its draws/dispatches are dropped, like any pipeline that fails to build).
//
// Files live in <shader dir>/driver_crash_guard/. Delete blocked.txt to retry the pipelines after a
// driver update. EDEN_PIPELINE_CRASH_GUARD=0 disables the guard.

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_set>

#include <fmt/format.h>

#include "common/common_types.h"
#include "common/fs/path_util.h"
#include "common/logging.h"

namespace Vulkan::PipelineCrashGuard {

enum class Kind : char { Graphics = 'G', Compute = 'C' };

namespace detail {

struct State {
    bool enabled{};
    std::filesystem::path dir;
    std::mutex mutex;
    std::unordered_set<std::string> blocked;
};

inline std::string MakeId(Kind kind, u64 hash) {
    return fmt::format("{}{:016x}", static_cast<char>(kind), hash);
}

inline std::filesystem::path MarkerPath(const State& state) {
    const auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
    return state.dir / fmt::format("inflight-{:016x}", static_cast<u64>(tid));
}

inline void Init(State& s) {
    const char* env = std::getenv("EDEN_PIPELINE_CRASH_GUARD");
    s.enabled = env == nullptr || env[0] != '0';
    if (!s.enabled) {
        return;
    }
    std::error_code ec;
    s.dir = Common::FS::GetEdenPath(Common::FS::EdenPath::ShaderDir) / "driver_crash_guard";
    std::filesystem::create_directories(s.dir, ec);
    if (ec) {
        LOG_WARNING(Render_Vulkan, "Pipeline crash guard disabled: cannot create {}: {}",
                    s.dir.string(), ec.message());
        s.enabled = false;
        return;
    }
    const auto blocked_path = s.dir / "blocked.txt";
    // Markers left by a process that died inside the driver become blocked pipelines.
    for (const auto& entry : std::filesystem::directory_iterator(s.dir, ec)) {
        if (entry.path().filename().string().rfind("inflight-", 0) != 0) {
            continue;
        }
        std::string id;
        {
            std::ifstream marker(entry.path());
            std::getline(marker, id);
        }
        if (!id.empty()) {
            std::ofstream(blocked_path, std::ios::app) << id << '\n';
            LOG_WARNING(Render_Vulkan,
                        "Pipeline {} crashed the host driver last time; it will be skipped "
                        "(delete {} to retry)",
                        id, blocked_path.string());
        }
        std::error_code remove_ec;
        std::filesystem::remove(entry.path(), remove_ec);
    }
    std::ifstream blocked_file(blocked_path);
    for (std::string line; std::getline(blocked_file, line);) {
        if (!line.empty()) {
            s.blocked.insert(line);
        }
    }
    if (!s.blocked.empty()) {
        LOG_WARNING(Render_Vulkan, "Pipeline crash guard: {} pipeline(s) blocked",
                    s.blocked.size());
    }
}

inline State& Get() {
    static State state;
    static std::once_flag once;
    std::call_once(once, [] { Init(state); });
    return state;
}

} // namespace detail

/// Whether this pipeline crashed the driver on an earlier run and must not be built.
inline bool IsBlocked(Kind kind, u64 hash) {
    auto& state = detail::Get();
    if (!state.enabled) {
        return false;
    }
    const auto id = detail::MakeId(kind, hash);
    std::scoped_lock lock{state.mutex};
    if (!state.blocked.contains(id)) {
        return false;
    }
    static std::atomic<u32> reports{0};
    if (reports.fetch_add(1, std::memory_order_relaxed) < 32) {
        LOG_WARNING(Render_Vulkan, "Skipping pipeline {}: it crashed the host driver before", id);
    }
    return true;
}

/// Marks the calling thread as inside the driver building this pipeline for its lifetime.
class Scope {
public:
    Scope(Kind kind, u64 hash) {
        auto& state = detail::Get();
        if (!state.enabled) {
            return;
        }
        path = detail::MarkerPath(state);
        std::ofstream marker(path, std::ios::trunc);
        marker << detail::MakeId(kind, hash) << '\n';
        marker.flush();
        active = static_cast<bool>(marker);
    }
    ~Scope() {
        if (active) {
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    std::filesystem::path path;
    bool active{};
};

} // namespace Vulkan::PipelineCrashGuard
