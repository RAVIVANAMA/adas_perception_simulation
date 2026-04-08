#pragma once

#include <atomic>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <chrono>

namespace adas {

// ---------------------------------------------------------------------------
// Log severity levels
// ---------------------------------------------------------------------------
enum class LogLevel : int { Trace=0, Debug, Info, Warn, Error, Fatal };

inline const char* toString(LogLevel l) noexcept {
    static constexpr const char* names[] = {
        "TRACE","DEBUG"," INFO"," WARN","ERROR","FATAL"
    };
    return names[static_cast<int>(l)];
}

// ---------------------------------------------------------------------------
// Thread-safe singleton logger
// ---------------------------------------------------------------------------
class Logger {
public:
    using SinkFn = std::function<void(LogLevel, const std::string&)>;

    static Logger& instance() {
        static Logger inst;
        return inst;
    }

    void setLevel(LogLevel l) noexcept  { minLevel_.store(l); }
    LogLevel level()          const     { return minLevel_.load(); }

    void addFileSink(const std::string& path) {
        std::lock_guard<std::mutex> lk(mu_);
        fileStreams_.emplace_back(path, std::ios::out | std::ios::app);
    }

    void addCustomSink(SinkFn fn) {
        std::lock_guard<std::mutex> lk(mu_);
        sinks_.push_back(std::move(fn));
    }

    void log(LogLevel lvl, const std::string& msg,
             const char* file, int line) {
        if (lvl < minLevel_.load()) return;

        std::ostringstream ss;
        // ISO-8601-ish timestamp
        auto now  = std::chrono::system_clock::now();
        auto tt   = std::chrono::system_clock::to_time_t(now);
        auto ms   = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now.time_since_epoch()).count() % 1000;
        std::tm tm_info{};
#ifdef _WIN32
        localtime_s(&tm_info, &tt);
#else
        localtime_r(&tt, &tm_info);
#endif
        ss << std::put_time(&tm_info, "%Y-%m-%dT%H:%M:%S")
           << '.' << std::setfill('0') << std::setw(3) << ms
           << " [" << toString(lvl) << "] "
           << '[' << shortFile(file) << ':' << line << "] "
           << msg;

        std::string out = ss.str();

        std::lock_guard<std::mutex> lk(mu_);
        auto& stream = (lvl >= LogLevel::Error) ? std::cerr : std::cout;
        stream << out << '\n';
        for (auto& fs : fileStreams_) fs << out << '\n';
        for (auto& fn : sinks_)      fn(lvl, out);
    }

private:
    Logger() : minLevel_(LogLevel::Info) {}
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    static const char* shortFile(const char* path) noexcept {
        const char* p = path;
        const char* last = path;
        while (*p) { if (*p == '/' || *p == '\\') last = p + 1; ++p; }
        return last;
    }

    std::atomic<LogLevel>      minLevel_;
    std::mutex                 mu_;
    std::vector<std::ofstream> fileStreams_;
    std::vector<SinkFn>        sinks_;
};

} // namespace adas

// ---------------------------------------------------------------------------
// Convenience macros
// ---------------------------------------------------------------------------
#define ADAS_LOG(lvl, msg) \
    do { \
        std::ostringstream _oss; _oss << msg; \
        ::adas::Logger::instance().log(lvl, _oss.str(), __FILE__, __LINE__); \
    } while (0)

#define LOG_TRACE(m) ADAS_LOG(::adas::LogLevel::Trace, m)
#define LOG_DEBUG(m) ADAS_LOG(::adas::LogLevel::Debug, m)
#define LOG_INFO(m)  ADAS_LOG(::adas::LogLevel::Info,  m)
#define LOG_WARN(m)  ADAS_LOG(::adas::LogLevel::Warn,  m)
#define LOG_ERROR(m) ADAS_LOG(::adas::LogLevel::Error, m)
#define LOG_FATAL(m) ADAS_LOG(::adas::LogLevel::Fatal, m)
