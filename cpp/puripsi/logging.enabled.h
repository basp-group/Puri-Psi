#ifndef PURIPSI_LOGGING_ENABLED_H
#define PURIPSI_LOGGING_ENABLED_H

#include "puripsi/config.h"
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace puripsi {
namespace logging {
void set_level(std::string const &level, std::string const &name = "");

//! \brief Initializes a logger.
//! \details Logger only exists as long as return is kept alive.
inline std::shared_ptr<spdlog::logger> initialize(std::string const &name = "") {
  auto const result = spdlog::stdout_logger_mt(default_logger_name() + name);
  set_level(default_logging_level(), name);
  return result;
}

//! Returns shared pointer to logger or null if it does not exist
inline std::shared_ptr<spdlog::logger> get(std::string const &name = "") {
  return spdlog::get(default_logger_name() + name);
}

//! \brief Sets loggin level
//! \details Levels can be one of
//!     - "trace"
//!     - "debug"
//!     - "info"
//!     - "warn"
//!     - "err"
//!     - "critical"
//!     - "off"
inline void set_level(std::string const &level, std::string const &name) {
  auto const logger = get(name);
  if(not logger)
    throw std::runtime_error("No logger by the name of " + std::string(name));
#define PURIPSI_MACRO(LEVEL)                                                                        \
  if(level == #LEVEL)                                                                              \
  logger->set_level(spdlog::level::LEVEL)
  PURIPSI_MACRO(trace);
  else PURIPSI_MACRO(debug);
  else PURIPSI_MACRO(info);
  else PURIPSI_MACRO(warn);
  else PURIPSI_MACRO(err);
  else PURIPSI_MACRO(critical);
  else PURIPSI_MACRO(off);
#undef PURIPSI_MACRO
  else throw std::runtime_error("Unknown logging level " + std::string(level));
}

inline bool has_level(std::string const &level, std::string const &name = "") {
  auto const logger = get(name);
  if(not logger)
    return false;

#define PURIPSI_MACRO(LEVEL)                                                                        \
  if(level == #LEVEL)                                                                              \
  return logger->level() >= spdlog::level::LEVEL
  PURIPSI_MACRO(trace);
  else PURIPSI_MACRO(debug);
  else PURIPSI_MACRO(info);
  else PURIPSI_MACRO(warn);
  else PURIPSI_MACRO(err);
  else PURIPSI_MACRO(critical);
  else PURIPSI_MACRO(off);
#undef PURIPSI_MACRO
  else throw std::runtime_error("Unknown logging level " + std::string(level));
}
}
}

//! \macro For internal use only
#define PURIPSI_LOG_(NAME, TYPE, ...)                                                               \
  if(auto puripsi_logging_##__func__##_##__LINE__ = puripsi::logging::get(NAME))                     \
  puripsi_logging_##__func__##_##__LINE__->TYPE(__VA_ARGS__)

#endif
