#ifndef PURIPSI_LOGGING_H
#define PURIPSI_LOGGING_H

#include "puripsi/config.h"

#ifdef PURIPSI_DO_LOGGING
#include "puripsi/logging.enabled.h"
#else
#include "puripsi/logging.disabled.h"
#endif

//! \macro Normal but signigicant condition
#define PURIPSI_CRITICAL(...) PURIPSI_LOG_(, critical, __VA_ARGS__)
//! \macro Something is definitely wrong, algorithm exits
#define PURIPSI_ERROR(...) PURIPSI_LOG_(, error, __VA_ARGS__)
//! \macro Something might be going wrong
#define PURIPSI_WARN(...) PURIPSI_LOG_(, warn, __VA_ARGS__)
//! \macro Informational message about normal condition
//! \details Say "Residuals == "
#define PURIPSI_INFO(...) PURIPSI_LOG_(, info, __VA_ARGS__)
//! \macro Output some debugging
#define PURIPSI_DEBUG(...) PURIPSI_LOG_(, debug, __VA_ARGS__)
//! \macro Output internal values of no interest to anyone
//! \details Except maybe when debugging.
#define PURIPSI_TRACE(...) PURIPSI_LOG_(, trace, __VA_ARGS__)

//! High priority message
#define PURIPSI_HIGH_LOG(...) PURIPSI_LOG_(, critical, __VA_ARGS__)
//! Medium priority message
#define PURIPSI_MEDIUM_LOG(...) PURIPSI_LOG_(, info, __VA_ARGS__)
//! Low priority message
#define PURIPSI_LOW_LOG(...) PURIPSI_LOG_(, debug, __VA_ARGS__)
#endif
