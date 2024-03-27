// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for SYCL backend macro and type definitions for JiT source
#ifndef CEED_SYCL_JIT_H
#define CEED_SYCL_JIT_H

#define CEED_QFUNCTION(name) inline static int name
#define CEED_QFUNCTION_HELPER inline static
#define CeedPragmaSIMD
#define CEED_Q_VLA 1

// Need quotes for recursive header inclusion
#include "sycl-types.h"

#endif  // CEED_SYCL_JIT_H
