# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logging utilities for srtctl."""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for srtctl."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    for lib in ("urllib3", "boto3", "botocore"):
        logging.getLogger(lib).setLevel(logging.WARNING)
