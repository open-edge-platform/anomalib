# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Runtime hook to configure matplotlib for frozen applications."""

import os
import sys

# Disable matplotlib font scanning which is slow in frozen apps
os.environ["MPLCONFIGDIR"] = os.path.join(sys._MEIPASS if getattr(sys, "frozen", False) else ".", ".matplotlib")

# Use non-interactive backend
os.environ["MPLBACKEND"] = "Agg"

print("Matplotlib Hook: Configured matplotlib for frozen app (backend=Agg)")
