"""Run hpo sweep."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from anomalib.utils.hpo import Sweep, get_hpo_parser

if __name__ == "__main__":
    parser = get_hpo_parser()
    args = parser.parse_args()
    sweep = Sweep(
        model=args.model,
        model_config=args.model_config,
        sweep_config=args.sweep_config,
        backend=args.backend,
        entity=args.entity,
    )
    sweep.run()
