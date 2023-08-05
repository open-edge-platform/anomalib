"""Test tiled ensemble training script"""

import sys
from tools.train_ensemble import get_parser, train

sys.path.append("tools")


def test_train():
    """Test train.py."""
    # Test when model key is passed
    args = get_parser().parse_args(
        [
            "--config",
            "tests/pre_merge/models/ensemble/dummy_padim_config.yaml",
            "--ens_config",
            "tests/pre_merge/models/ensemble/dummy_ens_config.yaml",
        ]
    )
    train(args)
