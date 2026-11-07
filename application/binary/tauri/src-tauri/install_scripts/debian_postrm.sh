#!/bin/sh
# Post-remove script for the Anomalib Studio .deb package.
# Cleans up the _internal symlink created by the postinst script.
set -e

TARGET="/usr/bin/_internal"

if [ -L "$TARGET" ]; then
    rm -f "$TARGET"
    echo "Removed $TARGET symlink"
fi

exit 0
