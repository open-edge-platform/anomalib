#!/bin/sh
# Post-install script for the Anomalib Studio .deb package.
#
# Tauri's deb bundler places files in two directories:
#   - externalBin (sidecar)  → /usr/bin/anomalib-studio-backend-<triple>
#   - resources (_internal)  → /usr/lib/Anomalib Studio/_internal/
#
# PyInstaller requires _internal to be in the same directory as the sidecar
# binary. This script creates a symlink so the runtime can find it.
set -e

if [ "$1" = "configure" ]; then
    APP_LIB="/usr/lib/Anomalib Studio"
    TARGET="/usr/bin/_internal"

    # Search known locations where Tauri may have placed _internal
    for candidate in \
        "$APP_LIB/_internal" \
        "$APP_LIB/resources/_internal" \
        "$APP_LIB/sidecar/_internal" \
    ; do
        if [ -d "$candidate" ]; then
            ln -snf "$candidate" "$TARGET"
            echo "Linked $candidate -> $TARGET"
            exit 0
        fi
    done

    echo "Warning: _internal resource directory not found; backend sidecar may fail to start."
fi

exit 0
