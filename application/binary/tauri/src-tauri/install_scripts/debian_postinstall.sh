#!/bin/sh
# Creates a symlink to the _internal directory as pyinstaller binary expects it to be in the same folder as the binary.
set -e


SOURCE_DIR="/usr/lib/Geti Inspect/_up_/_up_/sidecar/dist/geti-inspect-backend/_internal"
TARGET_DIR="/usr/bin/_internal"

if [ "$1" = "configure" ]; then
    if [ -d "$SOURCE_DIR" ]; then
        echo "Linking $SOURCE_DIR to $TARGET_DIR for backend support..."
        ln -sf "$SOURCE_DIR" "$TARGET_DIR"
    else
        echo "Warning: Source directory $SOURCE_DIR not found."
    fi
fi

exit 0