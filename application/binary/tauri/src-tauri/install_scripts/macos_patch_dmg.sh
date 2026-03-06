#!/bin/bash
# Post-build script for the Anomalib Studio macOS .dmg package.
#
# Tauri's macOS bundler places files in two directories inside the .app:
#   - externalBin (sidecar)  → Contents/MacOS/anomalib-studio-backend-<triple>
#   - resources (_internal)  → Contents/Resources/_internal/
#
# PyInstaller's bootloader, when it detects it is running inside a .app bundle
# (by checking if the executable's parent directory ends with
# ".app/Contents/MacOS"), sets the application home directory to
# "Contents/Frameworks" instead of looking for _internal/ next to the binary.
#
# This means PyInstaller expects the CONTENTS of _internal/ (base_library.zip,
# Python, all collected packages, etc.) to be placed directly in
# Contents/Frameworks/ — NOT in Contents/Frameworks/_internal/.
#
# Reference: PyInstaller v6.18.0 bootloader/src/pyi_main.c lines ~462-486
#
# This script patches the .app inside the .dmg by:
#   - Moving the contents of Resources/_internal/ into Contents/Frameworks/
#   - Removing the now-empty Resources/_internal/ directory
#
# Usage:
#   ./macos_patch_dmg.sh <path-to-dmg>
#
# The script modifies the DMG in-place by:
# 1. Converting the compressed DMG to a read-write format
# 2. Mounting it
# 3. Rearranging the directories
# 4. Unmounting
# 5. Converting back to a compressed read-only DMG
set -euo pipefail

APP_NAME="Anomalib Studio"

if [ $# -ne 1 ]; then
    echo "Usage: $0 <path-to-dmg>"
    exit 1
fi

DMG_PATH="$1"

if [ ! -f "$DMG_PATH" ]; then
    echo "Error: DMG file not found: $DMG_PATH"
    exit 1
fi

echo "Patching DMG: $DMG_PATH"

# Create a temporary directory for the work
WORK_DIR="$(mktemp -d)"

cleanup() {
    hdiutil detach "$MOUNT_POINT" -force 2>/dev/null || true
    rm -rf "$WORK_DIR"
}
trap cleanup EXIT

RW_DMG="$WORK_DIR/rw.dmg"
MOUNT_POINT="$WORK_DIR/mnt"
mkdir -p "$MOUNT_POINT"

# Step 1: Convert to read-write
echo "Converting DMG to read-write format..."
hdiutil convert "$DMG_PATH" -format UDRW -o "$RW_DMG"

# Step 2: Mount the read-write DMG
echo "Mounting DMG..."
hdiutil attach "$RW_DMG" -mountpoint "$MOUNT_POINT" -nobrowse -noverify

# Step 3: Rearrange directories inside the .app bundle
APP_BUNDLE="$MOUNT_POINT/${APP_NAME}.app"
if [ ! -d "$APP_BUNDLE" ]; then
    echo "Error: App bundle not found at $APP_BUNDLE"
    exit 1
fi

MACOS_DIR="$APP_BUNDLE/Contents/MacOS"
RESOURCES_INTERNAL="$APP_BUNDLE/Contents/Resources/_internal"
FRAMEWORKS_DIR="$APP_BUNDLE/Contents/Frameworks"

# Check for source _internal directory
if [ ! -d "$RESOURCES_INTERNAL" ]; then
    # _internal might have already been moved (idempotent)
    if [ -d "$FRAMEWORKS_DIR" ] && [ -f "$FRAMEWORKS_DIR/base_library.zip" ]; then
        echo "_internal contents already in Frameworks/, nothing to do"
    else
        echo "Error: _internal directory not found at $RESOURCES_INTERNAL and Frameworks/ is empty"
        exit 1
    fi
else
    # Move contents of _internal/ into Contents/Frameworks/
    # We move each item individually (not the directory itself) because we
    # want the contents to be directly in Frameworks/, not in Frameworks/_internal/.
    # Using mv instead of cp avoids doubling disk usage on the DMG.
    echo "Moving contents of Resources/_internal/ into Contents/Frameworks/"
    mkdir -p "$FRAMEWORKS_DIR"

    for item in "$RESOURCES_INTERNAL"/*; do
        mv "$item" "$FRAMEWORKS_DIR/"
    done
    # Also move hidden files if any exist
    for item in "$RESOURCES_INTERNAL"/.*; do
        case "$(basename "$item")" in
            .|..) continue ;;
            *) mv "$item" "$FRAMEWORKS_DIR/" ;;
        esac
    done

    rmdir "$RESOURCES_INTERNAL" 2>/dev/null || rm -rf "$RESOURCES_INTERNAL"
    echo "Removed empty Resources/_internal/"
fi

# Remove any stale _internal symlink/directory from MacOS/ that may have been
# left by a previous (incorrect) patch attempt
if [ -L "$MACOS_DIR/_internal" ]; then
    echo "Removing stale _internal symlink from MacOS/"
    rm "$MACOS_DIR/_internal"
elif [ -d "$MACOS_DIR/_internal" ]; then
    echo "Removing stale _internal directory from MacOS/"
    rm -rf "$MACOS_DIR/_internal"
fi

# Verify the layout
echo ""
echo "Verifying patched layout..."

if [ ! -f "$FRAMEWORKS_DIR/base_library.zip" ]; then
    echo "Error: base_library.zip not found in Frameworks/"
    exit 1
fi

if [ ! -f "$FRAMEWORKS_DIR/Python" ]; then
    echo "Error: Python shared library not found in Frameworks/"
    exit 1
fi

echo "  Contents/Frameworks/base_library.zip  OK"
echo "  Contents/Frameworks/Python            OK"
ITEM_COUNT=$(ls -1 "$FRAMEWORKS_DIR" | wc -l | tr -d ' ')
echo "  Contents/Frameworks/ contains $ITEM_COUNT items"
echo ""

# Step 4: Unmount
echo "Unmounting DMG..."
hdiutil detach "$MOUNT_POINT"

# Step 5: Convert back to compressed read-only DMG, replacing the original
echo "Converting back to compressed DMG..."
hdiutil convert "$RW_DMG" -format UDZO -o "$WORK_DIR/final.dmg"

# Replace the original DMG
mv "$WORK_DIR/final.dmg" "$DMG_PATH"

echo "DMG patched successfully: $DMG_PATH"
