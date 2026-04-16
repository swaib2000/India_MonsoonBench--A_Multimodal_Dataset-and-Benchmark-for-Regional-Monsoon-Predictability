#!/usr/bin/env bash
set -euo pipefail

# Sync selected GEE exports from Google Drive to the local GEE_Exports folder.
#
# Requirements:
#   - rclone configured for your Google Drive.
#   - A remote name such as "gdrive:" or "drive:".
#
# Example:
#   bash sync_missing_gee_exports.sh gdrive:
#
# If your rclone remote points directly to "My Drive", the script expects
# completed exports under: <remote>/GEE_Exports/

REMOTE="${1:-}"
DEST="${2:-GEE_Exports}"

if [[ -z "$REMOTE" ]]; then
  echo "Usage: bash sync_missing_gee_exports.sh <rclone-remote:> [local-dest]"
  echo "Example: bash sync_missing_gee_exports.sh gdrive: GEE_Exports"
  exit 1
fi

if ! command -v rclone >/dev/null 2>&1; then
  echo "rclone was not found. Install/configure rclone or download the files manually from Google Drive/GEE_Exports."
  exit 1
fi

mkdir -p "$DEST"

rclone copy "${REMOTE%/}/GEE_Exports" "$DEST" \
  --progress \
  --include "Odisha_*.tif" \
  --include "Telangana_*.tif" \
  --include "Jammu and Kashmir_*.tif" \
  --include "Jammu & Kashmir_*.tif" \
  --exclude "*"

echo "Synced selected exports into: $DEST"

