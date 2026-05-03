#!/usr/bin/env bash
# Legacy entrypoint kept for old docs/bookmarks.
# The canonical setup is QuicksetupScripts/lerobotSetup.sh.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$(realpath "$0")")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

exec bash "$REPO_DIR/QuicksetupScripts/lerobotSetup.sh" "$@"
