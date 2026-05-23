#!/usr/bin/env zsh
set -euo pipefail

# Backward-compatible wrapper for the old script name.
exec "$(dirname "$0")/run_external_query_benchmark.sh" "$@"
