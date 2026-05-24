#!/usr/bin/env zsh
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p vendor

clone_or_update() {
  local name="$1"
  local url="$2"
  local rev="$3"
  local path="vendor/$name"

  if [[ ! -d "$path/.git" ]]; then
    git clone "$url" "$path"
  fi

  git -C "$path" fetch --tags origin
  git -C "$path" checkout "$rev"
}

clone_or_update "SAELens" \
  "https://github.com/jbloomAus/SAELens.git" \
  "3b3f4cacf992645f1f7c08525ed6c122a9cd30a1"

clone_or_update "param-decomp" \
  "https://github.com/goodfire-ai/param-decomp.git" \
  "c6314c9f702b81af593927025aa0ae5aaed4ca4c"
