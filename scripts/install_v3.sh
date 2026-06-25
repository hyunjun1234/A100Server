#!/usr/bin/env bash
set -euo pipefail
PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-/home/jun/unified_bench}"

# 1) Install the complete v2 overlay.
bash "$PACKAGE_ROOT/scripts/install.sh" "$TARGET"

# 2) Copy the A100 fixes accumulated from real runs.
cp "$PACKAGE_ROOT/patches/a100_autokernel_no_uv_fix.sh" "$TARGET/"
cp "$PACKAGE_ROOT/patches/a100_no_uv_followup_fix.sh" "$TARGET/"
cp "$PACKAGE_ROOT/patches/a100_official5_expand.sh" "$TARGET/"
chmod +x \
  "$TARGET/a100_autokernel_no_uv_fix.sh" \
  "$TARGET/a100_no_uv_followup_fix.sh" \
  "$TARGET/a100_official5_expand.sh"

cd "$TARGET"

# Apply in this order. These do not start the long benchmark.
bash ./a100_autokernel_no_uv_fix.sh patch
bash ./a100_no_uv_followup_fix.sh patch
bash ./a100_official5_expand.sh patch

echo
echo "Installed A100 v3 bundle into: $TARGET"
echo "Recommended smoke test:"
echo "  cd $TARGET"
echo "  bash a100_official5_expand.sh tiny5"
echo "  tail -f logs/a100_official5_tiny5_round1.out"
echo
echo "Then subset50 round3:"
echo "  bash a100_official5_expand.sh round3"
