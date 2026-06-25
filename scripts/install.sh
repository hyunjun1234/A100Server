#!/usr/bin/env bash
set -euo pipefail
PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-/home/jun/unified_bench}"
mkdir -p "$TARGET" "$TARGET/scripts" "$TARGET/drivers" "$TARGET/systems"
cp -a "$PACKAGE_ROOT/telemetry" "$TARGET/"
cp -a "$PACKAGE_ROOT/native" "$TARGET/"
cp -a "$PACKAGE_ROOT/registry" "$TARGET/"
cp -a "$PACKAGE_ROOT/scripts/." "$TARGET/scripts/"
cp -a "$PACKAGE_ROOT/drivers/kernelllm_adapter.py" "$TARGET/drivers/"
cp -a "$PACKAGE_ROOT/systems/kernelllm.env" "$TARGET/systems/"
cp -a "$PACKAGE_ROOT/systems/qimeng_kernel.env" "$TARGET/systems/"
cp "$PACKAGE_ROOT/config/a100_2gpu.env" "$TARGET/open_suite_config.env"
if [ ! -f "$TARGET/native_benchmarks.env" ]; then cp "$PACKAGE_ROOT/config/native_benchmarks.env" "$TARGET/native_benchmarks.env"; fi
cp "$PACKAGE_ROOT/requirements-telemetry.txt" "$TARGET/"
cp "$PACKAGE_ROOT/METRICS.md" "$TARGET/"
cp "$PACKAGE_ROOT/SOURCES_REVIEW.md" "$TARGET/"
cp "$PACKAGE_ROOT/README_KO.md" "$TARGET/README_OPEN_SUITE_KO.md"
cp "$PACKAGE_ROOT/run_a100_open_suite.sh" "$TARGET/"
chmod +x "$TARGET/run_a100_open_suite.sh" "$TARGET/telemetry/"*.py "$TARGET/telemetry/"*.sh "$TARGET/native/"*.sh "$TARGET/scripts/"*.sh "$TARGET/scripts/"*.py "$TARGET/drivers/kernelllm_adapter.py"
cd "$TARGET"
python3 -m pip install -r requirements-telemetry.txt
mkdir -p results/{telemetry,native_benchmarks,method_capabilities} logs runs models third_party
echo "Installed into: $TARGET"
echo "Next:"
echo "  cd $TARGET"
echo "  ./run_a100_open_suite.sh clone"
echo "  ./run_a100_open_suite.sh doctor"
echo "  nohup ./run_a100_open_suite.sh tiny5 > logs/open_suite_tiny5.out 2>&1 &"
