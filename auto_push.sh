#!/usr/bin/env bash
# Auto commit + push unified_bench code. Safe to run repeatedly.
# Used by the Claude Code Stop hook. Never commits files >95MB.
set -uo pipefail

REPO="/home/jun/unified_bench"
cd "$REPO" 2>/dev/null || exit 0
[ -d .git ] || exit 0

git add -A 2>/dev/null

# Safety guard: refuse to commit any staged file >95MB (GitHub hard limit 100MB)
big=""
while IFS= read -r -d '' f; do
  [ -f "$f" ] || continue
  sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
  [ "$sz" -gt 99000000 ] && big="$big$f ($((sz/1000000))MB)\n"
done < <(git diff --cached --name-only -z)
if [ -n "$big" ]; then
  echo -e "auto_push: ABORT — oversized files staged:\n$big" >&2
  git reset -q
  exit 0
fi

# Nothing changed? exit quietly
git diff --cached --quiet && exit 0

git commit -q -m "auto: $(date '+%Y-%m-%d %H:%M:%S')" || exit 0

# Push only if a remote named origin exists
if git remote get-url origin >/dev/null 2>&1; then
  git push -q origin main 2>/dev/null || git push -q -u origin main 2>/dev/null || \
    echo "auto_push: commit ok, push failed (check remote/auth)" >&2
fi
