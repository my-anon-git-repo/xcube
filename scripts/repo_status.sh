#!/usr/bin/env bash
# "Where am I?" snapshot: local branch vs its anon mirror.
# Usage: ./scripts/repo_status.sh [branch]   (defaults to current branch)
# Read-only: never modifies the working tree, only fetches anon for freshness.

set -uo pipefail

REPO_DIR="$HOME/xcube"
REMOTE="anon"

cd "$REPO_DIR" || exit 1

CURRENT_BRANCH="$(git symbolic-ref --short HEAD 2>/dev/null || echo 'DETACHED')"
BRANCH="${1:-$CURRENT_BRANCH}"

bold()   { printf "\033[1m%s\033[0m\n" "$1"; }
green()  { printf "\033[32m%s\033[0m\n" "$1"; }
yellow() { printf "\033[33m%s\033[0m\n" "$1"; }
red()    { printf "\033[31m%s\033[0m\n" "$1"; }

echo "========================================"
bold "📍 Where am I? — $BRANCH"
echo "========================================"

if [ "$BRANCH" != "$CURRENT_BRANCH" ]; then
  yellow "⚠️  Checked-out branch is '$CURRENT_BRANCH', not '$BRANCH'."
  echo "   Working-tree section below reflects '$CURRENT_BRANCH'; commit/publish"
  echo "   comparison below is for '$BRANCH' (no checkout needed for that part)."
  echo
fi

# ---------------- working tree (always the checked-out branch) ----------------
STAGED=$(git diff --cached --name-only | grep -c . || true)
UNSTAGED=$(git diff --name-only | grep -c . || true)
UNTRACKED=$(git ls-files --others --exclude-standard | grep -c . || true)

if [ "$STAGED" -eq 0 ] && [ "$UNSTAGED" -eq 0 ]; then
  green "✅ Working tree clean — nothing to commit."
else
  yellow "⚠️  Uncommitted changes: $STAGED file(s) staged, $UNSTAGED modified-not-staged"
  { git diff --cached --stat 2>/dev/null | sed 's/^/     staged:   /'
    git diff --stat 2>/dev/null | sed 's/^/     unstaged: /'
  } | head -20
fi
if [ "$UNTRACKED" -gt 0 ]; then
  echo "   ($UNTRACKED untracked file(s) present, not shown — usually scratch clutter)"
fi

echo
LOCAL_SHA="$(git rev-parse --short "$BRANCH" 2>/dev/null)"
if [ -z "$LOCAL_SHA" ]; then
  red "❌ Local branch '$BRANCH' does not exist."
  exit 1
fi
LOCAL_MSG="$(git log -1 --format='%s' "$BRANCH")"
LOCAL_DATE="$(git log -1 --format='%cr' "$BRANCH")"
echo "🌿 Local  $BRANCH:  $LOCAL_SHA  \"$LOCAL_MSG\"  ($LOCAL_DATE)"

echo
echo "🔄 Fetching $REMOTE/$BRANCH (checking for updates from other machines)..."
if ! git fetch -q "$REMOTE" "$BRANCH" 2>/dev/null; then
  red "❌ Could not fetch $REMOTE/$BRANCH (offline, or branch not on remote yet)"
  exit 1
fi

if ! git show-ref --verify --quiet "refs/remotes/$REMOTE/$BRANCH"; then
  yellow "📦 '$BRANCH' has never been published to $REMOTE."
  echo "========================================"
  yellow "👉 Next: ./scripts/publish_anon.sh $BRANCH"
  exit 0
fi

ANON_SHA="$(git rev-parse --short "$REMOTE/$BRANCH")"
ANON_MSG="$(git log -1 --format='%s' "$REMOTE/$BRANCH")"
ANON_DATE="$(git log -1 --format='%cr' "$REMOTE/$BRANCH")"
echo "🌍 $REMOTE/$BRANCH: $ANON_SHA  \"$ANON_MSG\"  ($ANON_DATE)"

echo
echo "----------------------------------------"

# Anon is a squashed snapshot (no shared history), so compare tree content,
# not commit ancestry -- "ahead/behind/diverged" would be meaningless here.
if git diff --quiet "$BRANCH" "$REMOTE/$BRANCH" --; then
  PUBLISHED=1
  green "✅ Published: $REMOTE/$BRANCH content matches local $BRANCH exactly."
else
  PUBLISHED=0
  yellow "📦 Not yet published: $REMOTE/$BRANCH differs from local $BRANCH."
  echo "   Changed since last publish:"
  git diff --stat "$REMOTE/$BRANCH" "$BRANCH" -- | head -15 | sed 's/^/     /'
fi

echo "========================================"

if [ "$STAGED" -gt 0 ] || [ "$UNSTAGED" -gt 0 ]; then
  red   "👉 Next: commit your changes."
elif [ "$PUBLISHED" -eq 0 ]; then
  yellow "👉 Next: ./scripts/publish_anon.sh $BRANCH"
else
  green "👉 All caught up. Nothing to commit or publish."
fi
