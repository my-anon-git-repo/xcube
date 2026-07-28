#!/usr/bin/env bash
# Sync a local branch to match anon/<branch> exactly (reset-based sync).
# Usage:
#   ./sync_anon_branch.sh reasoning
#   ./sync_anon_branch.sh plant

set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <branch>"
  echo "Example: $0 reasoning"
  exit 1
fi

BRANCH="$1"
REPO_DIR="$HOME/xcube"
REMOTE="anon"

cd "$REPO_DIR"

# Basic checks
if [ ! -d .git ]; then
  echo "❌ ERROR: $REPO_DIR is not a git repository"
  exit 1
fi

if ! git remote get-url "$REMOTE" >/dev/null 2>&1; then
  echo "❌ ERROR: Remote '$REMOTE' does not exist."
  echo "Add it like:"
  echo "  git remote add anon git@github-myanon:my-anon-git-repo/xcube.git"
  exit 1
fi

echo "🔹 Syncing branch '$BRANCH' to '$REMOTE/$BRANCH'..."
echo "   Repo   : $REPO_DIR"
echo "   Remote : $REMOTE"
echo

# Warn if there are uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
  echo "⚠️ WARNING: You have uncommitted changes."
  echo "Please commit or stash before syncing."
  echo
  git status -sb
  exit 1
fi

# Fetch remote
git fetch "$REMOTE"

# Ensure remote branch exists
if ! git show-ref --verify --quiet "refs/remotes/$REMOTE/$BRANCH"; then
  echo "❌ ERROR: Remote branch '$REMOTE/$BRANCH' not found."
  echo "Available branches on $REMOTE:"
  git branch -r | grep "$REMOTE/" || true
  exit 1
fi

# Checkout local branch (create if missing)
if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  git checkout "$BRANCH"
else
  echo "🆕 Local branch '$BRANCH' does not exist. Creating it..."
  git checkout -b "$BRANCH"
fi

# Backup current local branch HEAD
TS="$(date +%Y%m%d-%H%M%S)"
BACKUP_BRANCH="backup/${BRANCH}-local-${TS}"
git branch "$BACKUP_BRANCH"
echo "🛟 Backup created: $BACKUP_BRANCH"

# Hard reset to remote branch
git reset --hard "$REMOTE/$BRANCH"

# Set upstream
git branch --set-upstream-to="$REMOTE/$BRANCH" "$BRANCH" >/dev/null 2>&1 || true

echo
echo "✅ Sync complete."
echo "   Local branch now matches: $REMOTE/$BRANCH"
echo "   Upstream set."
echo
git status -sb
