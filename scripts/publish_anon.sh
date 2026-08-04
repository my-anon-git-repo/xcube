#!/usr/bin/env bash
# Publish an anonymous snapshot of ~/xcube
# Usage: ./publish_anon.sh <local-branch>

set -euo pipefail

# ---------------- USAGE ----------------
if [ "$#" -ne 1 ]; then
  echo "❌ Usage: $0 <local-branch>"
  exit 1
fi

LOCAL_BRANCH="$1"

# ---------------- CONFIG ----------------
LOCAL_REPO="$HOME/xcube"
PUBLISH_DIR="$HOME/xcube-publish"

ANON_REMOTE_NAME="anon"
ANON_REMOTE_URL="git@github-myanon:my-anon-git-repo/xcube.git"

ANON_NAME="Anonymous"
ANON_EMAIL="anonymous@users.noreply.github.com"
COMMIT_MSG="Update anonymous snapshot"
# --------------------------------------

confirm() {
  echo
  read -rp "👉 $1 [y/N]: " response
  case "$response" in
    [yY][eE][sS]|[yY]) ;;
    *) echo "❌ Aborted."; exit 1 ;;
  esac
}

echo "========================================"
echo "🚀 Anonymous Publish Script"
echo "========================================"
echo "📁 Local repo   : $LOCAL_REPO"
echo "🌿 Local branch : $LOCAL_BRANCH"
echo "📦 Publish dir  : $PUBLISH_DIR"
echo "🌍 Remote repo  : $ANON_REMOTE_URL"
echo "👤 Commit as    : $ANON_NAME <$ANON_EMAIL>"
echo "========================================"

confirm "Continue with these settings?"

# -------- SAFETY CHECKS --------
echo
echo "🔍 Running safety checks..."

if [ ! -d "$LOCAL_REPO/.git" ]; then
  echo "❌ ERROR: $LOCAL_REPO is not a git repository"
  exit 1
fi

if ! git -C "$LOCAL_REPO" rev-parse --verify "$LOCAL_BRANCH" >/dev/null 2>&1; then
  echo "❌ ERROR: Branch '$LOCAL_BRANCH' does not exist"
  exit 1
fi

echo "✅ Repository and branch look good."

# -------- CLEAN PUBLISH DIR --------
echo
echo "🧹 This will DELETE the publish directory:"
echo "   $PUBLISH_DIR"
confirm "Proceed with cleaning publish directory?"

rm -rf "$PUBLISH_DIR"
mkdir -p "$PUBLISH_DIR"
cd "$PUBLISH_DIR"

# -------- EXPORT SNAPSHOT --------
echo
echo "📦 Exporting snapshot of '$LOCAL_BRANCH' (no git history)..."
confirm "Proceed with snapshot export?"

git -C "$LOCAL_REPO" archive "$LOCAL_BRANCH" | tar -x

# -------- INIT FRESH REPO --------
echo
echo "🆕 Initializing fresh git repository..."
git init -q
git checkout -qb "$LOCAL_BRANCH"

git config user.name  "$ANON_NAME"
git config user.email "$ANON_EMAIL"

# -------- COMMIT --------
echo
echo "📝 Creating anonymous commit..."
git add -A
git commit -qm "$COMMIT_MSG"

# -------- PUSH --------
echo
echo "🚀 About to FORCE PUSH to:"
echo "   $ANON_REMOTE_URL"
echo "   branch: $LOCAL_BRANCH"
confirm "This will overwrite the remote branch. Continue?"

git remote remove "$ANON_REMOTE_NAME" 2>/dev/null || true
git remote add "$ANON_REMOTE_NAME" "$ANON_REMOTE_URL"
git push -f "$ANON_REMOTE_NAME" "$LOCAL_BRANCH"

echo
echo "========================================"
echo "✅ Anonymous snapshot published successfully!"
echo "🌍 Remote : $ANON_REMOTE_URL"
echo "🌿 Branch : $LOCAL_BRANCH"
echo "========================================"
