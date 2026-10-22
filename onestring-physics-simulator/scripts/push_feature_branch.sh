#!/usr/bin/env bash
# Push the current branch to GitHub safely (never force-pushes).
#   bash scripts/push_feature_branch.sh            -> pushes HEAD to origin/<current branch>
#   bash scripts/push_feature_branch.sh <branch>   -> pushes HEAD to origin/<branch>
set -euo pipefail
branch="${1:-$(git rev-parse --abbrev-ref HEAD)}"
if [ -d "$(git rev-parse --git-path rebase-apply)" ]; then
  echo "A git am session is still open. Run: git am --quit   (or finish it), then retry." >&2
  exit 1
fi
git fetch origin "$branch" || true
echo "Commits to push to origin/$branch:"
git log --oneline "origin/$branch..HEAD" 2>/dev/null || git log --oneline -10
if ! git push origin "HEAD:$branch"; then
  echo "Push was rejected (the remote has new commits). Rebasing onto origin/$branch ..." >&2
  git pull --rebase origin "$branch"
  git push origin "HEAD:$branch"
fi
echo "Pushed HEAD to origin/$branch."
