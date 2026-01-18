#!/usr/bin/env bash
# create_project.sh
# Creates the customer-churn-prediction scaffold in the current directory.
# Usage: ./create_project.sh --branch feature/add-churn-project --push
set -euo pipefail

BRANCH="feature/add-churn-project"
PUSH=false
PR=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --branch) BRANCH="$2"; shift 2;;
    --push) PUSH=true; shift;;
    --pr) PR=true; shift;;
    -h|--help) echo "Usage: $0 [--branch name] [--push] [--pr]"; exit 0;;
    *) echo "Unknown arg $1"; exit 1;;
  esac
done

echo "Scaffold created by the provided files. Add, commit, and push manually or use the commands below."

echo "To initialize git branch and push:"
cat <<'EOT'
git checkout -b feature/add-churn-project
git add .
git commit -m "Add customer churn prediction project scaffold"
git push -u origin feature/add-churn-project

# Create PR with GitHub CLI if available:
# gh pr create --title "Add customer churn prediction scaffold" --body "Adds churn pipeline, training, scoring scripts and sample data." --base main --head feature/add-churn-project
EOT

if [ "$PUSH" = true ]; then
  git checkout -b "${BRANCH}"
  git add .
  git commit -m "Add customer churn prediction project scaffold"
  git push -u origin "${BRANCH}"
  echo "Branch pushed: ${BRANCH}"
  if [ "$PR" = true ]; then
    if command -v gh >/dev/null 2>&1; then
      gh pr create --title "Add customer churn prediction scaffold" --body "Adds churn pipeline, training, scoring scripts and sample data." --base main --head "${BRANCH}"
    else
      echo "gh CLI not installed; please create a PR manually."
    fi
  fi
fi
