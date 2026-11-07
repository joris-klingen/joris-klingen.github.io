#!/usr/bin/env bash
# Publish the site: render the committed source (HEAD) outside Dropbox and push _site to gh-pages.
# Rendering inside Dropbox fails: Dropbox locks .quarto/ mid-render.
# Usage (Git Bash, from anywhere):  bash publish.sh
set -euo pipefail

src="$(cd "$(dirname "$0")" && pwd)"
quarto="${QUARTO:-/c/Program Files/RStudio/resources/app/bin/quarto/bin/quarto.exe}"
remote="$(git -C "$src" remote get-url origin)"
tmp="$(mktemp -d)"
cleanup() { [ -n "$tmp" ] && [ -d "$tmp" ] && rm -r "$tmp"; }
trap cleanup EXIT

if [ -n "$(git -C "$src" status --porcelain -- . ':!_site' ':!.quarto')" ]; then
  echo "Uncommitted changes: commit first (the site is built from HEAD)." >&2
  exit 1
fi

# Render ----
mkdir "$tmp/src"
git -C "$src" archive HEAD | tar -x -C "$tmp/src"
(cd "$tmp/src" && "$quarto" render)

# Push to gh-pages ----
git clone -q --branch gh-pages --single-branch --depth 1 "$remote" "$tmp/pages"
cd "$tmp/pages"
git rm -rq --ignore-unmatch .
cp -r "$tmp/src/_site/." .
touch .nojekyll
git add -A
if git diff --cached --quiet; then
  echo "Nothing changed on the site."
else
  git commit -qm "Publish site from $(git -C "$src" rev-parse --short HEAD)"
  git push -q origin gh-pages
  echo "Published $(git -C "$src" rev-parse --short HEAD) to gh-pages."
fi
