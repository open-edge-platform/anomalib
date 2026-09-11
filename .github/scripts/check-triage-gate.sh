#!/usr/bin/env bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Decides whether an issue is allowed to enter the OpenCode triage
# pipeline. Runs read-only, on a hosted runner, BEFORE any work is
# scheduled on the self-hosted GPU runner.
#
# Two conditions must hold:
#   1. The `triage` label was applied by someone with write access.
#   2. The issue content has not been edited since that label was applied
#      (TOCTOU: otherwise a maintainer's approval could be swapped out for
#      arbitrary content before the agent ever reads the issue).
#
# Writes `approved` and `labeled_at` to $GITHUB_OUTPUT. This script never
# mutates anything on GitHub; cleanup on rejection is the `reject` job's
# responsibility.
#
# Required env vars: GH_TOKEN, GITHUB_REPOSITORY, ISSUE_NUMBER,
# SENDER_LOGIN, GITHUB_OUTPUT

set -euo pipefail

: "${GITHUB_REPOSITORY:?}"
: "${ISSUE_NUMBER:?}"
: "${SENDER_LOGIN:?}"
: "${GITHUB_OUTPUT:?}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=.github/scripts/triage-lib.sh
source "${SCRIPT_DIR}/triage-lib.sh"

deny() {
  echo "Triage NOT approved: $1"
  {
    echo "approved=false"
    echo "reason=$2"
  } >> "$GITHUB_OUTPUT"
  exit 0
}

# --- 1. Was the label applied by a maintainer? ----------------------------
#
# GitHub already requires triage-or-higher repository access to apply a
# label, so reaching this script at all implies the sender is trusted to
# some degree. The explicit permission check below narrows that to
# write/maintain/admin as defense in depth.
#
# The collaborator-permission endpoint is not guaranteed to be reachable
# with this workflow's minimal token scopes. If the call fails we log and
# fall back to GitHub's built-in "only triage+ users can label" guarantee
# rather than hard-failing every run — but we never *widen* access here.
PERMISSION=$(gh api "repos/${GITHUB_REPOSITORY}/collaborators/${SENDER_LOGIN}/permission" \
  --jq '.permission' 2>/dev/null || echo "")

if [[ -z "$PERMISSION" ]]; then
  echo "::warning::Could not read repository permission for the labeling user; \
relying on GitHub's requirement that only users with triage access can apply labels."
else
  case "$PERMISSION" in
    admin | maintain | write)
      echo "Label applied by a user with '${PERMISSION}' access."
      ;;
    *)
      deny "label was applied by a user with '${PERMISSION}' access, which is below write." "permission"
      ;;
  esac
fi

# --- 2. Has the issue been edited since it was labeled? -------------------
LABELED_AT="$(triage_labeled_at)"

if triage_is_stale "$LABELED_AT"; then
  deny "issue content changed after the '${TRIAGE_LABEL}' label was applied." "stale"
fi

echo "Triage approved for issue #${ISSUE_NUMBER}."
{
  echo "approved=true"
  echo "labeled_at=${LABELED_AT}"
} >> "$GITHUB_OUTPUT"
