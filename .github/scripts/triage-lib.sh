# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# shellcheck shell=bash
#
# Shared helpers for the OpenCode issue-triage pipeline.
#
# This file is meant to be sourced, not executed. It contains the TOCTOU
# guard used by both the `gate` job (before any GPU/LLM work starts) and
# the `apply` job (after it finishes, since the issue can be edited during
# the up-to-20-minute analyze window).
#
# Required env vars for all functions: GH_TOKEN, GITHUB_REPOSITORY,
# ISSUE_NUMBER.

# Label a maintainer applies to request triage. Overridable for testing.
TRIAGE_LABEL="${TRIAGE_LABEL:-triage}"
export TRIAGE_LABEL

# Timestamp (ISO-8601 UTC) of the most recent time the triage label was
# applied to the issue. Empty if the label was never applied.
#
# GitHub's `issues.labeled` webhook payload does not carry the label
# application time, so it has to be recovered from the issue timeline.
triage_labeled_at() {
  gh api "repos/${GITHUB_REPOSITORY}/issues/${ISSUE_NUMBER}/timeline" \
    --paginate \
    -H "Accept: application/vnd.github+json" \
    --jq '.[] | select(.event == "labeled") | select(.label.name == env.TRIAGE_LABEL) | .created_at' |
    sort | tail -n 1
}

# Timestamp (ISO-8601 UTC) of the last edit to the issue's *content*.
#
# Deliberately NOT `issue.updated_at`: that field is bumped by comments,
# label changes, assignment, milestones, etc., so comparing against it
# would abort valid triage runs constantly. The content-edit signals are:
#   - `lastEditedAt` (GraphQL)  -> body edits
#   - `renamed` timeline events -> title edits
# The later of the two is returned. Empty if the issue was never edited.
triage_last_edited_at() {
  local owner="${GITHUB_REPOSITORY%%/*}"
  local repo="${GITHUB_REPOSITORY##*/}"
  local body_edit title_edit

  # shellcheck disable=SC2016 # $owner/$repo/$number are GraphQL variables, not shell ones.
  body_edit=$(gh api graphql \
    -f query='query($owner:String!,$repo:String!,$number:Int!){
      repository(owner:$owner,name:$repo){issue(number:$number){lastEditedAt}}
    }' \
    -f owner="$owner" -f repo="$repo" -F number="$ISSUE_NUMBER" \
    --jq '.data.repository.issue.lastEditedAt // empty')

  title_edit=$(gh api "repos/${GITHUB_REPOSITORY}/issues/${ISSUE_NUMBER}/timeline" \
    --paginate \
    -H "Accept: application/vnd.github+json" \
    --jq '.[] | select(.event == "renamed") | .created_at' |
    sort | tail -n 1)

  printf '%s\n%s\n' "$body_edit" "$title_edit" | grep -v '^$' | sort | tail -n 1
}

# Returns 0 (stale) if the issue content was edited after the triage label
# was applied — i.e. the maintainer vouched for content that is no longer
# what the pipeline would act on. Returns 1 (fresh) otherwise.
#
# Fails closed: a missing label timestamp is treated as stale.
#
# GitHub returns both timestamps as `YYYY-MM-DDTHH:MM:SSZ`, a fixed-width
# UTC format, so lexicographic comparison is equivalent to chronological.
#
# Usage: triage_is_stale "$LABELED_AT"
triage_is_stale() {
  local labeled_at="${1:-}"
  local last_edit

  if [[ -z "$labeled_at" ]]; then
    echo "No '${TRIAGE_LABEL}' label application found on issue #${ISSUE_NUMBER}; failing closed."
    return 0
  fi

  last_edit=$(triage_last_edited_at)

  if [[ -z "$last_edit" ]]; then
    echo "Issue #${ISSUE_NUMBER} has never been edited; content is unchanged since labeling."
    return 1
  fi

  if [[ "$last_edit" > "$labeled_at" ]]; then
    echo "Issue #${ISSUE_NUMBER} was edited at ${last_edit}, after the '${TRIAGE_LABEL}' label was applied at ${labeled_at}."
    return 0
  fi

  echo "Issue #${ISSUE_NUMBER} last edited at ${last_edit}, before the '${TRIAGE_LABEL}' label was applied at ${labeled_at}."
  return 1
}
