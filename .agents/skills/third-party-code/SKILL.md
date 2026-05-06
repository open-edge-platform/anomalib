---
name: third-party-code
description: Review/generate third-party code attribution, licensing, and notice requirements
---

# Third-Party Code Review

Use this skill when reviewing or generating code that is copied from, adapted from, or substantially based on an
external project.

## Purpose and scope

Use this skill for licensing, attribution, notices, and repository bookkeeping around third-party-derived code.

## Core rule

- Do not add third-party-derived code unless its license allows redistribution and modification compatible with this repository.
- Preserve upstream attribution and required notices.
- Track the imported/adapted component in the repository's third-party inventory.

## Request changes when

- third-party-derived code is added without a matching `third-party-programs.txt` entry;
- a colocated `LICENSE` file is missing;
- upstream attribution or license text is removed or weakened;
- license compatibility has not been verified for copied or adapted code.

## Required actions for new third-party-derived code

- Add or update an entry in `third-party-programs.txt`.
- Add a colocated `LICENSE` file in the relevant component or subtree.
- Name the upstream project and author/source in that `LICENSE` file.
- Include the upstream license text or required notice in that `LICENSE` file.
- Keep the anomalib-side copyright/SPDX notice pattern when the repository's existing third-party examples do so.

## Required actions for updates to existing third-party-derived code

- Preserve existing attribution, SPDX tags, and license text.
- Do not remove or weaken upstream notices.
- Update `third-party-programs.txt` if the tracked component, source, or licensing metadata changes.

## Repo-grounded review anchors

- `third-party-programs.txt`
- `src/**/LICENSE`

## Review prompts

- Is this code copied or adapted from a third-party source?
- If yes, is there a matching entry in `third-party-programs.txt`?
- Is there a colocated `LICENSE` file with upstream attribution and license text?
- Are required notices preserved in modified files and surrounding documentation?
- Has anyone verified that the upstream license is compatible with redistribution in this repository?

## Reviewer checklist

- Check whether the code is third-party-derived.
- Check `third-party-programs.txt`.
- Check the colocated `LICENSE` file.
- Check attribution and compatibility.
