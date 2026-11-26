# Agentic Actions Auditor

Static security analysis skill for GitHub Actions workflows that invoke AI coding agents (Claude Code Action, Gemini CLI, OpenAI Codex, GitHub AI Inference).

Detects attack vectors where attacker-controlled input reaches AI agents running in CI/CD pipelines, including env var intermediary patterns, direct expression injection, dangerous sandbox configurations, and wildcard user allowlists.

## Source

This skill is derived from the Trail of Bits `skills` project and adapts content from the upstream `plugins/agentic-actions-auditor` subtree.

- **Upstream**: <https://github.com/trailofbits/skills> (`plugins/agentic-actions-auditor`)
- **Attribution**: Original content Copyright Trail of Bits and contributors.
- **License**: CC-BY-SA-4.0
- **Retrieved**: 2026-05-12

Redistribution and adaptation of this skill must preserve the upstream attribution and CC-BY-SA-4.0 licensing terms.
