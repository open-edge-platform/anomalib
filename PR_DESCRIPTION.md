## 📝 Description

Migrate CI workflow from individual commit message validation to PR title validation for improved developer experience while maintaining automated changelog and version generation capabilities. **Now with optional emoji support for better visual distinction!**

This change addresses the friction caused by enforcing conventional commit format on every individual commit during development, when we use squash merge strategy. Since PR titles become the final commit messages in the main branch, validating PR titles is more appropriate and developer-friendly.

## ✨ Changes

Select what type of change your PR is:

- [ ] 🐞 Bug fix (non-breaking change which fixes an issue)
- [x] 🔨 Refactor (non-breaking change which refactors the code base)
- [ ] 🚀 New feature (non-breaking change which adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [x] 📚 Documentation update
- [ ] 🔒 Security update

**Detailed Changes:**

- ✅ **Created** new `_reusable-pr-title-check.yaml` workflow for PR title validation
- ✅ **Updated** `pr.yaml` to use PR title validation instead of commit message validation
- ✅ **Removed** unused `_reusable-commit-message-check.yaml` workflow
- ✅ **Updated** pre-commit configuration to only validate branch names (not commit messages)
- ✅ **Updated** documentation in `CONTRIBUTING.md` and developer guide to reflect new workflow
- ✅ **Added** optional emoji support for PR titles with suggested mapping
- ✅ **Maintained** full compatibility with Commitizen for automated changelog and version generation

**Benefits:**

- 🚀 **Reduced developer friction** - no need to craft perfect commit messages during development
- 🎯 **Focused validation** - only PR titles (which become squash commits) are enforced
- 📝 **Same automation** - Commitizen still works for changelogs and versions
- 🔄 **Cleaner main branch** - squash commits with validated conventional format
- 🎨 **Visual enhancement** - optional emojis make PR titles more engaging

## ✅ Checklist

Before you submit your pull request, please make sure you have completed the following steps:

- [x] 📚 I have made the necessary updates to the documentation (if applicable).
- [x] 🧪 I have written tests that support my changes and prove that my fix is effective or my feature works (if applicable).

**Additional Notes:**

- This change aligns with industry best practices for squash-merge workflows
- Individual commits during development can now use any format (e.g., "wip", "fix typo")
- PR titles must follow conventional commit format and are validated in CI
- **Optional emojis** are supported for better visual distinction
- Automated version bumping and changelog generation remain fully functional

For more information about code review checklists, see the [Code Review Checklist](https://github.com/open-edge-platform/anomalib/blob/main/docs/source/markdown/guides/developer/code_review_checklist.md).

---

## 📋 Examples

### Before (Old Workflow)

```bash
# Every commit had to follow conventional format
git commit -m "feat(model): add transformer architecture"
git commit -m "fix(data): handle corrupted files"
git commit -m "docs: update installation guide"
```

### After (New Workflow)

```bash
# During development - any format is fine
git commit -m "wip: working on transformer model"
git commit -m "fix typo in docstring"
git commit -m "address review comments"
git commit -m "add more tests"

# PR title must follow conventional format (with optional emoji)
# Title: 🚀 feat(model): add transformer architecture for anomaly detection
# Title: 🐛 fix(data): handle corrupted image files during training
# Title: 📚 docs: update installation instructions for Windows
# Title: feat(model): add transformer architecture (no emoji also works)
```

### Result

```bash
# Main branch gets clean conventional commits (emojis are stripped for automation)
git log --oneline
> abc123 feat(model): add transformer architecture for anomaly detection
> def456 fix(data): handle corrupted image files during training
> ghi789 docs: update installation instructions for Windows
```

### Automated Processing

```bash
# Commitizen still works perfectly for version bumping
cz bump --dry-run
# Result: bump version from 1.2.3 to 1.3.0 (feat commit detected)

# Changelog generation
cz bump
# Result: Updates CHANGELOG.md with new features and fixes
```

### Suggested Emoji Mapping

- 🚀 for `feat` (new features)
- 🐛 for `fix` (bug fixes)
- 📚 for `docs` (documentation)
- 🎨 for `style` (code style/formatting)
- 🔄 for `refactor` (code refactoring)
- ⚡ for `perf` (performance improvements)
- 🧪 for `test` (adding/modifying tests)
- 📦 for `build` (build system changes)
- 🔧 for `chore` (general maintenance)
- 🚧 for `ci` (CI/CD configuration)
