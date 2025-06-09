# Bandit (composite)

This composite action executes GitHub Actions workflows scanning using [Bandit](https://github.com/PyCQA/bandit), providing configurable security analysis capabilities.

## Usage

Example usage in a repository on PR (checks only changed files):

```yaml
name: Bandit scan

on:
  pull_request:

jobs:
  Bandit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Bandit scan
        uses: ./.github/actions/security/bandit
        with:
          scan-scope: changed
          severity-level: MEDIUM
          confidence-level: HIGH
          fail-on-findings: true
```

Example usage in a repository on schedule (checks all scope), uploads results in SARIF format:

```yaml
name: Bandit scan

on:
  schedule:
    - cron: "0 2 * * *"

permissions:
  contents: read
  security-events: write # to upload sarif output

jobs:
  Bandit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Bandit scan
        uses: ./.github/actions/security/bandit
        with:
          scan-scope: all
          severity-level: LOW
          confidence-level: LOW
          fail-on-findings: false
```

## Inputs

| Name               | Type    | Description                                          | Default Value | Required |
| ------------------ | ------- | ---------------------------------------------------- | ------------- | -------- |
| `scan-scope`       | String  | Scope of files to scan (all/changed)                 | `changed`     | No       |
| `paths`            | String  | Paths to scan when using all scope                   | `.`           | No       |
| `severity-level`   | String  | Minimum severity level to report (LOW/MEDIUM/HIGH)   | `LOW`         | No       |
| `confidence-level` | String  | Minimum confidence level to report (LOW/MEDIUM/HIGH) | `LOW`         | No       |
| `output-format`    | String  | Format for scan results (plain/json/sarif)           | `sarif`       | No       |
| `fail-on-findings` | boolean | Whether to fail the action if issues are found       | `true`        | No       |
| `Bandit-version`   | String  | Bandit version                                       | `1.9.0`       | No       |

## Outputs

| Name          | Type   | Description                       |
| ------------- | ------ | --------------------------------- |
| `scan_result` | String | Exit code of the Bandit scan      |
| `report_path` | String | Path to the generated report file |

## Required permissions

This composite action requires `security-events: write` to upload SARIF results into Security tab.

scan-scope:
description: "Scope of files to scan (all/changed)"
required: false
default: "changed"
paths:
description: "Paths to scan when using all scope"
required: false
default: "." # all scope by default, exclude_dirs are taken from pyproject.toml
config_file:
description: "Path to pyproject.toml or custom bandit config"
required: false
default: "pyproject.toml"
severity-level:
description: "Minimum severity level to report (all/LOW/MEDIUM/HIGH)"
default: "LOW"
confidence-level:
description: "Minimum confidence level to report (all/LOW/MEDIUM/HIGH)"
required: false
default: "LOW"
output-format:
description: "Format for scan results (json/txt/html/csv/sarif)"
required: false
default: "sarif" # by default to upload into Security tab
fail-on-findings:
description: "Whether to fail the action if issues are found"
required: false
default: "true"
