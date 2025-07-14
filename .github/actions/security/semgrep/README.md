# Semgrep (composite)

This composite action executes GitHub Actions workflows scanning using [Semgrep](https://github.com/semgrep/semgrep), providing configurable security analysis capabilities.

## Usage

Example usage in a repository on PR (checks only changed files):

```yaml
name: Semgrep scan

on:
  pull_request:

jobs:
  Semgrep:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Semgrep scan
        uses: ./.github/actions/security/semgrep
        with:
          scan-scope: changed
          severity: HIGH
          fail-on-findings: false
```

Example usage in a repository on schedule (checks all scope), uploads results in SARIF format:

```yaml
name: Semgrep scan

on:
  schedule:
    - cron: "0 2 * * *"

permissions:
  contents: read
  security-events: write # to upload sarif output

jobs:
  Semgrep:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Semgrep scan
        uses: ./.github/actions/security/semgrep
        with:
          scan-scope: all
          severity: LOW
          fail-on-findings: false
```

## Inputs

| Name               | Type    | Description                                        | Default Value                                          | Required |
| ------------------ | ------- | -------------------------------------------------- | ------------------------------------------------------ | -------- |
| `scan-scope`       | String  | Scope of files to scan (all/changed)               | `changed`                                              | No       |
| `paths`            | String  | Paths to scan when using all scope                 | `.`                                                    | No       |
| `severity`         | String  | Minimum severity level to report (LOW/MEDIUM/HIGH) | `LOW`                                                  | No       |
| `config`           | String  | Semgrep rules or config to use                     | `p/default p/cwe-top-25 p/trailofbits p/owasp-top-ten` | No       |
| `output-format`    | String  | Format for scan results (plain/json/sarif)         | `sarif`                                                | No       |
| `fail-on-findings` | boolean | Whether to fail the action if issues are found     | `true`                                                 | No       |
| `timeout`          | String  | Maximum time to run semgrep in seconds             | `300`                                                  | No       |

## Outputs

| Name          | Type   | Description                       |
| ------------- | ------ | --------------------------------- |
| `scan_result` | String | Exit code of the Semgrep scan     |
| `report_path` | String | Path to the generated report file |

## Required permissions

This composite action requires `security-events: write` to upload SARIF results into Security tab.
