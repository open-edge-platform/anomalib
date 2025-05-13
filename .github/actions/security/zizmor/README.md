# Zizmor (composite)

This composite action executes GitHub Actions workflows scanning using [Zizmor](https://github.com/woodruffw/zizmor), providing configurable security analysis capabilities.

## Usage

Example usage in a repository on PR (checks only changed files):

```yaml
name: Zizmor scan

on:
  pull_request:

jobs:
  zizmor:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Zizmor scan
        uses: ./.github/actions/security/zizmor
        with:
          scan-scope: changed
          severity-level: MEDIUM
          confidence-level: HIGH
          fail-on-findings: true
```

Example usage in a repository on schedule (checks all scope), uploads results in SARIF format:

```yaml
name: Zizmor scan

on:
  schedule:
    - cron: "0 2 * * *"

permissions:
  contents: read
  security-events: write # to upload sarif output

jobs:
  zizmor:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Zizmor scan
        uses: ./.github/actions/security/zizmor
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
| `zizmor-version`   | String  | Zizmor version                                       | `1.6.0`       | No       |

If necessary, put zizmor configuration into default location `.github/zizmor.yml` - zizmor will discover and us it.

There's no top-level YAML way to declare a variable scoped to a composite action and available in step options, therefore we use input to pass Zizmor version.

## Outputs

| Name          | Type   | Description                       |
| ------------- | ------ | --------------------------------- |
| `scan_result` | String | Exit code of the Zizmor scan      |
| `report_path` | String | Path to the generated report file |

## Required permissions

This composite action requires `security-events: write` to upload SARIF results into Security tab.
