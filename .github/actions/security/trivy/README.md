# Trivy (composite)

This composite action executes GitHub Actions workflows scanning using [Trivy](https://github.com/aquasecurity/trivy), providing configurable security analysis capabilities.

## Usage

Example usage in a repository on schedule (checks all scope), uploads results in SARIF format:

```yaml
name: Trivy scan

on:
  schedule:
    - cron: "0 2 * * *"

permissions:
  contents: read
  security-events: write # to upload sarif output

  trivy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0 # Required for changed files detection
          persist-credentials: false

      # These steps are required to lock dependencies in requirements.txt that will by used by Trivy
      # Adjust based on your project
      - name: Set up Python
        uses: actions/setup-python@0b93645e9fea7318ecaed2b359559ac225c90a2b # v5.3.0
        with:
          python-version: "3.10"
      - name: Install dependencies
        run: python -m pip install pip-tools
      - name: Freeze dependencies
        run: pip-compile --extra=full -o requirements.txt pyproject.toml

      - name: Run Trivy scan
        id: trivy
        uses: ./.github/actions/security/trivy
        with:
          scan_type: "fs"
          scan-scope: all
          severity: LOW
          scanners: "vuln,secret,config"
          format: "sarif"
          timeout: "15m"
          ignore_unfixed: "false"
          generate_sbom: "true"
```

## Inputs

| Name                 | Type    | Description                                                                                                | Default Value                                                                 | Required |
| -------------------- | ------- | ---------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | -------- |
| `scan_type`          | String  | Type of scan to perform (fs/config/image/repo/rootfs)                                                      | `fs`                                                                          | No       |
| `scan-scope`         | String  | Scope of files to scan (all/changed)                                                                       | `changed`                                                                     | No       |
| `scan_target`        | String  | Target to scan (path, image name, or repo URL)                                                             | `.`                                                                           | No       |
| `severity`           | String  | Minimum severity level to report (LOW/MEDIUM/HIGH)                                                         | `LOW`                                                                         | No       |
| `ignore_unfixed`     | String  | Ignore unpatched/unfixed vulnerabilities                                                                   | `false`                                                                       | No       |
| `scanners`           | String  | Scanners to enable (vuln,secret,config)                                                                    | `vuln`                                                                        | No       |
| `misconfig_scanners` | String  | Misconfig scanners to enable (azure-arm,cloudformation,dockerfile,helm,kubernetes,terraform,terraformplan) | `azure-arm,cloudformation,dockerfile,helm,kubernetes,terraform,terraformplan` | No       |
| `format`             | String  | Output format (table,json,sarif,template)                                                                  | `sarif`                                                                       | No       |
| `timeout`            | String  | Timeout duration (e.g., 5m, 10m)                                                                           | `10m`                                                                         | No       |
| `generate_sbom`      | boolean | Generate Software Bill of Materials (SBOM)                                                                 | `false`                                                                       | No       |
| `sbom_format`        | String  | SBOM output format (cyclonedx, spdx, spdx-json)                                                            | `spdx-json`                                                                   | No       |

## Outputs

| Name          | Type   | Description                       |
| ------------- | ------ | --------------------------------- |
| `scan_result` | String | Exit code of the Trivy scan       |
| `report_path` | String | Path to the generated report file |

## Required permissions

This composite action requires `security-events: write` to upload SARIF results into Security tab.
