# 🔒 Security Policy

Intel is committed to rapidly addressing security vulnerabilities affecting our
customers and providing clear guidance on the solution, impact, severity, and
mitigation.

## Security Tools and Practices

### Integrated Security Scanning

To ensure our codebase remains secure, we leverage GitHub Actions for continuous
security scanning (on pre-commit, PR and periodically) with the following tools:

- [CodeQL](https://docs.github.com/en/code-security/code-scanning/introduction-to-code-scanning/about-code-scanning-with-codeql): static analysis tool to check Python code and GitHub Actions workflows
- [Semgrep](https://github.com/semgrep/semgrep): static analysis tool to check Python code; ML-specific Semgrep rules developed by [Trail of Bits](https://github.com/trailofbits/semgrep-rules?tab=readme-ov-file#python) are used
- [Bandit](https://github.com/PyCQA/bandit): Static analysis tool to check Python code
- [Zizmor](https://github.com/woodruffw/zizmor): Static analysis tool to check GitHub Actions workflows
- [Trivy](https://github.com/aquasecurity/trivy): Check misconfigurations and detect security issues in dependencies
- [Dependabot](https://docs.github.com/en/code-security/getting-started/dependabot-quickstart-guide): to detect security issues in dependencies

| Tool       | Pre-commit | PR-checks | Periodic |
| ---------- | ---------- | --------- | -------- |
| CodeQL     |            | ✅        | ✅       |
| Semgrep    |            | ✅        | ✅       |
| Bandit     | ✅         | ✅        | ✅       |
| Zizmor     | ✅         | ✅        | ✅       |
| Trivy      |            |           | ✅       |
| Dependabot |            |           | ✅       |

> **NOTE:** Semgrep [does not support](https://github.com/semgrep/semgrep/issues/1330) Windows, therefore it is not currently used in pre-commit.

## 🚨 Reporting a Vulnerability

Please report any security vulnerabilities in this project utilizing [Intel's vulnerability handling guidelines](https://www.intel.com/content/www/us/en/security-center/vulnerability-handling-guidelines.html).

## 📢 Security Updates and Announcements

Users interested in keeping up-to-date with security announcements and updates
can:

- Follow the [GitHub repository](https://github.com/open-edge-platform/anomalib) 🌐
- Check the [Releases](https://github.com/open-edge-platform/anomalib/releases)
  section of our GitHub project 📦

We encourage users to report security issues and contribute to the security of
our project 🛡️. Contributions can be made in the form of code reviews, pull
requests, and constructive feedback. Refer to our
[CONTRIBUTING.md](CONTRIBUTING.md) for more details.

---

> **NOTE:** This security policy is subject to change 🔁. Users are encouraged
> to check this document periodically for updates.
