# Contributing to Anomalib

We welcome your input! 👐

We want to make it as simple and straightforward as possible to contribute to this project, whether it is a:

- Bug Report
- Discussion
- Feature Request
- Creating a Pull Request (PR)
- Becoming a maintainer

## Bug Report

We use GitHub issues to track the bugs. Report a bug by using our Bug Report Template in [Issues](https://github.com/open-edge-platform/anomalib/issues/new?assignees=&labels=&projects=&template=bug_report.yaml&title=%5BBug%5D%3A+).

## Discussion

We enabled [GitHub Discussions](https://github.com/open-edge-platform/anomalib/discussions/) in anomalib to welcome the community to ask questions and/or propose ideas/solutions. This will not only provide a medium to the community to discuss about anomalib but also help us de-clutter [Issues](https://github.com/open-edge-platform/anomalib/issues/new?assignees=&labels=&template=bug_report.md).

## Feature Request

We utilize GitHub issues to track the feature requests as well. If you are certain regarding the feature you are interested and have a solid proposal, you could then create the feature request by using our [Feature Request Template](https://github.com/open-edge-platform/anomalib/issues/new?assignees=&labels=&template=feature_request.md) in Issues. If it's still in an idea phase, you could then discuss that with the community in our [Discussion](https://github.com/open-edge-platform/anomalib/discussions/categories/ideas).

## Development & PRs

We actively welcome your pull requests:

###  Getting Started

#### 1. Fork and Clone the Repository

First, fork the Anomalib repository by following the GitHub documentation on [forking a repo](https://docs.github.com/en/enterprise-cloud@latest/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo). Then, clone your forked repository to your local machine and create a new branch from `main`.

#### 2. Set Up Your Development Environment

Set up your development environment to start contributing. This involves installing the required dependencies and setting up pre-commit hooks for code quality checks. Note that this guide assumes you are using [Conda](https://docs.conda.io/en/latest/) for package management. However, the steps are similar for other package managers.

<details>
<summary>Development Environment Setup Instructions</summary>

1. Create and activate a new Conda environment:

   ```bash
   conda create -n anomalib_dev python=3.10
   conda activate anomalib_dev
   ```

2. Install the development requirements:

   ```bash
   # Option I: Via anomalib install
   anomalib install --option dev

   #Option II: Via pip install
   pip install -e .[dev]
   ```

   Optionally, for a full installation with all dependencies:

   ```bash
   # Option I: via anomalib install
   anomalib install --option full

   # Option II: via pip install
   pip install -e .[full]
   ```

3. Install and configure pre-commit hooks:

   ```bash
   pre-commit install
   ```

Pre-commit hooks help ensure code quality and consistency. After each commit,
`pre-commit` will automatically run the configured checks for the changed file.
If you would like to manually run the checks for all files, use:

```bash
pre-commit run --all-files
```

To bypass pre-commit hooks temporarily (e.g., for a work-in-progress commit),
use:

```bash
git commit -m 'WIP commit' --no-verify
```

However, make sure to address any pre-commit issues before finalizing your pull request.

</details>

### Making Changes

1. **Write Code:** Follow the project's coding standards and write your code with clear intent. Ensure your code is well-documented and includes examples where appropriate. For code quality we use ruff, whose configuration is in [`pyproject.toml`](pyproject.toml) file.

2. **Add Tests:** If your code includes new functionality, add corresponding tests using [pytest](https://docs.pytest.org/en/7.4.x/) to maintain coverage and reliability.

3. **Update Documentation:** If you've changed APIs or added new features, update the documentation accordingly. Ensure your docstrings are clear and follow [Google's docstring guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

4. **Pass Tests and Quality Checks:** Ensure the test suite passes and that your code meets quality standards by running:

   ```bash
   pre-commit run --all-files
   pytest tests/
   ```

5. **Update the Changelog:** For significant changes, add a summary to the [CHANGELOG](CHANGELOG.md).

6. **Check Licensing:** Ensure you own the code or have rights to use it, adhering to appropriate licensing.

7. **Follow Conventional Commits:** We use [Commitizen](https://commitizen-tools.github.io/commitizen/) to enforce conventional commit messages and branch names. This helps maintain a clean and consistent git history.

   <details>
   <summary>Commit Message Format</summary>

   Each commit message consists of a **header**, a **body**, and a **footer**:

   ```text
   <type>(<scope>): <description>

   [optional body]

   [optional footer]
   ```

   **Types:**

   - `feat`: A new feature
   - `fix`: A bug fix
   - `docs`: Documentation changes
   - `style`: Code style changes
   - `refactor`: Code refactoring
   - `perf`: Performance improvements
   - `test`: Adding or modifying tests
   - `build`: Build system changes
   - `ci`: CI configuration changes
   - `chore`: General maintenance

   **Scopes:**

   - `data`: Data loading, processing, or augmentation
   - `model`: Model architecture or implementation
   - `metric`: Evaluation metrics
   - `utils`: Utility functions
   - `cli`: Command-line interface
   - `docs`: Documentation
   - `ci`: CI/CD configuration
   - `engine`: Training/inference engine
   - `visualization`: Visualization tools
   - `benchmarking`: Benchmarking tools
   - `logger`: Logging functionality
   - `openvino`: OpenVINO integration
   - `notebooks`: Jupyter notebooks

   **Rules:**

   - The type and scope are case-sensitive
   - The type must be lowercase
   - The description should be in present tense
   - The description should not end with a period
   - The description should not be in sentence-case, start-case, pascal-case, or upper-case

   **Examples:**

   ```text
   feat(model): add transformer architecture for anomaly detection

   - Implement self-attention mechanism
   - Add positional encoding
   - Add transformer encoder blocks

   Closes #123
   ```

   ```text
   fix(data): handle corrupted image files

   - Add error handling for corrupted images
   - Skip corrupted files during training
   - Log skipped files for debugging
   ```

   </details>

   <details>
   <summary>Branch Naming</summary>

   Branch names must follow the format:

   ```text
   <type>/<scope>/<description>
   ```

   **Examples:**

   - `feat/model/add-transformer`
   - `fix/data/load-image-bug`
   - `docs/readme/update-installation`
   - `refactor/utils/optimize-performance`

   The type and scope should match the ones used in commit messages.
   </details>

   <details>
   <summary>Using Commitizen</summary>

   1. Stage your changes:

   ```bash
   git add <files>
   # or
   git add .  # to add all changes
   ```

   1. Create a commit using Commitizen:

   ```bash
   # Regular commit
   cz commit
   # or
   cz c

   # Signed commit (recommended)
   cz commit -- -s
   # or
   cz c -- -s
   ```

   Commitizen will guide you through an interactive process to create a conventional commit message.

   To check if your commits follow the conventional format:

   ```bash
   cz check
   ```

   To bump the version based on commit history:

   ```bash
   cz bump
   ```

   </details>

<details>
<summary>Suppressing False Positives</summary>

If necessary, to suppress _false_ positives, add inline comment with specific syntax.
Please also add a comment explaining _why_ you decided to disable a rule or provide a risk-acceptance reason.

#### Bandit

Findings can be ignored inline with `# nosec BXXX` comments.

```python
import subprocess # nosec B404 # this is actually fine
```

[Details](https://bandit.readthedocs.io/en/latest/config.html#exclusions) in Bandit docs.

#### Zizmor

Findings can be ignored inline with `# zizmor: ignore[rulename]` comments.

```yaml
uses: actions/checkout@v3 # zizmor: ignore[artipacked] this is actually fine
```

[Details](https://woodruffw.github.io/zizmor/usage/#with-comments) in Zizmor docs.

#### Semgrep

Findings can be ignored inline with `# nosemgrep: rule-id` comments.

```python
    # nosemgrep: python.lang.security.audit.dangerous-system-call.dangerous-system-call # this is actually fine
    r = os.system(' '.join(command))
```

[Details](https://semgrep.dev/docs/ignoring-files-folders-code) in Semgrep docs.

</details>

### Submitting Pull Requests

Once you've followed the above steps and are satisfied with your changes:

1. Push your changes to your forked repository.
2. Go to the original Anomalib repository you forked and click "New pull request".
3. Choose your fork and the branch with your changes to open a pull request.
4. Fill in the pull request template with the necessary details about your changes.

We look forward to your contributions!

## License

You accept that your contributions will be licensed under the [Apache-2.0 License](https://choosealicense.com/licenses/apache-2.0/) if you contribute to this repository. If this is a concern, please notify the maintainers.
