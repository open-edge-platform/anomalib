# Contribution Guidelines

## {octicon}`light-bulb` Getting Started

To get started with Anomalib development, follow these steps:

### {octicon}`repo-forked` Fork and Clone the Repository

First, fork the Anomalib repository by following the GitHub documentation on [forking a repo](https://docs.github.com/en/enterprise-cloud@latest/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo). Then, clone your forked repository to your local machine and create a new branch from `main`.

### {octicon}`gear` Set Up Your Development Environment

Set up your development environment to start contributing. This involves installing the required dependencies and setting up pre-commit hooks for code quality checks. Note that this guide assumes you are using [Conda](https://docs.conda.io/en/latest/) for package management. However, the steps are similar for other package managers.

:::{dropdown} Development Environment Setup Instructions
:icon: gear

1. Create and activate a new Conda environment:

   ```bash
   conda create -n anomalib_dev python=3.10
   conda activate anomalib_dev
   ```

   Install the development requirements:

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

2. Install and configure pre-commit hooks:

   ```bash
   prek install
   ```

   Pre-commit hooks help ensure code quality and consistency. After each commit,
   `prek` will automatically run the configured checks for the changed file.
   If you would like to manually run the checks for all files, use:

   ```bash
   prek run --all-files
   ```

   To bypass pre-commit hooks temporarily (e.g., for a work-in-progress commit), use:

   ```bash
   git commit -m 'WIP commit' --no-verify
   ```

   However, make sure to address any pre-commit issues before finalizing your pull request.

:::

## {octicon}`diff` Making Changes

1. **Write Code:** Follow the project's coding standards and write your code with clear intent. Ensure your code is well-documented and includes examples where appropriate. For code quality we use ruff, whose configuration is in [`pyproject.toml`](https://github.com/open-edge-platform/anomalib/blob/main/pyproject.toml) file.

2. **Add Tests:** If your code includes new functionality, add corresponding tests using [pytest](https://docs.pytest.org/en/7.4.x/) to maintain coverage and reliability.

3. **Update Documentation:** If you've changed APIs or added new features, update the documentation accordingly. Ensure your docstrings are clear and follow [Google's docstring guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

4. **Pass Tests and Quality Checks:** Ensure the test suite passes and that your code meets quality standards by running:

   ```bash
   prek run --all-files
   pytest tests/
   ```

5. **Check Licensing:** Ensure you own the code or have rights to use it, adhering to appropriate licensing.

6. **Follow Conventional Commits for PR Titles:** We use [Commitizen](https://commitizen-tools.github.io/commitizen/) to enforce conventional commit format for PR titles and branch names. Since we squash merge PRs, individual commit messages can be in any format during development, but your **PR title must follow conventional commit format**.

   :::{dropdown} PR Title Format (Required)
   :icon: git-commit

   Your **PR title** must follow conventional commit format. Individual commit messages during development can be any format (e.g., "wip", "fix typo"), but the PR title becomes the squash commit message.

   Each PR title consists of a **header**, and optionally a **body** and **footer**:

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

   **PR Title Examples:**

   ```text
   feat(model): add transformer architecture for anomaly detection
   ```

   ```text
   fix(data): handle corrupted image files during training
   ```

   ```text
   docs: update installation instructions for Windows
   ```

   ```text
   chore(ci): migrate from commit message validation to PR title validation
   ```

   **Note:** The PR description can contain additional details, but the title must be concise and follow the format above.

   **Optional Emojis:**
   You can optionally add emojis at the beginning of your PR title for better visual distinction:

   ```text
   🚀 feat(model): add transformer architecture for anomaly detection
   🐞 fix(data): handle corrupted image files during training
   📚 docs: update installation instructions for Windows
   🔧 chore(ci): migrate from commit message validation to PR title validation
   ```

   **Suggested Emoji Mapping (Optional):**
   - 🚀 for `feat` (new features)
   - 🐞 for `fix` (bug fixes)
   - 📚 for `docs` (documentation)
   - 🎨 for `style` (code style/formatting)
   - 🔄 for `refactor` (code refactoring)
   - ⚡ for `perf` (performance improvements)
   - 🧪 for `test` (adding/modifying tests)
   - 📦 for `build` (build system changes)
   - 🔧 for `chore` (general maintenance)
   - 🚧 for `ci` (CI/CD configuration)

   **Note:** Emojis are completely optional. PR titles without emojis are equally valid.

   :::

   :::{dropdown} Branch Naming
   :icon: git-branch

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
   :::

   :::{dropdown} Development Workflow
   :icon: terminal
   <summary>Using Commitizen</summary>
   1. Stage your changes and create a commit using Commitizen:

   **During Development:**
   Individual commits can use any format for convenience:

   ```bash
   git add <files>
   git commit -m "wip: working on transformer model"
   git commit -m "fix typo"
   git commit -m "address review comments"
   ```

   **Creating the PR:**
   Ensure your PR title follows conventional commit format. The PR title becomes the final commit message when merged.

   **Optional - Using Commitizen for PR titles:**
   You can use Commitizen to help format your PR titles:

   ```bash
   # Check if a message follows conventional format
   echo "feat(model): add transformer architecture" | cz check --commit-msg-file -
   ```

   **Version Management:**
   To bump the version based on PR history:

   ```bash
   cz bump
   ```

   :::

## {octicon}`git-pull-request` Submitting Pull Requests

Once you've followed the above steps and are satisfied with your changes:

1. Push your changes to your forked repository.
2. Go to the original Anomalib repository you forked and click "New pull request".
3. Choose your fork and the branch with your changes to open a pull request.
4. Fill in the pull request template with the necessary details about your changes.

We look forward to your contributions!
