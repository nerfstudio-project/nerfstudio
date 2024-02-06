# Contributing

**💝 We're excited to have you join the nerfstudio family 💝**

We welcome community contributions to Nerfstudio! Whether you want to fix bugs, improve the documentation, or introduce new features, we appreciate your input.

Bug fixes and documentation improvements are highly valuable to us. If you come across any bugs or find areas where the documentation can be enhanced, please don't hesitate to submit a pull request (PR) with your proposed changes. We'll gladly review and integrate them into the project.

For larger feature additions, we kindly request you to reach out to us on [Discord](https://discord.gg/uMbNqcraFc) in the `#contributing` channel and create an issue on GitHub. This will allow us to discuss the feature in more detail and ensure that it aligns with the goals and direction of the repository. We cannot guarantee that the feature will be added to Nerfstudio.

In addition to code contributions, we also encourage contributors to add their own methods to our documentation. For more information on how to contribute new methods, please refer to the documentation [here](../developer_guides/new_methods.md).

## Overview

Below are the various tooling features our team uses to maintain this codebase.

| Tooling              | Support                                                    |
| -------------------- | ---------------------------------------------------------- |
| Formatting & Linting | [Ruff](https://beta.ruff.rs/docs/)                         |
| Type checking        | [Pyright](https://github.com/microsoft/pyright)            |
| Testing              | [pytest](https://docs.pytest.org/en/7.1.x/)                |
| Docs                 | [Sphinx](https://www.sphinx-doc.org/en/master/)            |
| Docstring style      | [Google](https://google.github.io/styleguide/pyguide.html) |
| JS Linting           | [eslint](https://eslint.org/)                              |

## Requirements

To install the required packages and register the pre-commit hook:

```bash
pip install -e .[dev]
pip install -e .[docs]
pre-commit install
```

This will ensure you have the required packages to run the tests, linter, build the docs, etc.
The pre-commit hook will ensure your commits comply with the repository's code style rules.

You may also need to install [pandoc](https://pandoc.org/). If you are using `conda` you can run the following:

```bash
conda install -c conda-forge pandoc
```

## Committing code

1. Make your modifications ✏️
2. Perform local checks ✅

   To ensure that you will be passing all tests and checks on github, you will need to run the following command:

   ```bash
   ns-dev-test
   ```

   This will perform the following checks and actions:

   - Formatting and linting: Ensures code is consistently and properly formatted.
   - Type checking: Ensures static type safety.
   - Pytests: Runs pytests locally to make sure added code does not break existing logic.
   - Documentation build: Builds docs locally. Ensures changes do not result in warnings/errors.
   - Licensing: Automatically adds licensing headers to the correct files.

   :::{admonition} Attention
   :class: attention
   In order to merge changes to the code base, all of these checks must be passing. If you pass these tests locally, you will likely pass on github servers as well (results in a green checkmark next to your commit).
   :::

3. Open pull request! 💌

:::{admonition} Note
:class: info

We will not review the pull request until it is passing all checks.
:::

## Maintaining documentation

### Building

Run the following to build the documentation:

```bash
python nerfstudio/scripts/docs/build_docs.py
```

:::{admonition} Tip
:class: info

- Rerun `make html` when documentation changes are made
- `make clean` is necessary if the documentation structure changes.
  :::

### Auto build

As you change or add models/components, the auto-generated [Reference API](https://docs.nerf.studio/reference/api/index.html) may change.
If you want the code to build on save you can use [sphinx autobuild](https://github.com/executablebooks/sphinx-autobuild).

:::{admonition} Tip
:class: info

If changes to the structure are made, the build files may be incorrect.
:::

```bash
pip install sphinx-autobuild
sphinx-autobuild docs docs/_build/html
```

### Adding notebooks

We support jupyter notebooks in our documentation. To improve the readability, the following custom tags can be added to the top of each code cell to hide or collapse the code.

| Tag           | Effect                                               |
| ------------- | ---------------------------------------------------- |
| # HIDDEN      | Hide code block and output                           |
| # COLLAPSED   | Collapse the code in a dropdown but show the results |
| # OUTPUT_ONLY | Only show the cell's output                          |
