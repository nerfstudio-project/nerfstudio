# Contributing

**💝 We're excited to have you join the nerfstudio family 💝**

Below, we document the contribution pipeline and good-to-knows for when you're ready to submit a PR. If you have any questions at all, please don't hesitate to reach out to us on [Discord](https://discord.gg/uMbNqcraFc). We'd love to hear from you!

The project is set up for development in VSCode, we recommend using it if you plan to contribute.

## Overview
Below are the various tooling features our team uses to maintain this codebase.

|    Tooling      |    Support    |
| --------------- | ------------- |
| Linter          | [Black](https://black.readthedocs.io/en/stable/)  |
| Testing         | [PyTest](https://docs.pytest.org/en/7.1.x/) |
| Docs            | [Sphinx](https://www.sphinx-doc.org/en/master/) |
| Docstring style | [Google](https://google.github.io/styleguide/pyguide.html) |

## Requirements

To install the required packages:

```bash
pip install -e .[dev]
pip install -e .[docs]
```

This will ensure you have the required packages to run the tests, linter, build the docs, etc.

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

   - Black/ Linting style check: Ensures code is consistently and properly formatted.
   - Pytests: Runs pytests locally to make sure added code does not break existing logic.
   - Documentation build: Builds docs locally. Ensures changes do not result in warnings/errors.
   - Licensing: Automatically adds licensing headers to the correct files.

   :::{admonition} Attention
   :class: attention
      In order to merge changes to the code base, all of these checks must be passing. If you pass these tests locally, you will likely pass on github servers as well (results in a green checkmark next to your commit).
      :::

3. Open pull request! 💌

## Maintaining documentation

### Building

Run the following to build the documentation:

```bash
python scripts/docs/build_docs.py
```

:::{admonition} Tip
:class: info

- Rerun `make html` when documentation changes are made
- `make clean` is necessary if the documentation structure changes.
  :::

### Auto build

As you change or add models/components, the auto-generated [Reference API](https://docs.nerf.studio/en/latest/reference/api/index.html) may change.
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
