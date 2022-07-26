# Contributing to nerfactory workflow

The project is set up for development in VSCode, we reccomend using it if you plan to contribute.

## TLDR

|                 |        |
| --------------- | ------ |
| linter          | Black  |
| Testing         | PyTest |
| Docs            | Sphinx |
| Docstring style | Google |

## Commiting

1. Make your modifications
2. Perform local checks

   To ensure that you will be passing all tests and checks on github, you will need to run the following command:

   ```bash
   python scripts/debugging/run_actions.py
   ```

   This will perform the following checks and actions:

   - Black/ Linting style check: Ensure that the code is consistently and properly formatted.
   - Pytests: Runs pytests locally to make sure the logic of any added code does not break existing logic.
   - Documentation build: Builds the documentation locally to make sure none of the changes result in any warnings or errors in the docs.
   - Licensing: automatically adds licensing headers to the correct files.

In order to merge changes to the code base, all of these checks must be passing. If you pass these tests locally, you will likely pass on github servers as well (results in a green checkmark next to your commit).

3. Open pull request

## Documentation

### Requirements

To install the required packages:

```bash
pip install -r docs/requirements.txt
```

You may also need to install [pandoc](https://pandoc.org/). If you are using `conda` you can run the following:

```bash
conda install -c conda-forge pandoc
```

### Building

Run the following to build the documentation:

```bash
python scripts/docs/build_docs.py
```

Rerun `make html` when documentation changes are made, `make clean` is necissary if the documentation structure changes.

### Notebooks

We support jupyter notbooks in our documentation. To improve the readability, the following custom tags can be added to the top of each code cell to hide or collapse the code.

| Tag           | Effect                                               |
| ------------- | ---------------------------------------------------- |
| # HIDDEN      | Hide code block and output                           |
| # COLLAPSED   | Collapse the code in a dropdown but show the restuls |
| # OUTPUT_ONLY | Only show the cell's output                          |

### Auto build

If you want the code to build on save you can use [sphinx autobuild](https://github.com/executablebooks/sphinx-autobuild). If changes to the structure are made, the build files may be incorrect.

```bash
pip install sphinx-autobuild
sphinx-autobuild docs docs/_build/html
```
