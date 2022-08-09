"""Helper that add tags to notebooks based on cell comments."""

import sys
from glob import glob

import dcargs
import nbformat as nbf


def main(check: bool = False):
    """Add tags to notebooks based on cell comments.

    In notebook cells, you can add  the folling tags to the notebook by adding a comment:
    "# HIDDEN" - This cell will be hidden from the notebook.
    "# OUTPUT_ONLY" - This cell will only show the output.
    "# COLLAPSED" - Hide the code and include a button to show the code.

    Args:
        check: check will not modify the notebooks.
    """
    # Collect a list of all notebooks in the content folder
    notebooks = glob("./docs/**/*.ipynb", recursive=True)

    # Text to look for in adding tags
    text_search_dict = {
        "# HIDDEN": "remove-cell",  # Remove the whole cell
        "# OUTPUT_ONLY": "remove-input",  # Remove only the input
        "# COLLAPSED": "hide-input",  # Hide the input w/ a button to show
    }

    print_yellow = lambda x: print(f"\033[33m{x}\033[0m")  # print in yellow

    # Search through each notebook and look for the text, add a tag if necessary
    any_missing = False
    for ipath in notebooks:
        ntbk = nbf.read(ipath, nbf.NO_CONVERT)

        missing_metadata = False
        for cell in ntbk.cells:
            cell_tags = cell.get("metadata", {}).get("tags", [])
            for key, val in text_search_dict.items():
                if key in cell["source"]:
                    if val not in cell_tags:
                        missing_metadata = True
                        cell_tags.append(val)
            if len(cell_tags) > 0:
                cell["metadata"]["tags"] = cell_tags
        if missing_metadata:
            if check:
                print_yellow(f"{ipath} is missing metadata. Call `python scripts.docs.add_nb_tags.py` to add it.")
                any_missing = True
            else:
                print(f"Adding metadata to {ipath}")
                nbf.write(ntbk, ipath)

    if not any_missing:
        print("All notebooks have correct metadata.")

    if check and any_missing:
        sys.exit(1)


if __name__ == "__main__":
    dcargs.cli(main)
