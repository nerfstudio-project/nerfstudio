# Contributing to pyRad workflow

1. Make your modifications
2. Perform local checks
   
   To ensure that you will be passing all tests and checks on github, you will need to run the following command:

   ```
   python scripts/debugging/run_actions.py
   ```

   This will perform the following checks and actions:
   - Black/ Linting style check: Ensure that the code is consistently and properly formatted. 
   - Pytests: Runs pytests locally to make sure the logic of any added code does not break existing logic.
   - Documentation build: Builds the documentation locally to make sure none of the changes result in any warnings or errors in the docs.
   - Licensing: automatically adds licensing headers to the correct files.
  
  In order to merge changes to the code base, all of these checks must be passing. If you pass these tests locally, you will likely pass on github servers as well (results in a green checkmark next to your commit).

3. Open pull request