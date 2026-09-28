# AI_USAGE.md

## Which AI tools were used

Claude (Anthropic), used interactively in a chat session, was used
throughout the development of this project.

## What it was used for

- Generating the initial project architecture, dataset schema, and target
  definition proposal.
- Writing the synthetic dataset generator (`src/generate_dataset.py`),
  including the weighted-scoring target formula and its documentation.
- Writing the EDA script/notebook, preprocessing pipeline, training,
  tuning, evaluation, explainability, failure-analysis, and CSV-validation
  code (all files under `src/`).
- Writing the Streamlit app (`app/app.py`).
- Writing the pytest suite (`tests/`).
- Writing this README and the final report.
- Running the code in a sandboxed environment to produce every metric,
  chart, and number reported in this repository (no result was invented
  or estimated by the AI without executing the underlying code).

## Which code was AI-assisted

Essentially all code in this repository was AI-assisted (written by
Claude, reviewed and directed turn-by-turn by the project owner). This is
stated plainly and is not presented as personally hand-written from
scratch.

## Decisions the project owner made directly

- Chose Logistic Regression over tuned Random Forest as the final model
  after being shown the real test-set comparison (Random Forest
  underperformed even after tuning) -- this was an explicit choice
  presented as a question, not an automatic AI decision.
- Directed the incremental, step-by-step build order and requested
  explanations at each stage, matching the brief's requirement that the
  approach be personally understandable for a viva.

## How the code/results were manually verified

- Every model metric in `reports/evaluation_report.md` and
  `reports/failure_analysis.md` was produced by actually running the
  training/evaluation scripts against the held-out test set inside the
  development sandbox -- not estimated or written by hand.
- The dataset's class balance and feature/target relationships were
  checked visually via the EDA figures (`reports/figures/`) before
  proceeding to modeling.
- The CSV-validation logic (`src/csv_validation.py`) was stress-tested
  with deliberately messy inputs (bad text, missing values, missing
  columns, text yes/no, extra columns) and the pytest suite
  (`tests/`) was run to confirm all 19 tests pass -- including one real
  bug that was caught and fixed during this process (a pandas-version
  dtype check that silently skipped normalizing "Yes"/"No" internship
  values; documented in `src/csv_validation.py`).
- **Before submission/demo, the project owner should personally re-run
  every command in the README's "How to Run" section** and be prepared
  to explain: the target-generation formula, why Logistic Regression was
  chosen as the final model, what the preprocessing pipeline does and why
  it's fit only on training data, what each evaluation metric means, and
  how the what-if simulator computes its score -- all of which are
  explained in plain language in the corresponding source files and
  `reports/` documents specifically so this is possible.

## What was NOT done

- No metric, confusion matrix, or chart in this repository was fabricated
  or hand-written without running the actual code -- this was a hard
  constraint throughout the project (see the brief's own requirement).
- No claim is made that the code was written entirely without AI
  assistance -- the opposite is true, as documented above.
