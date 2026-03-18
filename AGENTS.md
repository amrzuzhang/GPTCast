# GPTCast Project Rules

These rules apply to everything under `/home/ang/GPTCast` and refine the higher-level rules in `/home/ang/AGENTS.md`.

## Research Mainline

Current mainline is:

1. Stage 1 tokenizer: `vae_phuber_rzsm`
2. Stage 2 baseline: `soilcast_16x16_rzsm_hydro`
3. Stage 2 enhancement on route A: `soilcast_16x16_rzsm_hydro_static`

Do not silently replace this mainline with:

- weekly / subseasonal reformulations
- external-guidance hybrid variants without verified data
- homemade physics-loss branches
- new target variables that require redefining the task

If a change alters forecast horizon, temporal granularity, target variable, region, or evaluation protocol, label it explicitly as a **route change**, not as an optimization of the current project.

## Baseline First

For any new experiment:

1. keep a clean baseline runnable first
2. add one literature-supported enhancement second
3. compare them under the same data split and evaluation protocol

Do not recommend or launch an enhanced experiment before confirming the baseline is still runnable in the current code state.

## Mainline vs Exploratory

Every method change must be labeled as one of:

- `baseline`
- `mainline enhancement`
- `exploratory`

Exploratory work must not overwrite or replace mainline configs, training commands, or README guidance unless the user explicitly approves promotion.

## Literature Compatibility Check

Before adapting a paper method, verify all of the following against the current project:

1. target variable compatibility
2. temporal granularity compatibility
3. forecast horizon compatibility
4. spatial setting / region compatibility
5. required external data compatibility
6. official code/data availability

If the paper only partially matches the project, describe the result as `inspired by`, not `reproduced from`.

## Physics Claims

Use strict wording:

1. Validation-only `tf_phys_*` metrics are **physical monitoring**, not physical constraints.
2. A method can be called `physics-informed` only if the training objective or input design is directly grounded in the cited literature and actually participates in learning.
3. Do not claim that the model "satisfies physical formulas" unless this is explicitly implemented and verified.

## Current Project-Specific Constraints

1. Do not modify Stage 1 unless there is direct evidence that tokenizer quality is the bottleneck.
2. Do not revive the deleted homemade physics-loss branch as a mainline method.
3. Do not reintroduce ECMWF / GEFS guidance code unless usable guidance data has been obtained and the data chain is validated end-to-end.
4. For route A, prefer compatible static / geographic context over incompatible external weekly hybrid methods.

## Data Chain Gate

Do not say an experiment is "ready to train" until all three are verified:

1. required files exist on disk
2. datamodule can load a real sample
3. a minimal smoke test can run without path/schema errors

Code skeletons, downloader stubs, or archives with promising filenames are not enough.

## External Assets

Keep large external paper assets outside the repo by default.

Current convention:

- external archives and paper assets belong under `/home/ang/GPTCast_external/`

Before treating any external asset as training input, inspect its contents and determine whether it is:

- raw data
- preprocessed features
- model artifacts
- evaluation outputs

Do not assume an archive is usable guidance input based on naming alone.

## Server Sync Rule

Before giving a server-side command that depends on newly added files, first confirm that the required files have been synced.

If a server error is consistent with missing files or stale code, diagnose sync state before discussing method-level causes.

## Preferred Current Comparisons

For the present project stage, the preferred second-stage comparisons are:

1. `soilcast_16x16_rzsm_hydro` vs persistence
2. `soilcast_16x16_rzsm_hydro_static` vs `soilcast_16x16_rzsm_hydro`

Prefer strengthening these comparisons before proposing a new research branch.
