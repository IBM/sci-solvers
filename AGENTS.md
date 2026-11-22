# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Never

- Reformat or restyle existing code unless the task explicitly asks for it.
- Add features, abstractions, or error handling beyond what was requested.
- Remove pre-existing dead code unless asked.
- Add `__init__.py` re-exports unless required for backwards compatibility or explicitly requested; if required, use PEP 562 lazy `__getattr__`, and declare the same names in a `TYPE_CHECKING` block so linters and type checkers can resolve them.
- Silently drop a per-call solver kwarg the solver cannot honor; reject it at the top of `sci_solver_callable_eigen_states` with a `reject_*` guard (see `reject_init_vectors`) and advertise support with a capability flag.
- Copy or translate code from GPL programs into this repo. Implement from papers and file formats, and cite the paper in a docstring; reading another program only to learn a file's layout or a sign convention is fine, but write the code from the equation. GPL code lives in its own repo and is called as an external program.
- Reference issues, PRs or plans in code comments or docs; describe the mechanism instead.

## Workflow

- **Think before coding.** State assumptions explicitly. If multiple interpretations exist, present them. If something is unclear, stop and ask.
- **Simplicity first.** Minimum code that solves the problem. If you write 200 lines and it could be 50, rewrite it.
- **Surgical changes only.** Touch only what you must. Match existing style. Every changed line should trace directly to the user's request.

## Pull Requests

- Use `.github/PULL_REQUEST_TEMPLATE.md` for every PR an agent opens.
- Keep the body short, plain-language, and easy to scan. Start with a human-written, nontechnical goal. Avoid LLM-written summaries, rather put the user prompt verbatim. Add only brief, useful high-level gotchas. Do not include code line references or links to specific commits unless absolutely essential.
- Before opening a PR, ask a human to write the goal sentence, show them the complete proposed body, and get their explicit approval of that exact body. Wait for approval before creating the PR.
- Commit messages follow [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) (`feat:` / `fix:` / `test:` / …), as the template's checklist requires; PRs are squash-merged.

## Build & Test Commands

**Package Manager**: `uv` (not pip/poetry)
- Install: `uv sync`
- Run scripts: `uv run <command>` (e.g., `uv run pytest`)
- C++ kernel (truncated Davidson `backend="cpp"`): `uv sync` builds it once, but the editable `.pth` install does NOT recompile on `kernels.cpp` edits — after editing it run `uv pip install -e . --reinstall-package sci_solvers` to rebuild.

**Testing**:
- Run all tests: `uv run pytest -v`
- Skip slow tests: `uv run pytest -v -m "not slow"`
- Tests marked `@pytest.mark.slow` take significant time

**Linting/Formatting**:
- Lint: `uv run ruff check --fix`
- Format: `uv run ruff format`
- Type check: `uv run mypy` (scope and per-module overrides in `pyproject.toml`)

## Code Style (Non-Obvious Rules)

**Python Version**: Python 3.12+. Built-in generics (`list[int]`, `X | None`).

**Line Length**: 100 characters (not 88 like Black).

**Type Hints**: Required on all signatures.

**Imports**: Ruff enforces import sorting (I rule); use `from typing import` not `from typing_extensions` unless needed.

**Docstrings**: Required for all public functions/classes/methods (D101-D107 rules enforced). Docstrings should declare intent, not just behavior. For algorithms, include the math in a `Notes` section (NumPy-style).

**Naming**:
- Ruff enforces PEP8 naming (N801-N806, N816).
- Use `pathlib.Path` not `os.path` (PTH rule enforced).
- Physical units in names: `coupling_mhz`, `duration_ns`, `n_qubits`.

**Ignore F722**: Ruff configured to ignore F722 (forward annotation syntax errors).

**Complex-dtype numerics**: when making a `.dtype`-derived numeric path complex-capable, audit every plain inner product (`torch.dot`, `a * b` used as `⟨a|b⟩`) for the real-vs-Hermitian conjugate-linearity distinction, and every `x.pow(2)` used as an occupation/probability weight for the `x.abs().pow(2)` (`|x|²`, not `x²`) fix — both are silent no-ops on the real path, so they're easy to miss until a complex caller shows up.

**Copyright Headers**: Every `.py` file carries the IBM copyright banner as a
`__copyright__ = """..."""` assignment placed immediately after the module docstring (see
`LICENSE.tmpl` for the text).

## Project-Specific Patterns

**CLI Entry Points**: Defined in `pyproject.toml` under `[project.scripts]`.

**Package Data**: Resource files under `src/sci_solvers/` must be listed in `pyproject.toml` under `[tool.setuptools.package-data]`.

## Updating this file

If you discover a project-wide rule, constraint, or failure mode while working on a task, propose adding it to the relevant section in the same PR. Keep entries to one line. Prefer `Never` for things that should never happen; `Conventions` for stylistic norms.
