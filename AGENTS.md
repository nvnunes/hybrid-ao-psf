# AGENTS.md

## Scope

- Documentation surface profile: public-python.

## Source Of Truth Docs

- Follow [`README.md`](README.md) for the public summary and starting docs.
- Follow `docs/architecture.md` for package structure, public API boundaries,
  scientific ownership, and persisted-contract rules.
- Follow `docs/testing.md` for canonical verification commands and completion
  expectations.
- Follow `docs/development.md` for bootstrap, local environment setup, and Git
  hook activation.

## Skill Requirements

- For Python code, use `$python-code-writing`.
- For project docs under `docs/`, use `$project-docs-writing`.
- For `README.md`, use `$readme-writing`.
- For plan documents or phased execution docs, use `$plan-writing`.

## Working Rules

- Consult `docs/architecture.md` before changing public contracts, scientific
  ownership, interpolation artifacts, HDF5 schemas, or execution lifecycle.
- Preserve scientific behavior unless a change is explicitly approved and
  verified against the project-owned regression baselines.
- Before concluding substantial work, satisfy `docs/testing.md`.
- Use the repository-local `./.conda` environment and the workflow in
  `docs/development.md` unless a task explicitly requires another environment.
