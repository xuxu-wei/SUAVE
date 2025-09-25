# Coding Agent Behavior for SUAVE

## Golden Rules
1) **API stability > feature breadth**: keep `SUAVE` methods stable (`fit/predict/predict_proba/calibrate/encode/sample/save/load`) unless indicated.
2) **Small steps, small diffs**: each request updates as few files as possible; include docstrings and type hints.
3) **Tests & style**: add/maintain pytest for each new module; keep `black` and `ruff` clean.

## Documentation
- Each public method must include an example code snippet.
- Keep examples minimal and data-directory-centric (schema in the same folder as the CSV).
- When example scripts change their research workflows, update `docs/research_protocol.md` in the same PR to stay aligned.
- Public APIs must use pandas-style docstrings that include an `Example` section. For APIs related to evaluation, provide generally applicable or empirical benchmark thresholds within the docstring commentary.
- Add inline comments that explain the intent of each critical step in function bodies.
