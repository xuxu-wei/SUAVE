# Coding Agent Behavior for SUAVE

## Golden Rules
1) **API stability > feature breadth**: keep `SUAVE` methods stable (`fit/predict/predict_proba/calibrate/encode/sample/save/load`) unless indicated.
2) **No hidden state**: every training- or data-dependent default is explicit in arguments or saved alongside the model.
3) **Small steps, small diffs**: each request updates as few files as possible; include docstrings and type hints.
4) **Tests & style**: add/maintain pytest for each new module; keep `black` and `ruff` clean.

## Documentation
- Each public method must include an example code snippet.
- Keep examples minimal and data-directory-centric (schema in the same folder as the CSV).
- When example scripts change their research workflows, update `docs/research_protocol.md` in the same PR to stay aligned.
