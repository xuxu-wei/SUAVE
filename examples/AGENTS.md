# 指南

1. 当 `research-mimic_mortality_supervised.py` 和 `research-mimic_mortality_optimize.py` 代码更新时，请在 `docs/research_protocol.md` 的对应章节记录更新内容。
2. 当 `research-mimic_mortality_optimize.py`、`research-mimic_mortality_supervised.py`、`mimic_mortality_utils.py`、`cls_eval.py` 发生更新时，将改动同步到 `/research_template` 目录的对应脚本中。其中 `mimic_mortality_utils.py` 的函数更新需同步至 `analysis_utils.py`，需要暴露给用户修改的变量同步到 `analysis_config.py`。
3. 在 `/research_template` 目录的脚本发生改动后，请将改动同步记录到 `/research_template/RESEARCH_README.md`。
