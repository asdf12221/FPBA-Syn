# Contributing

Please keep new functionality dataset-agnostic. Put paths, category ids and prompts in configuration files rather than in Python modules, and add a small CPU-side validation path when possible.

Before opening a pull request:

```bash
python -m compileall fpba_syn
fpba-syn check --config configs/example.yaml --skip-models
```

Do not commit model weights, private datasets, generated images or machine-specific absolute paths.

