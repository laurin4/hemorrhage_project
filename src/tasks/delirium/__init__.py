"""
Delirium task (legacy report-centric pipeline).

Implementation remains in original locations for reproducibility:

- ``src.pipeline.run_pipeline`` — report-level inference
- ``src.agents.*`` — delirium agents + guardrails
- ``src.preprocessing.evidence_extraction`` — delirium prefilter
- ``src.pipeline.prepare_structured_data`` — ICD / ICDSC baseline

See ``BOUNDARIES.md`` for isolated delirium-specific surface area.
"""
