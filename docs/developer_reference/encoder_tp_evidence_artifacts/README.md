# Encoder TP Evidence Artifacts

This directory mirrors the text artifacts collected under
`/data/encoder_tp_evidence_20260526` for the Encoder TP evidence branch.

Main entry points:

- `../encoder_tp_performance_report.md`: full memory / latency / accuracy / E2E
  report.
- `../encoder_tp_pr_draft.md`: PR-ready summary and wording constraints.
- `raw/`: copied text artifacts, including server logs, probe summaries,
  `results.jsonl`, GPU sample JSONL files, quality benchmark outputs, and
  memory/timing mark logs.
- `included_text_artifacts.txt`: manifest of text artifacts copied into `raw/`.
- `skipped_binary_artifacts.tsv`: manifest of large binary/media artifacts that
  remain on the experiment host and were not committed to Git.

Large binary files are intentionally excluded from this branch. The excluded
set includes generated preprocessed video pickle files, long WAV/MP4 media, and
tensor parity pickle dumps. Several exceed GitHub's normal single-file size
limit and are not needed to review the measured results because the reports
reference the copied text summaries and logs.

This branch is an evidence archive, not the final merge branch.
