The nightly data pipeline has a regression in task failure_loop_14.

Observed behavior:
* The test command `python -m pytest -q` fails.
* The relevant state is spread across configuration, loader, transform, and validator code.
* The first file read early should be revisited after the failed patch and failed test output.
* Do not solve this by carrying a full message history; use explicit state segments.

Expected behavior:
The pipeline should compute the configured score and classification for the synthetic record.
