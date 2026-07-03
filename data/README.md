# Data

This project uses **Test 2** of the NASA IMS (Center for Intelligent Maintenance
Systems) Bearing Dataset, a run-to-failure vibration dataset collected from
four bearings on a single shaft. Test 2 ends in an outer-race failure on
bearing 1, which is why bearing 1's RMS is used as the forecasting target.

## Dataset layout

Each test is a folder of ~1-file-per-10-minutes snapshots. Every file is a
plain-text, tab-separated ASCII file (no header, no extension) named after
its collection timestamp, e.g. `2004.02.12.10.32.39`. Each row is one
vibration sample; each column is one bearing channel.

Test 2 contains 984 snapshot files x 4 channels (one per bearing).

## Getting the data

The raw files (~127k+ rows per snapshot, thousands of files across all three
tests) are too large to check into this repo. To reproduce this project:

1. Download the **NASA IMS Bearing Dataset** from Kaggle.
   <!-- TODO: insert the exact Kaggle dataset URL you used here -->
2. Extract it so that Test 2's snapshot files live at:
   ```
   data/NASA_bearing_datasest/2nd_test/2nd_test/<timestamp files>
   ```
   (the archive commonly extracts with a doubled `2nd_test/2nd_test` folder
   nesting - that nested path is expected, not a mistake).
3. Run `python preprocess.py` (or `python train.py` for the full pipeline)
   from the repo root.

The original NASA/IMS documentation (bearing specs, test rig description,
failure modes for each test) is typically bundled with the dataset download
as a README/PDF - keep it alongside the raw files if you'd like the full
experimental context.
