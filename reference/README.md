# Regression references

`eio/` contains the recovered nk24 surface-state summary. The EIO generator compares momenta, energies, band indices, radial velocities, and phase-aligned projected spinors against it.

`transport/` contains the archived unsmoothed resistivity CSVs. They are the
traceable baseline for the clean static pipeline. The out-of-plane reference is
the direct 4x4 result and does not include the questionable larger-system
zero-field substitution from the final historical plotting notebook.

The old files named `rho_*_smooth.csv` are intentionally not included. They do not follow from the saved Gaussian-smoothed matrix elements and appear to have been produced by a different hand-edited notebook state.
