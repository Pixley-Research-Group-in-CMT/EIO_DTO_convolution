# DTO snapshot input

The repository supports two snapshot files:

```text
gathered_snapshots_hex.hdf5                 # 8x8 in-plane angular scan
gathered_snapshots_out_of_plane_hex.hdf5    # 4x4 out-of-plane field scan
```

Place or link Zhengtao's files here. A nondefault path can also be supplied to
the Fourier stage:

```bash
python 02_compute_dto_fourier.py --snapshots /path/to/gathered_snapshots_hex.hdf5
python 02_compute_dto_fourier.py --scan out-of-plane \
    --snapshots /path/to/gathered_snapshots_out_of_plane_hex.hdf5
```
