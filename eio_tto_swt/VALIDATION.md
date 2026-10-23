# Package validation

Validated on October 23, 2026, with Python 3.13 on macOS.

- 33 packaged tests passed: component contraction, field/angle validation, EIO model selection, resistivity layouts, GIF encoding, and continuation through unstable kernel frames.
- Generated the symmetrized EIO nk24 surface states from the bundled model: 576 momenta, 15 retained surface states, maximum Hermiticity error 1.69e-16, maximum eigenpair residual 4.87e-15.
- Ran the transport driver through DSSF, scattering, relaxation times, resistivity, and PDF generation with those EIO states. Reduced smoke configuration: fields 1.0 and 1.1, nominal angles 0, 15, 30 degrees plus 0.1-degree evaluation offset, internal momentum grid 2 cubed, slice projection, total response. The PDF contains no image objects.
- Ran the full path-movie calculation at fields 1.0 and 1.1, angles 0.1 and 30.1 degrees, two points per segment, eight energies, internal grid 2 cubed, total response, zero transverse/quadratic field couplings, and 30 DPI. Four GIFs and the HDF5 output were written.
- Ran the full kernel-movie calculation with the generated EIO states at fields 1.0 and 1.1, angles 0 and 30 degrees, internal grid 2 cubed, slice projection, and 30 DPI. The GIF and HDF5 output were written.

These checks establish that the packaged workflows execute. The reduced grids are not convergence tests, production results, or a reproduction of the note's figures. Numerical smoke outputs are not included in this source distribution. Cluster submission and optional MP4 encoding were not exercised.

## Optional kernels below resistivity

The package also validates reconstruction from saved pair scattering, conservation of the grouped kernel sum, selection of nominal angles and reporting of their actual offsets, refusal to substitute reflected-only angles, map placement beneath static rows and overlays, shared scales within a field, synchronized GIF frames, and replot overrides of saved preferences. A complete six-condition transport smoke run with the option enabled produced the combined PDF; combined GIF encoding was also exercised. Static export explicitly disables rasterization of all plot collections.
