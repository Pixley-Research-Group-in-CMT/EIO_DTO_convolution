#!/usr/bin/env python3
"""Plot fixed-field SWT S_mu(q, omega_0) maps for mu=1,2,3."""

from swt_bz_maps import build_parser, compute_and_plot


parser = build_parser(
    "dynamic",
    "Compute the fixed six-condition SWT structure-factor scan for mu=1,2,3.",
)

if __name__ == "__main__":
    compute_and_plot("dynamic", parser.parse_args())
