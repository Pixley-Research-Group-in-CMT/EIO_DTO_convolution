#!/usr/bin/env python3
"""Plot fixed-field energy-integrated SWT S_mu(q) maps for mu=1,2,3."""

from swt_bz_maps import build_parser, compute_and_plot


parser = build_parser(
    "static",
    "Compute the fixed six-condition static SWT scan for mu=1,2,3.",
)

if __name__ == "__main__":
    compute_and_plot("static", parser.parse_args())
