"""Minimal zero-field EIO tight-binding and [111] slab utilities."""
import numpy as np


def load_bulk_model(cell_indices_path, hopping_path, wannier_centres_path):
    """Read the real-space Wannier Hamiltonian and its orbital geometry.

    The returned hopping tensor is indexed by source orbital, target orbital,
    and lattice translation. Its last input axis stores real and imaginary
    parts separately.
    """
    lattice_vectors = 5.1350 * np.array([
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 1],
    ])
    wannier_centres = np.loadtxt(wannier_centres_path, dtype=float)
    cell_indices = np.loadtxt(cell_indices_path, delimiter=",", dtype=int)
    hopping_components = np.loadtxt(hopping_path, delimiter=",", dtype=float)

    # Reconstruct the complex hopping tensor written by the upstream
    # DFT/Wannier workflow.
    n_orbitals = len(wannier_centres)
    hopping_components = hopping_components.reshape(
        n_orbitals, n_orbitals, len(cell_indices), 2
    )
    hoppings = hopping_components[..., 0] + 1j * hopping_components[..., 1]
    return {
        "lattice_vectors": lattice_vectors,
        "wannier_centres": wannier_centres,
        "cell_indices": cell_indices,
        "hoppings": hoppings,
    }


def truncate_hoppings(bulk, cutoff):
    """Remove hopping matrix elements whose Wannier-centre distance exceeds cutoff."""
    lattice_vectors = bulk["lattice_vectors"]
    wannier_centres = bulk["wannier_centres"]
    cell_indices = bulk["cell_indices"]
    hoppings = bulk["hoppings"]

    # Keep the archived row/column distance indexing. It may be transposed and is
    # therefore documented as a convention to test, rather than silently changed.
    centre_displacements = (
        wannier_centres[:, None, :] - wannier_centres[None, :, :]
    )
    retained_cells = []
    retained_hoppings = []
    for cell, hopping in zip(cell_indices, np.moveaxis(hoppings, 2, 0)):
        translation = cell @ lattice_vectors
        distances = np.linalg.norm(centre_displacements + translation, axis=2)
        if distances.min() <= cutoff:
            retained_cells.append(cell)
            retained_hoppings.append(np.where(distances < cutoff, hopping, 0))

    return {
        "lattice_vectors": lattice_vectors,
        "wannier_centres": wannier_centres,
        "cell_indices": np.asarray(retained_cells),
        "hoppings": np.asarray(retained_hoppings),
    }


def build_slab(bulk):
    """Group bulk hoppings by separation between adjacent [111] layers."""
    lattice_vectors = bulk["lattice_vectors"]
    cell_indices = bulk["cell_indices"]
    hoppings = bulk["hoppings"]
    # A bulk translation changes the [111] layer index by -(n1+n2+n3).
    # Only offsets up to two layers survive the real-space hopping cutoff.
    layer_offsets = -np.sum(cell_indices, axis=1)

    slab_hoppings = {}
    slab_vectors = {}
    for offset in (0, 1, -1, 2, -2):
        selected = layer_offsets == offset
        slab_hoppings[offset] = hoppings[selected]
        slab_vectors[offset] = cell_indices[selected] @ lattice_vectors
    return {"hoppings": slab_hoppings, "vectors": slab_vectors}


def transform_to_slab_coordinates(bulk, slab):
    """Rotate real and reciprocal vectors into the [111] slab frame."""
    lattice_vectors = bulk["lattice_vectors"]
    in_plane_vectors = np.empty((3, 3))
    in_plane_vectors[0] = lattice_vectors[1] - lattice_vectors[0]
    in_plane_vectors[1] = lattice_vectors[2] - lattice_vectors[0]
    in_plane_vectors[2] = np.cross(in_plane_vectors[0], in_plane_vectors[1])
    reciprocal_vectors = 2 * np.pi * np.linalg.inv(in_plane_vectors)

    # Define Cartesian x and y in the interface plane; z is the [111] normal.
    x_axis = in_plane_vectors[0] / np.linalg.norm(in_plane_vectors[0])
    y_axis = in_plane_vectors[1] - np.dot(in_plane_vectors[1], x_axis) * x_axis
    y_axis /= np.linalg.norm(y_axis)
    rotation = np.array([x_axis, y_axis, np.cross(x_axis, y_axis)])

    return {
        "reciprocal_vectors": rotation @ reciprocal_vectors,
        "slab_vectors": {
            offset: np.einsum("ij,rj->ri", rotation, vectors)
            for offset, vectors in slab["vectors"].items()
        },
        "wannier_centres": np.einsum(
            "ij,rj->ri", rotation, bulk["wannier_centres"]
        ),
    }


def _inside_polygon(point, vertices):
    """Return whether a 2D momentum lies inside the hexagonal Brillouin zone."""
    x, y = point
    inside = False
    for i, (x1, y1) in enumerate(vertices):
        x2, y2 = vertices[(i + 1) % len(vertices)]
        if y > min(y1, y2) and y <= max(y1, y2) and x <= max(x1, x2):
            if y1 != y2:
                intersection = (y - y1) * (x2 - x1) / (y2 - y1) + x1
                if x1 == x2 or x <= intersection:
                    inside = not inside
    return inside


def _select_hexagonal_cell(reciprocal_vectors, unit_cell_grid):
    """Fold a parallelogram momentum grid into one hexagonal unit cell."""
    reciprocal_length = np.linalg.norm(reciprocal_vectors[:, 0])
    radius = reciprocal_length / np.sqrt(3)
    eps = 1e-9
    vertices = np.array([
        [radius + 2 * eps, 0],
        [radius / 2 + eps, radius * np.sqrt(3) / 2 + eps],
        [-radius / 2 - eps, radius * np.sqrt(3) / 2 + eps],
        [-radius - 2 * eps, 0],
        [-radius / 2 - eps, -radius * np.sqrt(3) / 2 + eps],
        [radius / 2 + eps, -radius * np.sqrt(3) / 2 + eps],
    ])

    # Include neighboring reciprocal cells before clipping so that the final
    # hexagon contains the correct number of points near every boundary.
    shifted_grids = []
    for i, j in ((0, 0), (-1, 0), (0, -1), (-1, -1)):
        shift = i * reciprocal_vectors[:, 0] + j * reciprocal_vectors[:, 1]
        shifted_grids.append(unit_cell_grid + shift)
    candidates = np.concatenate(shifted_grids)
    return np.asarray([
        momentum for momentum in candidates
        if _inside_polygon(momentum[:2], vertices)
    ])


def hexagonal_momentum_grid(reciprocal_vectors, nk):
    """Generate the archived nk-by-nk first-Brillouin-zone sampling."""
    fractional = np.linspace(0, 1, nk + 1)[:-1]
    k0, k1 = np.meshgrid(fractional, fractional)
    unit_cell_grid = np.empty((nk, nk, 3))
    for component in range(3):
        unit_cell_grid[..., component] = (
            k0 * reciprocal_vectors[component, 0]
            + k1 * reciprocal_vectors[component, 1]
        )

    hexagonal_grid = _select_hexagonal_cell(
        reciprocal_vectors, unit_cell_grid.reshape(-1, 3)
    )
    neighboring_grids = []
    for i in (-1, 0, 1):
        j_values = (-1, 1) if i == 0 else (-1, 0, 1)
        for j in j_values:
            shift = i * reciprocal_vectors[:, 0] + j * reciprocal_vectors[:, 1]
            neighboring_grids.append(hexagonal_grid + shift)

    # Boundary points can enter the hexagon through more than one reciprocal
    # translation. Keep the historical positive-k representative once.
    repeated = []
    for momentum in np.concatenate(neighboring_grids):
        matches = np.where(
            np.linalg.norm(hexagonal_grid - momentum, axis=1) < 1e-5
        )[0]
        if len(matches) == 1 and momentum[0] > 0:
            repeated.append(matches[0])
    repeated = np.unique(np.asarray(repeated, dtype=int))
    return np.delete(hexagonal_grid, repeated, axis=0)


def slab_hamiltonian(momentum, slab_hoppings, slab_vectors, n_orbitals, n_layers):
    """Construct the open-boundary [111] slab Hamiltonian at one momentum."""
    # Fourier transform each group of real-space hoppings into a block that
    # connects layers separated by the corresponding offset.
    blocks = {}
    for offset, hoppings in slab_hoppings.items():
        block = np.zeros((n_orbitals, n_orbitals), dtype=complex)
        for hopping, vector in zip(hoppings, slab_vectors[offset]):
            block += hopping * np.exp(-1j * np.dot(momentum, vector))
        blocks[offset] = block

    # Assemble the block-banded slab Hamiltonian. Omitting targets outside the
    # layer range imposes open boundaries along [111].
    indices = np.arange(n_orbitals * n_layers).reshape(n_layers, n_orbitals).T
    hamiltonian = np.zeros(
        (n_orbitals * n_layers, n_orbitals * n_layers), dtype=complex
    )
    for source_layer in range(n_layers):
        source = indices[:, source_layer]
        for offset in (0, -1, -2, 1, 2):
            target_layer = source_layer + offset
            if 0 <= target_layer < n_layers:
                target = indices[:, target_layer]
                hamiltonian[np.ix_(target, source)] = blocks[offset]
    return hamiltonian
