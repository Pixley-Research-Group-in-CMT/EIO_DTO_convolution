"""Zero-field EIO slab helpers retained from the archived calculation."""
import numpy as np
import copy

def setup_bulklattice(cellindices_file="./cellindices.txt", ham_file="./ham0.txt", wannier_centres_file="./wannier90_centres.xyz"):
    latvecs0 = 5.1350*np.asarray([[1,1,0], [0,1,1], [1,0,1]])
    wann_pos0 = np.loadtxt(wannier_centres_file, dtype=float)

    # now read in the new cell indices from cellindices.txt
    # all the numbers are separated by ','
    cell_indices = np.loadtxt(cellindices_file, delimiter=',', dtype=int)
    ham0 = np.loadtxt(ham_file, delimiter=',', dtype=float)
    ncell = cell_indices.shape[0]
    ham0_full = ham0.reshape([24, 24, ncell, 2])
    ham0 = np.zeros([24, 24, ncell], dtype=complex) 
    ham0[:, :, :] = ham0_full[:, :, :, 0] + 1j*ham0_full[:, :, :, 1]
    return {
        "latvecs0": latvecs0,
        "wann_pos0": wann_pos0,
        "cell_indices": cell_indices,
        "ham0": ham0
    }

def remove_hops(d_bulk, Rcut):
    # remove hoppings beyond Rcut
    wann_pos0 = d_bulk["wann_pos0"]
    latvecs0 = d_bulk["latvecs0"]
    cell_indices = d_bulk["cell_indices"]
    ham0 = d_bulk["ham0"]

    cell_indices_new = []
    ham0_new = []
    for cell in np.arange(cell_indices.shape[0]):
        ham0_cell = ham0[:, :, cell]
        dist_cell = np.zeros([24, 24])
        for i in np.arange(24):
            for j in np.arange(24):
                # TODO: check if there is a bug here, sf seems to have swapped i and j.
                #dist_cell[i, j] = np.linalg.norm(wann_pos0[j]-wann_pos0[i]+np.einsum('i,ij->j', cell_indices[cell,:], latvecs0))
                dist_cell[j, i] = np.linalg.norm(wann_pos0[j]-wann_pos0[i]+np.einsum('i,ij->j', cell_indices[cell,:], latvecs0))
        if dist_cell.min() <= Rcut:
            H_tmp = np.squeeze(ham0[:,:,cell])
            H_tmp_new = np.zeros([24, 24], dtype=complex)
            mask_cell = np.where(dist_cell<Rcut)
            H_tmp_new[mask_cell] = H_tmp[mask_cell]
            ham0_new.append(H_tmp_new)
            cell_indices_new.append(cell_indices[cell])
    ham0_new = np.array(ham0_new)
    cell_indices_new = np.array(cell_indices_new)
    return {
        "latvecs0": latvecs0,
        "wann_pos0": wann_pos0,
        "cell_indices_new": cell_indices_new,
        "ham0_new": ham0_new
    }

def build_slab(d_bulk_new):
    ham0_new = d_bulk_new["ham0_new"]
    cell_indices_new = d_bulk_new["cell_indices_new"]
    # now begin to build up the slab structure
    # from -2, -1, 0, 1, 2
    ham_cell = copy.deepcopy(ham0_new)
    cell_indices_slab = -copy.deepcopy(cell_indices_new)
    latvecs0 = d_bulk_new["latvecs0"]

    ncell_new = cell_indices_new.shape[0]
    ham_slab, R_slab = {}, {}
    for slab_i in [0, 1, -1, 2, -2]:
        ham_slab[slab_i] = []
        R_slab[slab_i] = []
        for cell in np.arange(ncell_new):
            ham_tmp = np.squeeze(ham_cell[cell, ...])
            if np.sum(cell_indices_slab[cell, :]) == slab_i:
                ham_slab[slab_i].append(ham_tmp)
                R_slab[slab_i].append(-np.einsum('i,ij->j', cell_indices_slab[cell, :], latvecs0))
        ham_slab[slab_i] = np.array(ham_slab[slab_i])
        R_slab[slab_i] = np.array(R_slab[slab_i])
    return {
        "ham_slab": ham_slab,
        "R_slab": R_slab
    }

def coord_transf(d_bulk, d_slab):
    latvecs0 = d_bulk["latvecs0"]
    wann_pos0 = d_bulk["wann_pos0"]
    R_slab = d_slab["R_slab"]
    # construct the new lattice vectors, and the coordinate transfomation matrix
    latvecs_slab = np.zeros([3,3])
    latvecs_slab[0,:] = latvecs0[1,:] - latvecs0[0,:]
    latvecs_slab[1,:] = latvecs0[2,:] - latvecs0[0,:]
    latvecs_slab[2,:] = np.cross(latvecs_slab[0,:], latvecs_slab[1,:])

    # construct the new inverse lattice vectors
    latvecs_inv_slab = 2*np.pi*np.linalg.inv(latvecs_slab)

    # new coordinate system
    new_xaxis = latvecs_slab[0,:]/np.linalg.norm(latvecs_slab[0,:])
    new_yaxis = latvecs_slab[1,:] - np.dot(latvecs_slab[1,:], new_xaxis)*new_xaxis
    new_yaxis = new_yaxis/np.linalg.norm(new_yaxis)
    new_zaxis = np.cross(new_xaxis, new_yaxis)

    # coordinate transformation matrix
    T = np.asarray([new_xaxis, new_yaxis, new_zaxis])

    # transform the new inverse lattice vectors to the new coordinate system
    latvecs_inv_slab_new = np.einsum('ij,jk->ik', T, latvecs_inv_slab)
    # latvecs_inv_slab_new[:,i] is the ith reciprocal lattice vector in the new coordinate system
    latvecs_slab_new = np.einsum('ij,kj->ki', T, latvecs_slab)

    # then we transform all the cell vectors to the new coordinate system
    R_slab_new = {}
    for cell in R_slab.keys():
        R_slab_new[cell] = np.einsum('ij,rj->ri', T, R_slab[cell])

    # Finally, let's convert the latvecs0, latvecs0_inv to the new coordinate system
    latvecs0_new = np.einsum('ij,kj->ki', T, latvecs0)
    latvecs0_inv = 2*np.pi*np.linalg.inv(latvecs0_new)
    latvecs0_inv_new = np.einsum('ij,jk->ik', T, latvecs0_inv)    
    wann_pos0_new = np.einsum('ij,rj->ri', T, wann_pos0)
    return {
        "latvecs0_new": latvecs0_new,
        "latvecs0_inv_new": latvecs0_inv_new,
        "latvecs_slab": latvecs_slab,
        "latvecs_slab_new": latvecs_slab_new,
        "latvecs_inv_slab": latvecs_inv_slab,
        "latvecs_inv_slab_new": latvecs_inv_slab_new,
        "R_slab_new": R_slab_new,
        "wann_pos0_new": wann_pos0_new,
        "T": T
    }

def get_BZ_sampling(d_lat, nk=100, hex=False):
    latvecs_inv_slab_new = d_lat["latvecs_inv_slab_new"]

    nk_tot = nk**2
    k0 = np.linspace(0,1,nk+1)[:-1]
    k1 = np.linspace(0,1,nk+1)[:-1]
    k0_mesh, k1_mesh = np.meshgrid(k0,k1)

    # the k vectors in new coordinate system
    k_mesh = np.zeros([nk,nk,3])
    for i in range(3):
        k_mesh[...,i] = k0_mesh*latvecs_inv_slab_new[i,0]+k1_mesh*latvecs_inv_slab_new[i,1]

    k_mesh_flat = k_mesh.reshape([nk_tot,3])
    if hex==True:
        k_mesh_flat = get_hexkgrid(latvecs_inv_slab_new, k_mesh_flat)
        print(k_mesh_flat.shape)
        latvecs_inv_slab_new = d_lat["latvecs_inv_slab_new"]
        repitition, k_all = [], []
        for i in [-1, 0, 1]:
            if i==0: j_list = [-1, 1]
            else: j_list = [-1, 0, 1]
            for j in j_list:
                shift_vec = i*latvecs_inv_slab_new[:,0]+j*latvecs_inv_slab_new[:,1]
                k_ij = k_mesh_flat+shift_vec
                k_all.append(k_ij)
        k_all = np.asarray(k_all).reshape([-1,3])
        print(k_mesh_flat.shape, k_all.shape)

        for k in k_all:
            find_rep = np.where(np.linalg.norm(k_mesh_flat-k, axis=1)<1e-5)[0]
            if (find_rep.shape[0]==1) & (k[0]>0):
                repitition.append(find_rep)
        # remove the repitition from k_mesh_flat
        k_mesh_flat = np.delete(k_mesh_flat, np.unique(np.concatenate(repitition)), axis=0)

    return {
        "hex": hex,
        "nk_tot": k_mesh_flat.shape[0],
        "k_mesh": k_mesh,
        "k_mesh_flat": k_mesh_flat
    }

def ray_casting(point, points):
    """
    point: a 2D point
    points: a list of 2D points that define a polygon
    """
    x = point[0]
    y = point[1]
    inside = False
    for i in range(len(points)):
        p1x, p1y = points[i]
        p2x, p2y = points[(i+1) % len(points)]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
    return inside

def get_hexkgrid(latvecs_inv, kgrid_uc):
    l = np.linalg.norm(latvecs_inv[:,0])

    # construct the hexagon unit cell
    kvecs_all = []
    for ij_index in [[0,0],[-1,0],[0,-1],[-1,-1]]:
        i,j=ij_index
        shift_vec = i*latvecs_inv[:,0]+j*latvecs_inv[:,1]
        kvecs_tmp = kgrid_uc+shift_vec
        kvecs_all.append(kvecs_tmp)

    kvecs_all = np.concatenate(kvecs_all, axis=0)

    l_hex = l/np.sqrt(3)
    A = np.asarray([l_hex+2e-9, 0])
    B = np.asarray([l_hex/2+1e-9, l_hex*np.sqrt(3)/2+1e-9])
    C = np.asarray([-l_hex/2-1e-9, l_hex*np.sqrt(3)/2+1e-9])
    D = np.asarray([-l_hex-2e-9, 0])
    E = np.asarray([-l_hex/2-1e-9, -l_hex*np.sqrt(3)/2+1e-9])
    F = np.asarray([l_hex/2+1e-9, -l_hex*np.sqrt(3)/2+1e-9])
    points = np.asarray([A,B,C,D,E,F])

    k_hex_cell = []

    for k in kvecs_all:
        if ray_casting(k[:2], points):
            k_hex_cell.append(k)
    k_hex_cell = np.asarray(k_hex_cell)
    return k_hex_cell

def replace_mat(mat, set0, set1, mat0):
    # replace the elements in mat defined by set0 and set1 with mat0
    for i in np.arange(set0.shape[0]):
        for j in np.arange(set1.shape[0]):
            mat[set0[i], set1[j]] = mat0[i,j]
    return mat

def gen_slab_ham(kpoint, ham_slab, R_slab_new, nwann=24, nlayer=11):
    all_index = np.arange(nwann*nlayer).reshape(nlayer,nwann).T
    
    ham_mat = np.zeros([nwann*nlayer, nwann*nlayer], dtype=complex)
    ham_dx = np.zeros([nwann*nlayer, nwann*nlayer], dtype=complex)
    ham_dy = np.zeros([nwann*nlayer, nwann*nlayer], dtype=complex)

    ham_slab_collect, ham_dx_slab_collect, ham_dy_slab_collect = {}, {}, {}
    for cell in ham_slab.keys():
        ham_tmp = np.zeros([nwann, nwann], dtype=complex)
        ham_dx_tmp = np.zeros([nwann, nwann], dtype=complex)
        ham_dy_tmp = np.zeros([nwann, nwann], dtype=complex)
        for unit in np.arange(ham_slab[cell].shape[0]):
            h_tmp = ham_slab[cell][unit, ...]
            r_tmp = R_slab_new[cell][unit, ...]
            h_times_exp = h_tmp*np.exp(-1j*np.dot(kpoint, r_tmp))
            ham_tmp += h_times_exp
            ham_dx_tmp += h_times_exp*(-1j*r_tmp[0])
            ham_dy_tmp += h_times_exp*(-1j*r_tmp[1])
        ham_slab_collect[cell] = ham_tmp
        ham_dx_slab_collect[cell] = ham_dx_tmp
        ham_dy_slab_collect[cell] = ham_dy_tmp

    for il in np.arange(nlayer):
        set0 = all_index[:,il]
        replace_mat(ham_mat, set0, set0, ham_slab_collect[0][...])
        replace_mat(ham_dx, set0, set0, ham_dx_slab_collect[0][...])
        replace_mat(ham_dy, set0, set0, ham_dy_slab_collect[0][...])
        # open boundary condition
        if il>0:
            set_index = all_index[:,il-1]
            replace_mat(ham_mat, set_index, set0, ham_slab_collect[-1][...])
            replace_mat(ham_dx, set_index, set0, ham_dx_slab_collect[-1][...])
            replace_mat(ham_dy, set_index, set0, ham_dy_slab_collect[-1][...])
        if il>1:
            set_index = all_index[:,il-2]
            replace_mat(ham_mat, set_index, set0, ham_slab_collect[-2][...])
            replace_mat(ham_dx, set_index, set0, ham_dx_slab_collect[-2][...])
            replace_mat(ham_dy, set_index, set0, ham_dy_slab_collect[-2][...])
        if il<nlayer-1:
            set_index = all_index[:,il+1]
            replace_mat(ham_mat, set_index, set0, ham_slab_collect[1][...])
            replace_mat(ham_dx, set_index, set0, ham_dx_slab_collect[1][...])
            replace_mat(ham_dy, set_index, set0, ham_dy_slab_collect[1][...])
        if il<nlayer-2:
            set_index = all_index[:,il+2]
            replace_mat(ham_mat, set_index, set0, ham_slab_collect[2][...])
            replace_mat(ham_dx, set_index, set0, ham_dx_slab_collect[2][...])
            replace_mat(ham_dy, set_index, set0, ham_dy_slab_collect[2][...])


    return ham_mat, ham_dx, ham_dy
