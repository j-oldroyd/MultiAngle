import numpy as np
import numpy.typing as npt
from numpy import linalg as LA

import networkx as nx

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def mb_etf(d):
    """Construct the (N, d) Mercedes-Benz ETF."""
    G = (d + 1)/d * np.eye(d + 1) - (1/d) * np.ones(d + 1)
    return get_synthesis(G)


def get_synthesis(G, tol=1e-6):
    r"""Convert a Gram matrix to synthesis matrix.

    Given a Gram matrix $G$, this function returns a corresponding
    synthesis matrix $F$ using diagonalization.  The matrix $F$ will
    be a matrix satisfying $F^T F = G$.  In general, there are always
    infinitely many such $F$ depending on how the diagonalization is
    carried out, but all such matrices will be unitarily equivalent to
    each other.

    Arguments:

        G: An $n\times n$ positive semi-definite matrix of rank $d$
        represented as an ndarray.

        tol: A tolerance setting for rounding eigenvalues to zero.

    Returns:

        F: A $d\times N$ synthesis matrix with Gram matrix $G$,
        represented as an ndarray.

    """
    if type(G) is not np.ndarray:
        try:
            G = np.array(G, dtype=float)
        except ValueError:
            print("G must be a NumPy array or an array-like of floats.")
            raise

    if G.shape[0] is not G.shape[1]:
        raise ValueError("G must be a square array.")

    N = G.shape[0]
    d = LA.matrix_rank(G)

    D, U = LA.eigh(G)
    D[np.abs(D) < tol] = 0
    if np.min(D) < 0:
        raise ValueError("G must be positive semi-definite.")

    D = np.diag(D)
    F = U @ np.sqrt(D)

    return F[:, (N-d):].T


class MultiAngle:
    """A class for creating and working with multiangle tight frames."""

    def __init__(self, mat=None, synthesis_mat=False):
        """Initialize instance of a MultiAngle object.

        MultiAngle objects represent frames of N vectors in R^d. The default
        representative is chosen to be the Gram matrix of the frame since this
        is invariant under orthogonal transformations. The object can also be
        initialized with a synthesis matrix instead.

        Keyword arguments:

            synthesis_mat: boolean. Use synthesis matrix instead of a
            Gram matrix.

        """
        if mat is None:
            mat = 3/2 * np.eye(3) - 1/2 * np.ones(3)

        self.matrix = mat
        self.kwds = synthesis_mat

        self.num_vectors = mat.shape[1]
        self.dim = LA.matrix_rank(mat)

        if synthesis_mat is False:
            self.gram = mat
            self.synthesis_matrix = get_synthesis(mat)
        else:
            self.synthesis_matrix = mat
            self.gram = self.synthesis_matrix.T @ self.synthesis_matrix

        self.analysis = self.synthesis_matrix.T
        self.frame_matrix = self.synthesis_matrix @ self.synthesis_matrix.T
        self.frame_bounds = []
        self.max_coherence = np.abs(np.triu(self.gram, k=1)).max()

    def get_angles(self):
        """Get list of angles from MultiAngle object."""

        angles = \
            np.unique(self.gram[np.triu_indices(np.shape(self.gram)[0], 1)])
        return angles

    def incidence_matrices(self, only_incidence=True, zip_angles=False):
        """Return incidence matrices associated with angles of a frame.

        Keyword arguments:
        only_incidence: boolean. Include only incidence matrices.
        zip_angles: boolean. Zip angles with incidence matrices.
        """
        angles = self.get_angles()
        num_angles = angles.size
        angle_mats = np.zeros((num_angles, self.num_vectors, self.num_vectors))

        if only_incidence:
            # We only care about the incidence matrices, not the angles
            # themselves.
            for idx, angle in enumerate(angles):
                # Produce incidence matrices associated to each angle
                angle_mat = np.where(self.gram == angle, 1, 0)
                np.fill_diagonal(angle_mat, 0)
                angle_mats[idx, :, :] = angle_mat
            # If zip_angles is true, we zip angles with their corresponding
            # incidence matrices.
            if zip_angles:
                # We set strict=True since the length of angles and angle_mats
                # should be the same. Therefore, we want an error if this is
                # not the case.
                angle_mats = zip(angles, angle_mats, strict=True)
        elif not only_incidence:
            # Now we include the angles as well.
            for idx, angle in enumerate(angles):
                # Produce matrices associated to each angle
                angle_mat = np.where(self.gram == angle, self.gram, 0)
                np.fill_diagonal(angle_mat, 0)
                angle_mats[idx, :, :] = angle_mat

        return tuple(angle_mats)

    def get_graphs(self):
        """Produce graphs associated with frame angles."""
        adjacency_mats = self.incidence_matrices()
        graphs = []
        for mat in adjacency_mats:
            graphs.append(nx.from_numpy_array(mat))

        return graphs

    def draw_graphs(self, single_plot=True):
        """Draw graphs corresponding to incidence matrices."""
        graphs = self.get_graphs()
        colors = list(mcolors.TABLEAU_COLORS)
        if single_plot:
            for idx, graph in enumerate(graphs):
                nx.draw_circular(graph, edge_color=colors[idx % 10],
                                 node_size=125, with_labels=True)
        else:
            # We want to create a subplot that minimizes the number of blank
            # entries in a row
            n_graphs = len(graphs)
            divisors = tuple(range(2, np.sqrt(n_graphs).astype(int) + 1))
            empty_spaces = np.array([d - n_graphs % d for d in divisors])
            n_rows = divisors[np.argmax(empty_spaces == min(empty_spaces))]
            n_cols = np.ceil(n_graphs / n_rows).astype(int)

            fig, axes = plt.subplots(n_rows, n_cols)
            ax = axes.flat
            colors = list(mcolors.TABLEAU_COLORS)
            for idx, graph in enumerate(graphs):
                nx.draw_circular(graph, ax=ax[idx],
                                 edge_color=colors[idx % 10], node_size=125,
                                 with_labels=True)
            for idx in range(n_graphs, n_rows * n_cols):
                # Fill in empty subplots with empty graphs.
                nx.draw_circular(nx.empty_graph(0), ax=ax[idx])

        plt.show()

    def __str__(self):
        return f'A frame of {self.num_vectors} vectors in R^{self.dim}.'

    def __repr__(self):
        return f'MultiAngle({repr(self.matrix)}, synthesis_mat={self.kwds})'
