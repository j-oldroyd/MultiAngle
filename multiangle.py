import numpy as np
from numpy import linalg as LA
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def mb_etf(d):
    """Construct the (N, d) Mercedes-Benz ETF."""
    G = (d + 1) / d * np.eye(d + 1) - (1 / d) * np.ones(d + 1)
    return get_synthesis(G)


def get_synthesis(gram_mat, tol=1e-6):
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
    if type(gram_mat) is not np.ndarray:
        try:
            gram_mat = np.array(gram_mat, dtype=float)
        except ValueError:
            print("Input must be a NumPy array or an array-like of floats.")
            raise

    if gram_mat.shape[0] is not gram_mat.shape[1]:
        raise ValueError("G must be a square array.")

    frame_size = gram_mat.shape[0]
    frame_dim = LA.matrix_rank(gram_mat)

    eig_vals, eig_vecs = LA.eigh(gram_mat)
    eig_vals[np.abs(eig_vals) < tol] = 0
    if np.min(eig_vals) < 0:
        raise ValueError("G must be positive semi-definite.")

    eig_vals = np.diag(eig_vals)
    F = eig_vecs @ np.sqrt(eig_vals)

    return F[:, (frame_size - frame_dim) :].T


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
            mat = 3 / 2 * np.eye(3) - 1 / 2 * np.ones(3)
        elif type(mat) is int and mat > 0:
            d = mat
            mat = (d + 1) / d * np.eye(d + 1) - (1 / d) * np.ones(d + 1)

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

        self.analysis_matrix = self.synthesis_matrix.T
        self.frame_matrix = self.synthesis_matrix @ self.synthesis_matrix.T
        self.frame_bounds = []
        self.max_coherence = np.abs(np.triu(self.gram, k=1)).max()

    def get_angles(self, precision=3):
        r"""Get list of angles from MultiAngle object.

        Return the unique values of the inner products that appear in
        the Gram matrix of the frame (the "angles").  The values are
        by default rounded to three decimal places.

        Keyword arguments:

            precision: An integer representing the number of places to
              round the entries of the Gram matrix to.
        """
        gram_rounded = np.round(self.gram, decimals=precision)

        angles = np.unique(gram_rounded)
        return angles

    def incidence_matrices(self, only_incidence=True, zip_angles=False):
        r"""Return incidence matrices associated with angles of a frame.

        Keyword arguments:

            only_incidence: A boolean indicating whether or not to
              include only the incidence matrices or the relevant
              entries of the Gram matrix.

            zip_angles: A boolean indicating whether or not to zip the
              angles with their corresponding incidence matrices.

        """
        angles = self.get_angles()
        num_angles = angles.size
        angle_mats = np.zeros((num_angles, self.num_vectors, self.num_vectors))

        if only_incidence:
            for idx, angle in enumerate(angles):
                angle_mat = np.where(self.gram == angle, 1, 0)
                np.fill_diagonal(angle_mat, 0)
                angle_mats[idx, :, :] = angle_mat

            if zip_angles:
                # We set strict=True since the length of angles and angle_mats
                # should be the same. Therefore, we want an error if this is
                # not the case.
                angle_mats = zip(angles, angle_mats, strict=True)
        else:
            for idx, angle in enumerate(angles):
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
                nx.draw_circular(
                    graph, edge_color=colors[idx % 10], node_size=125, with_labels=True
                )
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
                nx.draw_circular(
                    graph,
                    ax=ax[idx],
                    edge_color=colors[idx % 10],
                    node_size=125,
                    with_labels=True,
                )

            for idx in range(n_graphs, n_rows * n_cols):
                # Fill in empty subplots with empty graphs.
                nx.draw_circular(nx.empty_graph(0), ax=ax[idx])

        plt.show()

    def __str__(self, scientific=False, N=3):
        """Return string representation of frame matrix.

        Be default this will return a MATLAB-esque representation of
        self.matrix.

        Keyword arguments:

            scientific: A boolean that determines whether or not
              scientific notation is used in the string output.

            N: An integer that represents the power used in the
              scientific notation representation.

        Resources

        ---------

        https://www.reddit.com/r/Numpy/comments/16pb8uo/prettyprint_array_matlabstyle/.

        """
        if not scientific:
            np.set_printoptions(formatter={"float": lambda x: f"{x:10.4g}"})
            matrix_str = f"{self.matrix=}".split("=")[0] + "=\n" + self.matrix.__str__()
        else:
            np.set_printoptions(
                formatter={
                    "float": lambda x: (
                        f"{x:10.0f}" if abs(x) < 1e-4 else f"{x:10,.0f}"
                    )
                }
            )
            matrix_str = (
                f"{self.matrix=}".split("=")[0]
                + f"= 1e{N}*\n"
                + (self.matrix / 10**N).__str__()
            )

        np.set_printoptions()

        return matrix_str

    def __repr__(self):
        return f"MultiAngle(np.{repr(self.matrix)}, synthesis_mat={self.kwds})"
