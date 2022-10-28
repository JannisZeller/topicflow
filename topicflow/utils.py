# ------------------------------------------------------------------------------
# topicflow.utils
# ---------------
# 
# Helper functions and plot-tools for `topicflow.models` and `topicflow.data`.
# ------------------------------------------------------------------------------


# %% Dependencies
# ------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt




# %% One-hot encoding for numpy
# ------------------------------------------------------------------------------

def np_one_hot(indices: np.ndarray, depth: int) -> np.ndarray:
    """One-hot encoding inspired by `tensorflow.one_hot`. Always adds one-hot 
    dimension as last axis.

    Parameters
    ----------
    indices : np.ndarray
        Array with discrete, finitely many distinct values.
    depth : int
        Number of one hot encodings.

    Returns
    -------
    np.ndarray : 
        One-hot encoded along chosen dimension.
    """
    res = np.eye(depth)[np.array(indices).reshape(-1)]
    res = res.reshape(list(indices.shape)+[depth])
    return res


# %% Visualizing multiple topics simultaniously in a grid as square images
# ------------------------------------------------------------------------------

def visualize_topics(Theta: np.ndarray, N_row: int, N_col: int) -> plt.Axes:
    """Visualizing multiple topics simultaniously in a grid as square images.

    Parameters
    ----------
    Theta : np.ndarray
        Array representing topics-word distriburions.
    N_row : int
        Number of one hot encodings.

    Returns
    -------
    np.ndarray : 
        One-hot encoded along chosen dimension.
    """
    if type(Theta) != np.ndarray:
        Theta = np.array(Theta)

    K = Theta.shape[0]
    V = Theta.shape[1]

    assert np.sqrt(V).is_integer(), "To be plotted and represented as square imgage V must be a square of an integer."

    sqrtV = int(np.sqrt(V))

    fig, axes = plt.subplots(N_row, N_col,figsize = (3*N_col, 3*N_row))
    
    idx = 0
    for i in range(N_row):
        for k in range(N_col):
            axes[i, k].imshow(Theta[idx,:].reshape((sqrtV, sqrtV)), cmap="Greys")
            axes[i, k].set_xticks([])
            axes[i, k].set_yticks([])

            idx += 1
    
    return fig


# %% Converting docs to images
# ------------------------------------------------------------------------------

def doc_to_image(document: np.ndarray, sqrt_V: int) -> np.ndarray:
    """Converting documents to images.

    Parameters
    ----------
    document : np.ndarray
        Array representing a full tokenized document.
    sqrt_V : int
        Square root of vocab size (must be int).

    Returns
    -------
    np.ndarray : 
        Returns counts per vocab element, transformed to a square image.
    """
    vals = dict(zip(*np.unique(document, return_counts=True)))
    img  = []

    for i in range(int(sqrt_V**2)):
        if i in vals:
            img.append(vals[i])
        else:
            img.append(0)

    img = np.array(img).reshape(sqrt_V, sqrt_V)

    return img


# %% Converting docs to images
# ------------------------------------------------------------------------------
def visualize_random_docs(
    documents: np.ndarray, 
    sqrt_V: int, 
    N_row: int = 2, 
    N_col: int = 5, 
    ) -> plt.Axes:
    """Visualized some random documents from a document-array or tensor (number
    of documents as 0-dimension).

    Parameters
    ----------
    documents : np.ndarray
        Array representing a full tokenized document.
    N_row : int
        Number of grid-rows.
    N_col : int
        Number of grid-columns.
    sqrt_V : int
        Square root of vocab size (must be int).

    Returns
    -------
    np.ndarray : 
        Plot of a grid of documents represented by counts.
    """

    if type(documents) != np.ndarray:
        documents = np.array(documents)

    D = documents.shape[0]

    idxs = np.random.choice(np.arange(D), size = N_row * N_col)
    print(f"Presenting documents {idxs}")

    images = []
    for idx in idxs:
        images.append(doc_to_image(documents[idx, :], sqrt_V=sqrt_V))

    fig, axes = plt.subplots(N_row, N_col,figsize = (3*N_col, 3*N_row))
    idx = 0
    for i in range(N_row):
        for k in range(N_col):
            axes[i, k].imshow(images[idx].reshape((sqrt_V, sqrt_V)), cmap="Greys")
            axes[i, k].set_xticks([])
            axes[i, k].set_yticks([])

            idx += 1

    return fig