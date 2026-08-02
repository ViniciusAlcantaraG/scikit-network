

class KullbackLeiblerDivergence(BaseLoss):
    """Kullback-Leibler divergence loss."""

    def __init__(self):
        super(KullbackLeiblerDivergence, self).__init__()
        self.name = 'Kullback-Leibler divergence'

    @staticmethod
    def loss(signal: np.ndarray, labels: np.ndarray) -> float:
        """Get loss value.

        Parameters
        ----------
        signal : np.ndarray, shape (n_samples, n_channels)
            Predicted probabilities.
        labels : np.ndarray, shape (n_samples, n_channels)
            Target probabilities.

        Returns
        -------
        value : float
            Loss value.
        """
        eps = 1e-10
        signal = np.clip(signal, eps, 1 - eps)
        labels = np.clip(labels, eps, 1 - eps)

        value = labels * (np.log(labels) - np.log(signal))

        return value.sum() / len(labels)

    @staticmethod
    def loss_gradient(signal: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Get the gradient of the loss function.

        Parameters
        ----------
        signal : np.ndarray, shape (n_samples, n_channels)
            Predicted probabilities.
        labels : np.ndarray, shape (n_samples, n_channels)
            Target probabilities.

        Returns
        -------
        gradient : np.ndarray, shape (n_samples, n_channels)
            Gradient.
        """
        eps = 1e-10
        signal = np.clip(signal, eps, 1 - eps)

        gradient = -labels / signal

        return gradient / len(labels)