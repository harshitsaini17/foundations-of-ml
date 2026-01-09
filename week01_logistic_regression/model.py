import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.1, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def _sigmoid(self, z):
        """
        TASK 1:
        Implement sigmoid.
        Question: what happens when z is very large or very small?
        """
        z = np.clip(z, -500, 500)
        result = 1/(1 + np.exp(-z))
        return result

    def _compute_loss(self, y, y_hat):
        """
        TASK 2:
        Implement Binary Cross-Entropy loss.
        Add a small epsilon to avoid log(0).
        """
        epsilon = 1e-15
        loss =  -( y*(np.log(y_hat+epsilon)) + (1-y)*(np.log(1-y_hat+epsilon))).mean()

        return loss
    
    def fit(self, X, y):
        """
        TASK 3:
        - initialize w, b
        - write training loop
        - compute y_hat
        - compute gradients dw, db
        - update parameters
        """
        m,n = X.shape

        w = np.zeros((n,))
        b = 0

        for i in range(self.n_iters):

            z = X@w + b

            y_hat = self._sigmoid(z)

            dw = (X.T @ (y_hat - y))/m
            db = (y_hat - y).mean()
            if i % 100 == 0:
                print("Loss: ",self._compute_loss(y, y_hat), "\n")

            w = w - dw * self.lr
            b = b - db * self.lr

        self.w = w
        self.b = b


    def predict_proba(self, X):
        """
        TASK 4:
        Return probabilities P(y=1|x)
        """
        z = X@self.w + self.b
        
        y_hat = self._sigmoid(z)

        return y_hat

    def predict(self, X, threshold=0.5):
        """
        TASK 5:
        Convert probabilities to class labels
        """
        prob = self.predict_proba(X)

        return (prob>threshold).astype(int)
    

