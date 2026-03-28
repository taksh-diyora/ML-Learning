import numpy as np

class LinearRegression():
    def __init__(self, n_iterations = 10000, learning_rate = 0.01):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iterations):
            f_wb = np.dot(X, self.w) + self.b
            error = f_wb - y

            dw = np.dot(X.T, error) / n_samples
            db = np.sum(error) / n_samples

            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.w) + self.b
    
    def score(self, X, y):
        f_wb = self.predict(X)
        ss_res = np.sum((y-f_wb)**2)
        ss_tot = np.sum((y - np.mean(y))**2)

        return 1 - ss_res/ss_tot

class LogisticRegression():
    def __init__(self, alpha=0.01, n_iterations=10000, lambda_=0):
        self.alpha = alpha
        self.n_iterations = n_iterations
        self.lambda_ = lambda_
        self.b = None
        self.w = None
    
    def _sigmoid(self, z):
        return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))
    
    def fit(self, X, y):
        m, n_features = X.shape
        self.b = 0
        self.w = np.zeros(n_features)

        for _ in range(self.n_iterations):
            z = np.dot(X, self.w) + self.b
            f_wb = self._sigmoid(z)
            error = f_wb - y

            dw = np.dot(X.T, error)/m + (self.lambda_/m) * self.w
            db = np.sum(error)/m

            self.w = self.w - self.alpha*dw
            self.b = self.b - self.alpha*db

    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        f_wb = self._sigmoid(z)
        return (f_wb >= 0.5).astype(int)
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)