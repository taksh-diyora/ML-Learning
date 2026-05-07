import numpy as np

class LinearRegression():
    def __init__(self, n_iters=10000, alpha = 0.01):
        self.n_iters = n_iters
        self.alpha = alpha
        self.w = None
        self.b = None
        self.tolerance = 0.005

    def _initialize_parameter(self, X):
        self.w = np.zeros(X.shape[1])
        self.b = 0
    
    def _predict(self, X):
        return np.dot(X, self.w) + self.b
    
    def _cost_function(self, error, m):
        return np.sum((error)**2)/(m*2)
    
    def _compute_gradient(self, X, error):
        m = X.shape[0]
        dj_dw = np.dot(X.T, error)/m
        dj_db = np.sum(error)/m

        return dj_dw, dj_db
    
    def fit(self, X, y, verbose = False):
        self.cost_history = []
        self._initialize_parameter(X)
        epsilon = 1e-8
        for i in range(self.n_iters):
            y_pred = self._predict(X)
            error = y_pred - y

            dj_dw, dj_db = self._compute_gradient(X, error)

            self.w = self.w - self.alpha*dj_dw
            self.b = self.b - self.alpha*dj_db

            y_pred_new = self._predict(X)
            error_new = y_pred_new - y

            cost = self._cost_function(error_new, X.shape[0])
            if i>0 and abs(cost - self.cost_history[-1])/max(epsilon, self.cost_history[-1]) < self.tolerance:
                print(f"Stopped at iteration {i}")
                break
            self.cost_history.append(cost)

            if verbose == True and i % 100 == 0:
                print(cost)

    def predict(self, X):
        return self._predict(X)
    
    def score(self, X, y):
        y_pred = self._predict(X)
        sse = np.sum((y - y_pred)**2)
        sst = np.sum((y - np.mean(y))**2)

        return 1 - sse/(sst+1e-8)


class LogisticRegression():
    def __init__(self, n_iters=10000, alpha = 0.01):
        self.n_iters = n_iters
        self.alpha = alpha
        self.w = None
        self.b = None
        self.tolerance = 0.005

    def _sigmoid(self, z):
        return 1/(1+np.exp(-np.clip(z, -500, 500)))

    def _initialize_parameter(self, X):
        self.w = np.zeros(X.shape[1])
        self.b = 0
    
    def _predict(self, X):
        return self._sigmoid(np.dot(X, self.w) + self.b)
    
    def _cost_function(self, y, y_pred):
        m = y.shape[0]
        epsilon = 1e-8
        return -np.sum(y*np.log(y_pred+epsilon) + (1-y)*np.log(1-y_pred+epsilon)) / m
    
    def _compute_gradient(self, X, error):
        m = X.shape[0]
        dj_dw = np.dot(X.T, error)/m
        dj_db = np.sum(error)/m

        return dj_dw, dj_db
    
    def fit(self, X, y, verbose = False):
        self.cost_history = []
        self._initialize_parameter(X)
        epsilon = 1e-8
        for i in range(self.n_iters):
            y_pred = self._predict(X)
            error = y_pred - y

            dj_dw, dj_db = self._compute_gradient(X, error)

            self.w = self.w - self.alpha*dj_dw
            self.b = self.b - self.alpha*dj_db

            y_pred_new = self._predict(X)

            cost = self._cost_function(y, y_pred_new)
            if i>0 and abs(cost - self.cost_history[-1])/max(epsilon, self.cost_history[-1]) < self.tolerance:
                print(f"Stopped at iteration {i}")
                break
            self.cost_history.append(cost)

            if verbose == True and i % 100 == 0:
                print(cost)

    def predict(self, X):
        probs = self._predict(X)
        return (probs >= 0.5).astype(int)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)