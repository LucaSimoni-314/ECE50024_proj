import numpy as np
from scipy.optimize import minimize

class Optimizer:
    """Base class to handle the projection and shared data."""
    def __init__(self, x_init, y, A, alpha_param, c_param, bounds=(0, 1)):
        self.x = np.array(x_init, dtype=float)
        self.y = y
        self.A = A
        self.alpha_param = alpha_param
        self.c_param = c_param
        self.bounds = bounds

    def project(self, x):
        """Pi_C(x): Guarantees 0 <= x <= 1."""
        return np.clip(x, self.bounds[0], self.bounds[1])

    def compute_gradient(self, x_val):
        """
        Implementation from provided image: 
        grad = -2 * A.T @ ((y - h(Ax)) * h'(Ax))
        """
        # 1. Linear transformation
        z = self.A @ x_val
        # 2. Sigmoid activation
        h_z = 1 / (1 + np.exp(-self.alpha_param * (z - self.c_param)))
        # 3. Residual
        residual = self.y - h_z
        # 4. Derivative of h_z w.r.t z
        h_prime_z = self.alpha_param * h_z * (1 - h_z)
        # 5. Chain rule gradient
        grad = -2 * self.A.T @ (residual * h_prime_z)
        return grad

    def objective_fn(self, x_val):
        """Required for L-BFGS and tracking progress."""
        z = self.A @ x_val
        h_z = 1 / (1 + np.exp(-self.alpha_param * (z - self.c_param)))
        return np.sum((self.y - h_z)**2)

# --- 1. Standard Gradient Descent ---
class StandardGD(Optimizer):
    def __init__(self, x_init, y, A, alpha_param, c_param, learning_rate=0.01):
        super().__init__(x_init, y, A, alpha_param, c_param)
        self.lr = learning_rate

    def step(self):
        grad = self.compute_gradient(self.x)
        self.x = self.x - self.lr * grad
        return self.x

# --- 2. Projected Gradient Descent ---
class ProjectedGD(Optimizer):
    def __init__(self, x_init, y, A, alpha_param, c_param, learning_rate=0.01):
        super().__init__(x_init, y, A, alpha_param, c_param)
        self.lr = learning_rate

    def step(self):
        grad = self.compute_gradient(self.x)
        # Apply projection after the step
        self.x = self.project(self.x - self.lr * grad)
        return self.x

# --- 3. Stochastic Gradient Descent ---
class StochasticGD(Optimizer):
    def __init__(self, x_init, y, A, alpha_param, c_param, learning_rate=0.01, batch_size=32):
        super().__init__(x_init, y, A, alpha_param, c_param)
        self.lr = learning_rate
        self.batch_size = batch_size

    def step(self):
        # Sample a subset of the rows in A and y
        indices = np.random.choice(len(self.y), self.batch_size, replace=False)
        A_batch = self.A[indices]
        y_batch = self.y[indices]
        
        # Localized gradient calculation for the batch
        z = A_batch @ self.x
        h_z = 1 / (1 + np.exp(-self.alpha_param * (z - self.c_param)))
        h_prime_z = self.alpha_param * h_z * (1 - h_z)
        grad = -2 * A_batch.T @ ((y_batch - h_z) * h_prime_z)
        
        self.x = self.project(self.x - self.lr * grad)
        return self.x

# --- 4. Nesterov Momentum Acceleration ---
class NesterovMomentum(Optimizer):
    def __init__(self, x_init, y, A, alpha_param, c_param, learning_rate=0.01, momentum=0.9):
        super().__init__(x_init, y, A, alpha_param, c_param)
        self.lr = learning_rate
        self.mu = momentum
        self.v = np.zeros_like(self.x)

    def step(self):
        # Lookahead
        x_lookahead = self.x + self.mu * self.v
        grad = self.compute_gradient(x_lookahead)
        
        # Update velocity and project position
        self.v = self.mu * self.v - self.lr * grad
        self.x = self.project(self.x + self.v)
        return self.x

# --- 5. L-BFGS ---
class LBFGSOptimizer(Optimizer):
    def __init__(self, x_init, y, A, alpha_param, c_param):
        super().__init__(x_init, y, A, alpha_param, c_param)

    def run(self):
        # Scipy's L-BFGS-B handles projection via the 'bounds' parameter
        res = minimize(
            fun=self.objective_fn,
            x0=self.x,
            jac=self.compute_gradient,
            method='L-BFGS-B',
            bounds=[self.bounds] * len(self.x)
        )
        self.x = res.x
        return self.x
    
class HypergradientDescent(Optimizer):
    def __init__(self, x_init, y, A, alpha_param, c_param, initial_lr=0.01, hyper_lr=1e-4):
        super().__init__(x_init, y, A, alpha_param, c_param)
        self.lr = initial_lr
        self.beta = hyper_lr  # The "learning rate for the learning rate"
        self.prev_grad = np.zeros_like(self.x)

    def step(self):
        # 1. Compute current gradient
        grad = self.compute_gradient(self.x)
        
        # 2. Update the learning rate (Hypergradient step)
        # Formula: lr_t = lr_{t-1} + beta * <grad_t, grad_{t-1}>
        # If the directions match, speed up. If they oppose, slow down.
        h_grad = np.dot(grad, self.prev_grad)
        self.lr = self.lr + self.beta * h_grad
        
        # Guard against negative learning rates
        self.lr = max(self.lr, 1e-6)

        # 3. Standard update step
        self.x = self.project(self.x - self.lr * grad)
        
        # 4. Store gradient for the next iteration
        self.prev_grad = grad
        
        return self.x