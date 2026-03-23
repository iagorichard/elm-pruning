from typing import Dict, List
import math
import torch


class ELMRegressor:

    def __init__(self, hidden_dim: int, reg_lambda: float, activation_name: str, seed: int, eps: float, use_double_for_solver: bool):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = hidden_dim
        self.reg_lambda = reg_lambda
        self.activation_name = activation_name
        self.seed = seed
        self.eps = eps
        self.use_double_for_solver = use_double_for_solver


    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> Dict[str, torch.Tensor]:
        X = X.to(self.device)
        Y = Y.to(self.device)

        self.X_mean = X.mean(dim=0, keepdim=True)
        self.X_std = X.std(dim=0, keepdim=True, unbiased=False).clamp_min(self.eps)
        Xn = (X - self.X_mean) / self.X_std

        self.Y_mean = Y.mean(dim=0, keepdim=True)
        Yn = Y - self.Y_mean

        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed)

        in_dim = Xn.shape[1]
        out_dim = Yn.shape[1]

        self.W = torch.randn((in_dim, self.hidden_dim), generator=generator, device=self.device) / math.sqrt(max(in_dim, 1))
        self.b = torch.randn((self.hidden_dim,), generator=generator, device=self.device)

        H = self.__apply_activation(Xn @ self.W + self.b, self.activation_name)

        I = torch.eye(self.hidden_dim, device=self.device, dtype=H.dtype)
        lhs = H.T @ H + self.reg_lambda * I
        rhs = H.T @ Yn

        if self.use_double_for_solver:
            self.beta = torch.linalg.solve(lhs.double(), rhs.double()).to(H.dtype)
        else:
            self.beta = torch.linalg.solve(lhs, rhs)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.W.device)

        Xn = (X - self.X_mean) / self.X_std
        H = self.__apply_activation(
            Xn @ self.W + self.b,
            self.activation_name,
        )
        return H @ self.beta + self.Y_mean

    def compute_ablation_importance(self, X: torch.Tensor, Y: torch.Tensor) -> List[float]:
        """
        Importance = increase in ELM loss when feature is neutralized to its mean value.
        Since pruning removes less important filters, low scores should be pruned first.
        """
        X = X.to(self.W.device)
        Y = Y.to(self.W.device)

        base_pred = self.predict(X)
        base_loss = self.calculate_loss(base_pred, Y).item()

        importances: List[float] = []
        X_work = X.clone()

        for feature_idx in range(X.shape[1]):
            original_column = X_work[:, feature_idx].clone()

            # Neutralize feature by sending it to its mean value
            X_work[:, feature_idx] = self.X_mean[0, feature_idx]

            ablated_pred = self.predict(X_work)
            ablated_loss = self.calculate_loss(ablated_pred, Y).item()

            importance = max(ablated_loss - base_loss, 0.0)
            importances.append(float(importance))

            X_work[:, feature_idx] = original_column

        return importances
    
    def calculate_loss(self, Y_pred, Y_original):
        return self.__mse(Y_pred, Y_original)
    
    def __apply_activation(self, x: torch.Tensor, activation_name: str) -> torch.Tensor:
        activation_name = activation_name.lower()

        if activation_name == "tanh":
            return torch.tanh(x)
        if activation_name == "relu":
            return torch.relu(x)
        if activation_name == "sigmoid":
            return torch.sigmoid(x)

        raise ValueError(f"Unsupported activation: {activation_name}")
    
    def __mse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean((pred - target) ** 2)