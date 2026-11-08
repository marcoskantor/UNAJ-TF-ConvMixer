"""Modelos comparados en la réplica.

Cada modelo se vincula con una unidad del programa de "Tópicos en Aprendizaje Automático" (UNAJ):

* U2  Redes lineales ........ AsociadorLineal (seudoinversa / SVD) y PerceptronMulticlase.
* U3  Perceptrón multicapa .. MLP entrenado con retropropagación (torch).
* U4  CNN ................... ConvMixer (modelo del paper) y ResNetChica (CNN piramidal clásica).
* U5  ELM ................... ELM (salida por SVD con regularización) y ELMMulticapa (ML-ELM con ELM-autoencoders).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# =============================================================================================
# U4 — ConvMixer (Trockman & Kolter, 2023), transcripción de la Fig. 7 del paper
# =============================================================================================
class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def ConvMixer(dim: int, depth: int, kernel_size: int = 9, patch_size: int = 7, n_classes: int = 10,
              activacion: type[nn.Module] = nn.GELU) -> nn.Sequential:
    """ConvMixer-h/d: patch embedding + d bloques (depthwise k×k con residual, luego pointwise 1×1)."""
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        activacion(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
            Residual(nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                activacion(),
                nn.BatchNorm2d(dim),
            )),
            nn.Conv2d(dim, dim, kernel_size=1),
            activacion(),
            nn.BatchNorm2d(dim),
        ) for _ in range(depth)],
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes),
    )


def params_convmixer(h: int, d: int, k: int, p: int, c_in: int = 3, n_clases: int = 10) -> int:
    """Ec. (4) del paper: cantidad exacta de parámetros de un ConvMixer."""
    return h * (d * (k * k + h + 6) + c_in * p * p + n_clases + 3) + n_clases


# =============================================================================================
# U4 — CNN piramidal clásica (línea de base tipo ResNet, con submuestreo progresivo)
# =============================================================================================
class BloqueBasico(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int = 1):
        super().__init__()
        self.c1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(c_out)
        self.c2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(c_out)
        self.atajo = nn.Identity() if (stride == 1 and c_in == c_out) else nn.Sequential(
            nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        y = torch.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return torch.relu(y + self.atajo(x))


class ResNetChica(nn.Module):
    """ResNet de 3 etapas (32→16→8 de resolución), estilo He et al. (2016) para CIFAR."""

    def __init__(self, anchos=(48, 96, 192), bloques_por_etapa: int = 1, n_clases: int = 10):
        super().__init__()
        self.tallo = nn.Sequential(nn.Conv2d(3, anchos[0], 3, 1, 1, bias=False), nn.BatchNorm2d(anchos[0]), nn.ReLU())
        capas, c = [], anchos[0]
        for i, a in enumerate(anchos):
            for j in range(bloques_por_etapa):
                capas.append(BloqueBasico(c, a, stride=2 if (i > 0 and j == 0) else 1))
                c = a
        self.etapas = nn.Sequential(*capas)
        self.cabeza = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c, n_clases))

    def forward(self, x):
        return self.cabeza(self.etapas(self.tallo(x)))


# =============================================================================================
# U3 — Perceptrón multicapa
# =============================================================================================
def MLP(entrada: int = 3072, ocultas=(1024, 512), n_clases: int = 10, p_drop: float = 0.2) -> nn.Sequential:
    capas, a = [nn.Flatten()], entrada
    for h in ocultas:
        capas += [nn.Linear(a, h), nn.ReLU(), nn.Dropout(p_drop)]
        a = h
    capas.append(nn.Linear(a, n_clases))
    return nn.Sequential(*capas)


def contar_parametros(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# =============================================================================================
# U2 — Redes lineales (numpy)
# =============================================================================================
def one_hot(y: np.ndarray, k: int = 10, neg: float = 0.0) -> np.ndarray:
    t = np.full((len(y), k), neg, dtype=np.float64)
    t[np.arange(len(y)), y] = 1.0
    return t


def _agregar_sesgo(x: np.ndarray) -> np.ndarray:
    return np.hstack([x, np.ones((x.shape[0], 1), dtype=x.dtype)])


def pinv_ridge(H: np.ndarray, T: np.ndarray, lam=0.0):
    """Solución de mínimos cuadrados W = H⁺ T vía SVD, con regularización de Tikhonov opcional.

    H = U Σ Vᵀ  ⇒  W = V diag(σ / (σ² + λ)) Uᵀ T.  Con λ = 0 es la seudoinversa de Moore–Penrose.
    Si `lam` es una lista, reutiliza la misma SVD y devuelve una lista de soluciones (una por λ).
    """
    U, s, Vt = np.linalg.svd(H, full_matrices=False)
    tol = s.max() * max(H.shape) * np.finfo(H.dtype).eps
    UtT = U.T @ T
    del U

    def sol(l):
        filtro = np.where(s > tol, s / (s ** 2 + l), 0.0)
        return (Vt.T @ (filtro[:, None] * UtT)).astype(H.dtype)

    return [sol(l) for l in lam] if isinstance(lam, (list, tuple, np.ndarray)) else sol(lam)


class AsociadorLineal:
    """Asociador lineal / regresión lineal multisalida resuelta en forma cerrada con la seudoinversa (U2)."""

    def __init__(self, lam: float = 0.0):
        self.lam = lam

    def fit(self, X, y):
        self.W = pinv_ridge(_agregar_sesgo(X.astype(np.float32)), one_hot(y).astype(np.float32), self.lam)
        return self

    def decision(self, X):
        return _agregar_sesgo(X.astype(np.float32)) @ self.W

    def predict(self, X):
        return self.decision(X).argmax(1)


class PerceptronMulticlase:
    """Perceptrón simple de Rosenblatt, versión multiclase (una neurona escalón por clase, regla de corrección por error)."""

    def __init__(self, epocas: int = 10, eta: float = 1.0, semilla: int = 0):
        self.epocas, self.eta, self.semilla = epocas, eta, semilla

    def fit(self, X, y, X_val=None, y_val=None):
        rng = np.random.default_rng(self.semilla)
        Xb = _agregar_sesgo(X.astype(np.float32))
        self.W = np.zeros((Xb.shape[1], 10), dtype=np.float32)
        self.W_prom = np.zeros_like(self.W)  # perceptrón promediado (Freund & Schapire, 1999)
        self.historia = []
        for _ in range(self.epocas):
            errores = 0
            for i in rng.permutation(len(Xb)):
                pred = int(np.argmax(Xb[i] @ self.W))
                if pred != y[i]:
                    self.W[:, y[i]] += self.eta * Xb[i]
                    self.W[:, pred] -= self.eta * Xb[i]
                    errores += 1
                self.W_prom += self.W
            reg = {"errores_entrenamiento": errores / len(Xb)}
            if X_val is not None:
                reg["acc_val"] = float((self.predict(X_val) == y_val).mean())
            self.historia.append(reg)
        return self

    def predict(self, X, promediado: bool = True):
        W = self.W_prom if promediado else self.W
        return (_agregar_sesgo(X.astype(np.float32)) @ W).argmax(1)


# =============================================================================================
# U5 — Extreme Learning Machine (Huang, Zhu & Siew, 2006)
# =============================================================================================
def _activ(nombre: str):
    return {"sigmoide": lambda z: 1.0 / (1.0 + np.exp(-z)),
            "tanh": np.tanh,
            "relu": lambda z: np.maximum(z, 0.0)}[nombre]


class ELM:
    """ELM de una capa oculta: pesos de entrada aleatorios y fijos; pesos de salida β = H⁺T por SVD.

    Con λ > 0 la solución es la regularizada β = (HᵀH + λI)⁻¹HᵀT (Huang et al., 2012), calculada por SVD.
    """

    def __init__(self, n_ocultas: int = 2000, activacion: str = "relu", lam: float = 1.0, escala: float = 1.0,
                 semilla: int = 0):
        self.L, self.act, self.lam, self.escala, self.semilla = n_ocultas, activacion, lam, escala, semilla

    def ocultas(self, X):
        return _activ(self.act)(X @ self.A + self.b)

    def fit(self, X, y=None, T=None):
        rng = np.random.default_rng(self.semilla)
        d = X.shape[1]
        self.A = (rng.standard_normal((d, self.L)) * self.escala / np.sqrt(d)).astype(np.float32)
        self.b = (rng.standard_normal(self.L) * self.escala).astype(np.float32)
        H = self.ocultas(X.astype(np.float32))
        T = (one_hot(y) if T is None else T).astype(np.float32)
        self.beta = pinv_ridge(H, T, self.lam)
        if isinstance(self.beta, list):        # varios λ: se elige después con elegir_lambda()
            self.betas_lam, self.beta = self.beta, self.beta[0]
        return self

    def elegir_lambda(self, X_val, y_val):
        """Selecciona λ por exactitud en validación (sólo si fit recibió una lista de λ)."""
        Hv = self.ocultas(X_val.astype(np.float32))
        accs = [float(((Hv @ b).argmax(1) == y_val).mean()) for b in self.betas_lam]
        i = int(np.argmax(accs))
        self.beta, self.lam_elegido = self.betas_lam[i], self.lam[i]
        del self.betas_lam
        return dict(zip(map(float, self.lam), accs))

    def decision(self, X):
        return self.ocultas(X.astype(np.float32)) @ self.beta

    def predict(self, X):
        return self.decision(X).argmax(1)


class ELMMulticapa:
    """ML-ELM (Kasun, Zhou, Huang & Vong, 2013): capas apiladas entrenadas como ELM-autoencoders.

    Cada capa i aprende β_i reconstruyendo su propia entrada con una ELM (Xᵢ ≈ g(XᵢA+b)β_i) y usa
    g(Xᵢ β_iᵀ) como representación para la capa siguiente. La última capa es una ELM supervisada.
    """

    def __init__(self, capas=(1000, 1000), n_salida_ocultas: int = 3000, lam_ae: float = 1e-1, lam: float = 1.0,
                 semilla: int = 0):
        self.capas, self.L, self.lam_ae, self.lam, self.semilla = capas, n_salida_ocultas, lam_ae, lam, semilla

    def _transformar(self, X):
        for beta in self.betas:
            X = np.tanh(X @ beta.T)
        return X

    def fit(self, X, y):
        rng = np.random.default_rng(self.semilla)
        X = X.astype(np.float32)
        self.betas = []
        for n in self.capas:
            d = X.shape[1]
            A = rng.standard_normal((d, n)).astype(np.float32) / np.sqrt(d)
            A, _ = np.linalg.qr(A) if n <= d else (A, None)  # pesos ortogonales, como en el paper de ML-ELM
            b = (rng.standard_normal(n) * 0.1).astype(np.float32)
            H = np.tanh(X @ A + b)
            beta = pinv_ridge(H, X, self.lam_ae)       # (n, d): reconstruye X desde H
            self.betas.append(beta)
            X = np.tanh(X @ beta.T)
        self.salida = ELM(self.L, "relu", self.lam, semilla=self.semilla + 1).fit(X, y)
        return self

    def predict(self, X):
        return self.salida.predict(self._transformar(X.astype(np.float32)))
