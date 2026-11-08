"""Carga de CIFAR-10 (versión "python" oficial de la Universidad de Toronto) sin torchvision.

El archivo cifar-10-python.tar.gz (170 MB) se descarga una sola vez a data/.
Devuelve arreglos numpy uint8 de forma (N, 32, 32, 3) y etiquetas int64.
"""
from __future__ import annotations

import pickle
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CLASES = ["avión", "auto", "pájaro", "gato", "ciervo", "perro", "rana", "caballo", "barco", "camión"]
# Media y desvío por canal del conjunto de entrenamiento (los mismos que usa timm para CIFAR-10)
MEDIA = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
DESVIO = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)


def _carpeta_datos() -> Path:
    return Path(__file__).resolve().parent.parent / "data"


def descargar(carpeta: Path | None = None) -> Path:
    carpeta = carpeta or _carpeta_datos()
    carpeta.mkdir(parents=True, exist_ok=True)
    destino = carpeta / "cifar-10-batches-py"
    if destino.exists():
        return destino
    tgz = carpeta / "cifar-10-python.tar.gz"
    if not tgz.exists():
        print(f"Descargando CIFAR-10 de {URL} ...")
        urllib.request.urlretrieve(URL, tgz)
    with tarfile.open(tgz) as t:
        t.extractall(carpeta, filter="data")
    return destino


def _leer_lote(ruta: Path):
    with open(ruta, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    x = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y = np.array(d[b"labels"], dtype=np.int64)
    return x, y


def cargar_cifar10(carpeta: Path | None = None):
    """(x_train, y_train, x_test, y_test) con x en uint8 (N,32,32,3)."""
    base = descargar(carpeta)
    xs, ys = zip(*[_leer_lote(base / f"data_batch_{i}") for i in range(1, 6)])
    x_tr, y_tr = np.concatenate(xs), np.concatenate(ys)
    x_te, y_te = _leer_lote(base / "test_batch")
    return x_tr, y_tr, x_te, y_te


def normalizar(x_uint8: np.ndarray) -> np.ndarray:
    """uint8 (N,32,32,3) -> float32 (N,3,32,32) estandarizado por canal."""
    x = x_uint8.astype(np.float32) / 255.0
    x = (x - MEDIA) / DESVIO
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


def particion_validacion(n: int, n_val: int, semilla: int = 0):
    """Índices estratificables simples: una permutación fija separa n_val ejemplos para validación."""
    rng = np.random.default_rng(semilla)
    perm = rng.permutation(n)
    return perm[n_val:], perm[:n_val]
