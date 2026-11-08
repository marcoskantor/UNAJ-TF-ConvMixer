"""Bucle de entrenamiento con descenso por gradiente estocástico (mini-lotes) para los modelos de torch.

Receta tomada del paper (Apéndice B), reducida a lo que corre en CPU:
* AdamW + tasa de aprendizaje triangular (OneCycle lineal), recorte de la norma del gradiente.
* Aumentación: recorte aleatorio con relleno + espejo horizontal + escalado aleatorio ("random scaling",
  la aumentación que el paper encontró más importante). Opcionales: random erasing, mixup/cutmix.
* NO se usa RandAugment (requiere torchvision); es una diferencia declarada con el paper.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    epocas: int = 10
    lote: int = 128
    lr_max: float = 0.01
    weight_decay: float = 0.01
    clip: float | None = 1.0
    escalado: bool = True          # random scaling (zoom 0.8–1.2 seguido de recorte 32×32)
    recorte_espejo: bool = True
    borrado: float = 0.0           # probabilidad de random erasing
    mixup: float = 0.0             # alfa de mixup (0 = no)
    suavizado: float = 0.0         # label smoothing
    semilla: int = 0
    hilos: int | None = None
    dispositivo: str = "auto"      # "auto" usa CUDA si está disponible (p. ej., Google Colab)
    extra: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- aumentación (en tensores)
def aumentar(x: torch.Tensor, cfg: Config, g: torch.Generator) -> torch.Tensor:
    n = x.shape[0]
    if cfg.escalado:
        # zoom aleatorio por lote en [0.8, 1.2] mediante una transformación afín (equivale a RandomResizedCrop suave)
        s = 0.8 + 0.4 * torch.rand(n, generator=g)
        theta = torch.zeros(n, 2, 3)
        theta[:, 0, 0] = 1 / s
        theta[:, 1, 1] = 1 / s
        theta[:, :, 2] = (torch.rand(n, 2, generator=g) - 0.5) * 0.25
        grilla = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grilla, padding_mode="reflection", align_corners=False)
    if cfg.recorte_espejo:
        x = F.pad(x, (4, 4, 4, 4), mode="reflect")
        i, j = torch.randint(0, 9, (2,), generator=g).tolist()
        x = x[:, :, i:i + 32, j:j + 32]
        espejo = torch.rand(n, generator=g) < 0.5
        x = torch.where(espejo[:, None, None, None], x.flip(3), x)
    if cfg.borrado > 0:
        for k in torch.nonzero(torch.rand(n, generator=g) < cfg.borrado).flatten().tolist():
            h, w = torch.randint(6, 16, (2,), generator=g).tolist()
            i, j = torch.randint(0, 32 - h, (1,), generator=g).item(), torch.randint(0, 32 - w, (1,), generator=g).item()
            x[k, :, i:i + h, j:j + w] = torch.randn(3, h, w, generator=g)
    return x.contiguous()


@torch.no_grad()
def elegir_dispositivo(nombre: str = "auto") -> torch.device:
    if nombre == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(nombre)


def _a(x: torch.Tensor, disp: torch.device) -> torch.Tensor:
    return x.to(disp, non_blocking=True).contiguous(memory_format=torch.channels_last) if x.dim() == 4 else x.to(disp)


@torch.no_grad()
def evaluar(modelo: nn.Module, x: np.ndarray, y: np.ndarray, lote: int = 500):
    modelo.eval()
    disp = next(modelo.parameters()).device
    pred, perdida = [], 0.0
    for i in range(0, len(x), lote):
        xb = _a(torch.from_numpy(x[i:i + lote]), disp)
        yb = torch.from_numpy(y[i:i + lote]).to(disp)
        out = modelo(xb)
        perdida += F.cross_entropy(out, yb, reduction="sum").item()
        pred.append(out.argmax(1).cpu().numpy())
    pred = np.concatenate(pred)
    return float((pred == y).mean()), perdida / len(x), pred


def entrenar(modelo: nn.Module, x_tr, y_tr, x_val, y_val, cfg: Config, nombre: str = "", verbose: bool = True):
    """Entrena y devuelve un dict con la historia por época (pérdida, exactitud, tiempo)."""
    if cfg.hilos:
        torch.set_num_threads(cfg.hilos)
    torch.manual_seed(cfg.semilla)
    disp = elegir_dispositivo(cfg.dispositivo)
    # channels_last acelera ~2,5× la convolución depthwise en CPU (oneDNN) y no cambia el resultado
    modelo.to(disp).to(memory_format=torch.channels_last)
    g = torch.Generator().manual_seed(cfg.semilla)
    rng = np.random.default_rng(cfg.semilla)
    pasos_ep = math.ceil(len(x_tr) / cfg.lote)
    opt = torch.optim.AdamW(modelo.parameters(), lr=cfg.lr_max, weight_decay=cfg.weight_decay)
    total = cfg.epocas * pasos_ep
    # Programa triangular (el del repo oficial de ConvMixer): 0 → lr_max en 2/5, → lr_max/20 en 4/5, → 0 al final
    lr_de = lambda t: float(np.interp(t, [0, total * 2 / 5, total * 4 / 5, total], [0, cfg.lr_max, cfg.lr_max / 20, 0]))
    historia, paso, t0 = [], 0, time.time()
    for ep in range(cfg.epocas):
        modelo.train()
        perm = rng.permutation(len(x_tr))
        suma_perdida, aciertos = 0.0, 0
        for b in range(pasos_ep):
            idx = perm[b * cfg.lote:(b + 1) * cfg.lote]
            xb = _a(aumentar(torch.from_numpy(x_tr[idx]), cfg, g), disp)
            yb = torch.from_numpy(y_tr[idx]).to(disp)
            for grupo in opt.param_groups:
                grupo["lr"] = lr_de(paso)
            if cfg.mixup > 0:
                lam = float(rng.beta(cfg.mixup, cfg.mixup))
                orden = torch.randperm(len(idx), generator=g).to(disp)
                xb = lam * xb + (1 - lam) * xb[orden]
                out = modelo(xb)
                perdida = lam * F.cross_entropy(out, yb, label_smoothing=cfg.suavizado) + \
                    (1 - lam) * F.cross_entropy(out, yb[orden], label_smoothing=cfg.suavizado)
            else:
                out = modelo(xb)
                perdida = F.cross_entropy(out, yb, label_smoothing=cfg.suavizado)
            opt.zero_grad(set_to_none=True)
            perdida.backward()
            if cfg.clip:
                nn.utils.clip_grad_norm_(modelo.parameters(), cfg.clip)
            opt.step()
            paso += 1
            suma_perdida += perdida.item() * len(idx)
            aciertos += (out.argmax(1) == yb).sum().item()
        acc_val, perd_val, _ = evaluar(modelo, x_val, y_val)
        reg = {"epoca": ep + 1, "perdida_entr": suma_perdida / len(x_tr), "acc_entr": aciertos / len(x_tr),
               "acc_val": acc_val, "perdida_val": perd_val, "segundos": time.time() - t0}
        historia.append(reg)
        if verbose:
            print(f"[{nombre}] ep {ep + 1:3d}/{cfg.epocas}  pérdida {reg['perdida_entr']:.3f}  "
                  f"acc_entr {reg['acc_entr']:.3f}  acc_val {acc_val:.4f}  ({reg['segundos'] / 60:.1f} min)", flush=True)
    return {"config": asdict(cfg), "historia": historia, "dispositivo": str(disp)}


@torch.no_grad()
def throughput(modelo: nn.Module, n_lotes: int = 5, lote: int = 64) -> float:
    """Imágenes por segundo en inferencia (CPU), análogo a la columna 'Throughput' de las Tablas 1–2."""
    modelo.eval()
    disp = next(modelo.parameters()).device
    x = torch.randn(lote, 3, 32, 32, device=disp).contiguous(memory_format=torch.channels_last)
    modelo(x)
    t0 = time.perf_counter()
    for _ in range(n_lotes):
        modelo(x)
    return n_lotes * lote / (time.perf_counter() - t0)
