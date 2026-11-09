"""Réplica en GPU (Google Colab) con la receta del repositorio oficial de los autores para CIFAR-10.

Basado en train.py de https://github.com/locuslab/convmixer-cifar10 (Trockman y Kolter; licencia MIT).
Diferencias con ese script: guarda un JSON por época, mide el tiempo total y usa el modelo de nuestro
src/modelos.py (idéntico). La exactitud que se informa es la de la ÚLTIMA época (no se elige la mejor época
mirando el test).

Uso en Colab (Entorno de ejecución → Cambiar tipo → GPU T4):
    # Corrida 2 del TF (receta del repositorio, 100 épocas, 94,47 %):
    !python colab_receta_paper.py --epochs 100 --conv-ks 5
    # Receta del artículo (Apéndice B: 200 épocas, mixup + CutMix, sin weight decay; Tabla 4: 95,21 %),
    # con un checkpoint por época en Drive para reanudar si se corta la sesión (~3 h 10 min en T4):
    from google.colab import drive; drive.mount('/content/drive')
    !python colab_receta_paper.py --epochs 200 --conv-ks 5 --wd 0 --mixup 0.8 --cutmix 1.0 --ckpt /content/drive/MyDrive/convmixer_200ep_ckpt.pt --salida resultado_colab_200ep.json
    (si la sesión se corta, volver a ejecutar EXACTAMENTE el mismo comando: retoma desde la última época)

mixup/CutMix: por lote, en la GPU, como en timm (con probabilidad 0,5 se elige uno u otro; alfas 0,8 y 1,0,
los valores por defecto de timm/DeiT; el artículo no publica los suyos). Con --mixup 0 --cutmix 0 (por defecto)
el script hace exactamente lo mismo que antes.
"""
import os
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T

p = argparse.ArgumentParser()
p.add_argument("--name", default="ConvMixer-256/8-k5-p2")
p.add_argument("--batch-size", default=512, type=int)
p.add_argument("--scale", default=1.0, type=float)
p.add_argument("--reprob", default=0.2, type=float)
p.add_argument("--ra-m", default=12, type=int)
p.add_argument("--ra-n", default=2, type=int)
p.add_argument("--jitter", default=0.2, type=float)
p.add_argument("--hdim", default=256, type=int)
p.add_argument("--depth", default=8, type=int)
p.add_argument("--psize", default=2, type=int)
p.add_argument("--conv-ks", default=5, type=int)
p.add_argument("--wd", default=0.005, type=float)
p.add_argument("--epochs", default=100, type=int)
p.add_argument("--lr-max", default=0.05, type=float)
p.add_argument("--workers", default=2, type=int)
p.add_argument("--salida", default="resultado_colab.json")
p.add_argument("--mixup", default=0.0, type=float, help="alfa de mixup (0 = no usar)")
p.add_argument("--cutmix", default=0.0, type=float, help="alfa de CutMix (0 = no usar)")
p.add_argument("--ckpt", default="", help="checkpoint por época (p. ej. en Drive) para reanudar")
a = p.parse_args()


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def ConvMixer(dim, depth, kernel_size=5, patch_size=2, n_classes=10):
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size), nn.GELU(), nn.BatchNorm2d(dim),
        *[nn.Sequential(Residual(nn.Sequential(nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                                               nn.GELU(), nn.BatchNorm2d(dim))),
                        nn.Conv2d(dim, dim, kernel_size=1), nn.GELU(), nn.BatchNorm2d(dim)) for _ in range(depth)],
        nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(dim, n_classes))


media, desvio = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
aum = T.Compose([T.RandomResizedCrop(32, scale=(a.scale, 1.0), ratio=(1.0, 1.0)), T.RandomHorizontalFlip(),
                 T.RandAugment(num_ops=a.ra_n, magnitude=a.ra_m), T.ColorJitter(a.jitter, a.jitter, a.jitter),
                 T.ToTensor(), T.Normalize(media, desvio), T.RandomErasing(p=a.reprob)])
plano = T.Compose([T.ToTensor(), T.Normalize(media, desvio)])
tr = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=aum)
te = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=plano)
dl_tr = torch.utils.data.DataLoader(tr, a.batch_size, shuffle=True, num_workers=a.workers, pin_memory=True)
dl_te = torch.utils.data.DataLoader(te, a.batch_size, shuffle=False, num_workers=a.workers, pin_memory=True)

dev = "cuda"
modelo = ConvMixer(a.hdim, a.depth, a.conv_ks, a.psize).to(dev).to(memory_format=torch.channels_last)
n_par = sum(q.numel() for q in modelo.parameters())
lr_de = lambda t: np.interp([t], [0, a.epochs * 2 // 5, a.epochs * 4 // 5, a.epochs], [0, a.lr_max, a.lr_max / 20, 0])[0]
opt = optim.AdamW(modelo.parameters(), lr=a.lr_max, weight_decay=a.wd)
crit = nn.CrossEntropyLoss()
escala = torch.amp.GradScaler("cuda")
print(f"{a.name}: {n_par:,} parámetros | GPU: {torch.cuda.get_device_name(0)}", flush=True)


def mezclar(X, y):
    """mixup o CutMix sobre el lote (como timm). Devuelve X mezclado, y permutada y lambda."""
    usos = [m for m, alfa in (("mixup", a.mixup), ("cutmix", a.cutmix)) if alfa > 0]
    if not usos:
        return X, y, 1.0
    modo = usos[np.random.randint(len(usos))]
    lam = float(np.random.beta(a.mixup, a.mixup) if modo == "mixup" else np.random.beta(a.cutmix, a.cutmix))
    perm = torch.randperm(X.size(0), device=X.device)
    if modo == "mixup":
        X = lam * X + (1 - lam) * X[perm]
    else:
        H, W = X.shape[-2:]
        rh, rw = int(H * np.sqrt(1 - lam)), int(W * np.sqrt(1 - lam))
        cy, cx = np.random.randint(H), np.random.randint(W)
        y0, y1 = np.clip(cy - rh // 2, 0, H), np.clip(cy + rh // 2, 0, H)
        x0, x1 = np.clip(cx - rw // 2, 0, W), np.clip(cx + rw // 2, 0, W)
        X = X.clone()
        X[:, :, y0:y1, x0:x1] = X[perm, :, y0:y1, x0:x1]
        lam = 1 - (y1 - y0) * (x1 - x0) / (H * W)
    return X, y[perm], lam


hist, ep0, previos = [], 0, 0.0
if a.ckpt and os.path.exists(a.ckpt):
    c = torch.load(a.ckpt, map_location=dev, weights_only=False)
    modelo.load_state_dict(c["modelo"])
    opt.load_state_dict(c["opt"])
    escala.load_state_dict(c["escala"])
    hist, ep0, previos = c["hist"], c["epoca"], c["segundos"]
    print(f"Reanudando desde el checkpoint {a.ckpt}: {ep0} épocas hechas", flush=True)
t0 = time.time() - previos
for ep in range(ep0, a.epochs):
    ti = time.time()
    modelo.train()
    perd, acc, n = 0.0, 0, 0
    for i, (X, y) in enumerate(dl_tr):
        X = X.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(dev, non_blocking=True)
        lr = lr_de(ep + (i + 1) / len(dl_tr))
        opt.param_groups[0]["lr"] = lr
        opt.zero_grad(set_to_none=True)
        Xm, y2, lam = mezclar(X, y)
        with torch.autocast("cuda", dtype=torch.float16):
            out = modelo(Xm)
            loss = lam * crit(out, y) + (1 - lam) * crit(out, y2) if lam < 1 else crit(out, y)
        escala.scale(loss).backward()
        escala.unscale_(opt)
        nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        escala.step(opt)
        escala.update()
        perd += loss.item() * y.size(0)
        acc += (out.argmax(1) == y).sum().item()
        n += y.size(0)
    modelo.eval()
    ok, m = 0, 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for X, y in dl_te:
            out = modelo(X.to(dev).to(memory_format=torch.channels_last))
            ok += (out.argmax(1).cpu() == y).sum().item()
            m += y.size(0)
    hist.append({"epoca": ep + 1, "perdida_entr": perd / n, "acc_entr": acc / n, "acc_test": ok / m,
                 "segundos_epoca": time.time() - ti})
    print(f"[{a.name}] época {ep + 1:3d}/{a.epochs} | entr {acc / n:.4f} | test {ok / m:.4f} | "
          f"{time.time() - ti:.0f} s | lr {lr:.5f}", flush=True)
    if a.ckpt:
        torch.save({"modelo": modelo.state_dict(), "opt": opt.state_dict(), "escala": escala.state_dict(),
                    "hist": hist, "epoca": ep + 1, "segundos": time.time() - t0}, a.ckpt + ".tmp")
        os.replace(a.ckpt + ".tmp", a.ckpt)

res = {"nombre": a.name, "args": vars(a), "parametros": n_par, "gpu": torch.cuda.get_device_name(0),
       "acc_test_final": hist[-1]["acc_test"], "minutos": (time.time() - t0) / 60, "historia": hist,
       "fuente_receta": "https://github.com/locuslab/convmixer-cifar10"}
json.dump(res, open(a.salida, "w"), indent=1)
torch.save(modelo.state_dict(), a.salida.replace(".json", ".pt"))
print("RESULTADO_JSON " + json.dumps({k: v for k, v in res.items() if k != "historia"}), flush=True)
print("HISTORIA_TEST " + ",".join(f"{h['acc_test']:.4f}" for h in hist), flush=True)
