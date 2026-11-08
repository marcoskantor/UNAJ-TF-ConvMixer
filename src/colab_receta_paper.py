"""Réplica en GPU (Google Colab) con la receta del repositorio oficial de los autores para CIFAR-10.

Basado en train.py de https://github.com/locuslab/convmixer-cifar10 (Trockman y Kolter; licencia MIT).
Diferencias con ese script: guarda un JSON por época, mide el tiempo total y usa el modelo de nuestro
src/modelos.py (idéntico). La exactitud que se informa es la de la ÚLTIMA época (no se elige la mejor época
mirando el test).

Uso en Colab (Entorno de ejecución → Cambiar tipo → GPU T4):
    !python colab_receta_paper.py --epochs 100 --conv-ks 5
"""
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

hist, t0 = [], time.time()
for ep in range(a.epochs):
    ti = time.time()
    modelo.train()
    perd, acc, n = 0.0, 0, 0
    for i, (X, y) in enumerate(dl_tr):
        X = X.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(dev, non_blocking=True)
        lr = lr_de(ep + (i + 1) / len(dl_tr))
        opt.param_groups[0]["lr"] = lr
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            out = modelo(X)
            loss = crit(out, y)
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

res = {"nombre": a.name, "args": vars(a), "parametros": n_par, "gpu": torch.cuda.get_device_name(0),
       "acc_test_final": hist[-1]["acc_test"], "minutos": (time.time() - t0) / 60, "historia": hist,
       "fuente_receta": "https://github.com/locuslab/convmixer-cifar10"}
json.dump(res, open(a.salida, "w"), indent=1)
torch.save(modelo.state_dict(), a.salida.replace(".json", ".pt"))
print("RESULTADO_JSON " + json.dumps({k: v for k, v in res.items() if k != "historia"}), flush=True)
print("HISTORIA_TEST " + ",".join(f"{h['acc_test']:.4f}" for h in hist), flush=True)
