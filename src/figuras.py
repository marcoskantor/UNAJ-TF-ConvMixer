"""Genera las figuras de la presentación/informe y los números citados (resultados.tex / resultados.json).

Uso:  python figuras.py        (lee results/*.json; ~1 min)

Convenciones visuales (mismas en todas las figuras): azul UNAJ = modelo del paper (foco), naranja = línea de base
ResNet, gris = contexto, rojo = valor publicado en el artículo. Las identidades llevan además estilo de línea y
etiqueta directa, para no depender sólo del color.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
RES = RAIZ / "results"
FIGS = RAIZ / "presentacion" / "figuras"
FIGS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(AQUI))

AZUL, NARANJA, CELESTE, GRIS, ROJO, TINTA = "#003366", "#D9822B", "#5B9BD5", "#8C8C8C", "#C0392B", "#333333"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#999999",
                     "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "savefig.dpi": 200, "savefig.bbox": "tight", "font.family": "DejaVu Sans"})

ESTILO = {"cm128_k9": (AZUL, "-", "o", "ConvMixer-128/8  k=9 p=2"),
          "resnet": (NARANJA, "--", "s", "ResNet (misma receta)"),
          "cm128_k3": (CELESTE, "-.", "^", "ConvMixer  k=3 p=2"),
          "cm128_p4": (GRIS, ":", "D", "ConvMixer  k=9 p=4")}
# Valores publicados (Tabla 4 del artículo, ConvMixer-256/8, 200 épocas)
PAPER = {"k9_p2": 95.00, "k3_p2": 94.08, "k9_p1": 95.88, "k3_p1": 93.61, "k9_p4": 92.61}


def guardar_fig(fig, nombre):
    """Guarda en un temporal y lo mueve; reintenta si otro programa (visor, sincronizador) tiene el archivo tomado."""
    import os, time, tempfile
    fd, tmp = tempfile.mkstemp(suffix=".png", dir=FIGS)
    os.close(fd)
    fig.savefig(tmp)
    for _ in range(20):
        try:
            os.replace(tmp, FIGS / nombre)
            return
        except PermissionError:
            time.sleep(1.5)
    print(f"AVISO: {nombre} está abierto en otro programa; quedó en {tmp}")


def leer(nombre):
    f = RES / f"{nombre}.json"
    return json.loads(f.read_text(encoding="utf8")) if f.exists() else None


def fmt(x, dec=1):
    return "--" if x is None else f"{x:.{dec}f}".replace(".", ",")


def miles(n):
    return "--" if n is None else f"{n:,}".replace(",", "\\,")



def etiquetar(ax, puntos, sep=3.2, fx=1.18):
    """Etiquetas directas sin solapamiento (eje x logarítmico). Si dos puntos están cerca en x y sus etiquetas
    quedarían a menos de `sep` unidades en y, corre la de arriba hacia arriba y la une al punto con una línea fina."""
    pts = sorted(puntos, key=lambda q: q[1])
    ys = []
    for i, (x, y, t) in enumerate(pts):
        yl = y
        for (x2, _, _), yl2 in zip(pts[:i], ys):
            if abs(np.log10(x) - np.log10(x2)) < 0.9 and yl - yl2 < sep:
                yl = yl2 + sep
        ys.append(yl)
    for (x, y, t), yl in zip(pts, ys):
        movida = abs(yl - y) > 0.05
        ax.annotate(t, (x, y), xytext=(x * fx, yl), textcoords="data", fontsize=8.5, color=TINTA, va="center",
                    arrowprops=dict(arrowstyle="-", color="#BBBBBB", lw=0.7, shrinkA=0, shrinkB=4) if movida else None)


cnn = {n: leer(n) for n in ESTILO}
cla = leer("clasicos") or {}


# ----------------------------------------------------------------------------- 1. curvas de validación
def fig_curvas():
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    finales = []
    for n, r in cnn.items():
        if not r:
            continue
        c, ls, mk, et = ESTILO[n]
        ep = [h["epoca"] for h in r["historia"]]
        acc = [100 * h["acc_val"] for h in r["historia"]]
        ax.plot(ep, acc, color=c, ls=ls, lw=2, marker=mk, ms=5, label=et)
        corto = {"cm128_k9": "k=9", "resnet": "ResNet", "cm128_k3": "k=3", "cm128_p4": "p=4"}[n]
        finales.append([acc[-1], acc[-1], ep[-1], corto])
    # valores finales a la derecha, separados para que no se pisen
    finales.sort()
    for k in range(1, len(finales)):
        finales[k][1] = max(finales[k][1], finales[k - 1][1] + 3.6)
    # centrar el bloque de etiquetas que se movió alrededor de los valores reales, para que no suba hacia la línea roja
    if finales:
        corr = min(0.0, (np.mean([f[0] for f in finales]) - np.mean([f[1] for f in finales])))
        for f in finales:
            f[1] += corr
        for k in range(len(finales) - 2, -1, -1):
            finales[k][1] = min(finales[k][1], finales[k + 1][1] - 3.6)
    for real, y, x, corto in finales:
        ax.annotate(f"{corto}: {real:.1f} %".replace(".", ","), (x, real), xytext=(x + 0.25, y), textcoords="data", va="center",
                    fontsize=9, color=TINTA, arrowprops=dict(arrowstyle="-", color="#BBBBBB", lw=0.6) if abs(y - real) > 0.1 else None)
    ax.axhline(PAPER["k9_p2"], color=ROJO, lw=1, ls=(0, (4, 3)))
    ax.text(0.7, PAPER["k9_p2"] - 1.2, "Trockman y Kolter (2023): 95,0 % (256/8, k=9, p=2, 200 épocas)", color=ROJO,
            fontsize=8.5, va="top")
    ax.set_xlabel("época")
    ax.set_ylabel("exactitud de validación (%)")
    ax.set_xlim(0.6, 12.2)
    ax.set_xticks(range(1, 11))
    ax.set_ylim(30, 100)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    guardar_fig(fig, "f_curvas_validacion.png")
    plt.close(fig)


# ----------------------------------------------------------------------------- 2. exactitud vs parámetros
def filas_modelos():
    filas = []
    for n, r in cnn.items():
        if r:
            filas.append((ESTILO[n][3], r["parametros"], 100 * r["acc_test"], "U4", ESTILO[n][0]))
    if "mlp" in cla:
        filas.append(("MLP (U3)", cla["mlp"]["parametros"], 100 * cla["mlp"]["acc_test"], "U3", GRIS))
    if "elm" in cla:
        filas.append((f"ELM L={cla['elm']['L_mejor']} (U5)", cla["elm"]["parametros"], 100 * cla["elm"]["acc_test"], "U5", GRIS))
    if "lineal_seudoinversa" in cla:
        filas.append(("Seudoinversa (U2)", 30730, 100 * cla["lineal_seudoinversa"]["acc_test"], "U2", GRIS))
    if "perceptron" in cla:
        filas.append(("Perceptrón (U2)", 30730, 100 * cla["perceptron"]["acc_test"], "U2", GRIS))
    return filas


def fig_comparacion():
    filas = filas_modelos()
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for et, p, a, u, c in filas:
        ax.scatter(p, a, s=70 if c != GRIS else 45, color=c, edgecolor="white", linewidth=1.5, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("parámetros (escala log)")
    ax.set_ylabel("exactitud de test (%)")
    ax.set_xlim(1.5e4, 2e8)
    ax.set_ylim(30, 100)
    # la seudoinversa y el perceptrón caen casi en el mismo punto: una etiqueta arriba y otra abajo, cerca del punto
    # las cuatro CNN caen casi en el mismo punto: sus etiquetas van en una columna a la derecha, con líneas guía
    cnn_f = sorted([(p, a, et) for et, p, a, u, c in filas if u == "U4"], key=lambda q: -q[1])
    for k, (p, a, et) in enumerate(cnn_f):
        ax.annotate(f"{et}: {a:.1f} %".replace(".", ","), (p, a), xytext=(2.2e6, 97 - 5.2 * k), textcoords="data",
                    fontsize=8.5, color=TINTA, va="center",
                    ha="left", arrowprops=dict(arrowstyle="-", color="#BBBBBB", lw=0.7, shrinkA=3, shrinkB=5, relpos=(0, 0.5)))
    etiquetar(ax, [(p, a, et) for et, p, a, u, c in filas if "(U2)" not in et and u != "U4"])
    u2 = sorted([(p, a, et) for et, p, a, u, c in filas if "(U2)" in et], key=lambda q: -q[1])
    for k, (p, a, et) in enumerate(u2):
        ax.annotate(f"{et}: {a:.1f} %".replace(".", ","), (p, a), xytext=(9, 9 if k == 0 else -9), textcoords="offset points",
                    fontsize=8.5, color=TINTA, va="center")
    guardar_fig(fig, "f_comparacion.png")
    plt.close(fig)


# ----------------------------------------------------------------------------- 3. barras: todos los modelos
def fig_barras():
    filas = sorted(filas_modelos(), key=lambda f: f[2])
    if "ml_elm" in cla:
        filas.insert(0, ("ML-ELM (U5)", None, 100 * cla["ml_elm"]["acc_test"], "U5", GRIS))
        filas = sorted(filas, key=lambda f: f[2])
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    y = np.arange(len(filas))
    ax.barh(y, [f[2] for f in filas], color=[f[4] for f in filas], height=0.62, edgecolor="white", linewidth=2)
    for i, f in enumerate(filas):
        ax.text(f[2] + 0.8, i, f"{f[2]:.1f} %".replace(".", ","), va="center", fontsize=9, color=TINTA)
    ax.set_yticks(y, [f[0] for f in filas], fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("exactitud de test en CIFAR-10 (%)")
    ax.grid(axis="y", visible=False)
    guardar_fig(fig, "f_barras.png")
    plt.close(fig)


# ----------------------------------------------------------------------------- 4. ablaciones vs paper
def fig_ablaciones():
    pares = [("k=9, p=2\n(base)", "cm128_k9", PAPER["k9_p2"]), ("k=3, p=2", "cm128_k3", PAPER["k3_p2"]),
             ("k=9, p=4", "cm128_p4", PAPER["k9_p4"])]
    pares = [q for q in pares if cnn.get(q[1])]
    if not pares:
        return
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    x = np.arange(len(pares))
    nuestra = [100 * cnn[n]["acc_test"] for _, n, _ in pares]
    paper = [p for *_, p in pares]
    ax.bar(x - 0.19, paper, 0.36, color="#F2B8B0", edgecolor="white", linewidth=2, label="paper (256/8, 200 ép., GPU)")
    ax.bar(x + 0.19, nuestra, 0.36, color=AZUL, edgecolor="white", linewidth=2,
           label=f"réplica (128/8, {cnn['cm128_k9']['config']['epocas'] if cnn.get('cm128_k9') else ''} ép., CPU)")
    for i in range(len(pares)):
        ax.text(x[i] - 0.19, paper[i] + 0.8, f"{paper[i]:.1f}".replace(".", ","), ha="center", fontsize=9, color=TINTA)
        ax.text(x[i] + 0.19, nuestra[i] + 0.8, f"{nuestra[i]:.1f}".replace(".", ","), ha="center", fontsize=9, color=TINTA)
    ax.set_xticks(x, [p[0] for p in pares])
    ax.set_ylabel("exactitud de test (%)")
    ax.set_ylim(50, 100)
    ax.legend(frameon=False, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax.grid(axis="x", visible=False)
    guardar_fig(fig, "f_ablaciones.png")
    plt.close(fig)


# ----------------------------------------------------------------------------- 5. ELM: barrido de L
def fig_elm():
    if "elm" not in cla:
        return
    b = cla["elm"]["barrido"]
    L = [q["L"] for q in b]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(L, [100 * q["acc_entr"] for q in b], color=GRIS, lw=2, ls="--", marker="s", ms=6, label="entrenamiento")
    ax.errorbar(L, [100 * q["acc_test"] for q in b], yerr=[100 * q["acc_test_desvio"] for q in b], color=AZUL, lw=2,
                marker="o", ms=6, capsize=3, label="test (media ± desvío, 3 semillas)")
    if "lineal_seudoinversa" in cla:
        a = 100 * cla["lineal_seudoinversa"]["acc_test"]
        ax.axhline(a, color=NARANJA, lw=1, ls=":")
        ax.text(L[0], a - 2.8, f"seudoinversa sobre píxeles: {a:.1f} %", color=TINTA, fontsize=9)
    ax.set_xlabel("neuronas ocultas aleatorias L")
    ax.set_ylabel("exactitud (%)")
    ax.legend(frameon=False, fontsize=9)
    guardar_fig(fig, "f_elm.png")
    plt.close(fig)


# ----------------------------------------------------------------------------- 6. validación cruzada del lineal
def fig_cv():
    if "lineal_seudoinversa" not in cla:
        return
    r = cla["lineal_seudoinversa"]
    lams = [max(l, 1) for l in r["lambdas"]]
    m, s = 100 * np.array(r["cv_media"]), 100 * np.array(r["cv_desvio"])
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.errorbar(lams, m, yerr=s, color=AZUL, marker="o", lw=2, capsize=3)
    i = int(np.argmax(m))
    ax.scatter([lams[i]], [m[i]], s=140, facecolor="none", edgecolor=ROJO, lw=1.5, zorder=4)
    ax.annotate(f"λ elegido = {r['lambda_cv']:g}", (lams[i], m[i]), xytext=(-20, 12), textcoords="offset points",
                fontsize=9, color=TINTA, ha="right")
    ax.set_xscale("log")
    ax.set_xticks(lams, ["0"] + [f"{l:g}" for l in r["lambdas"][1:]])
    ax.set_xlabel("λ (Tikhonov)")
    ax.set_ylabel("exactitud CV 5 pliegues (%)")
    guardar_fig(fig, "f_cv_lineal.png")
    plt.close(fig)


# ----------------------------------------------------------------------------- 7. matriz de confusión
def fig_confusion():
    r = cnn.get("cm128_k9")
    if not r:
        return
    from datos import CLASES
    M = np.array(r["confusion"], float)
    M = 100 * M / M.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.imshow(M, cmap="Blues", vmin=0, vmax=100)
    for i in range(10):
        for j in range(10):
            if M[i, j] >= 3:
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=7.5,
                        color="white" if M[i, j] > 55 else TINTA)
    ax.set_xticks(range(10), CLASES, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(10), CLASES, fontsize=8.5)
    ax.set_xlabel("predicha")
    ax.set_ylabel("verdadera")
    ax.grid(False)
    guardar_fig(fig, "f_confusion.png")
    plt.close(fig)


# ----------------------------------------------------------------------------- 8. filtros del embebido de parches (cf. Apéndice C)
def fig_filtros():
    f = RES / "cm128_p4.pt"
    if not f.exists():
        f = RES / "cm128_k9.pt"
    if not f.exists():
        return
    import torch
    W = torch.load(f, map_location="cpu")["0.weight"].numpy()  # (h, 3, p, p)
    W = W.transpose(0, 2, 3, 1)
    W = (W - W.min((1, 2, 3), keepdims=True)) / (np.ptp(W, axis=(1, 2, 3), keepdims=True) + 1e-8)
    n = min(64, len(W))
    fig, axs = plt.subplots(4, 16, figsize=(8, 2.1))
    for i, a in enumerate(axs.flat):
        a.axis("off")
        if i < n:
            a.imshow(W[i], interpolation="nearest")
    guardar_fig(fig, "f_filtros.png")
    plt.close(fig)
    return f.stem


# ----------------------------------------------------------------------------- 9. throughput vs exactitud
def fig_throughput():
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for n, r in cnn.items():
        if not r:
            continue
        c, _, mk, et = ESTILO[n]
        ax.scatter(r["throughput_img_s"], 100 * r["acc_test"], s=80, color=c, marker=mk, edgecolor="white", lw=1.5, zorder=3)
    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    ax.xaxis.set_major_locator(FixedLocator([250, 500, 1000, 2000, 4000]))
    ax.xaxis.set_major_formatter(FixedFormatter(["250", "500", "1000", "2000", "4000"]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(250, 6000)
    ys = [100 * r["acc_test"] for r in cnn.values() if r]
    ax.set_ylim(min(ys) - 1.5, max(ys) + 1.5)
    for n, r in cnn.items():
        if r:
            abajo = n == "cm128_k9"   # su etiqueta a la derecha chocaría con la ResNet
            ax.annotate(f"{ESTILO[n][3]}: {100 * r['acc_test']:.1f} %".replace(".", ","), (r["throughput_img_s"], 100 * r["acc_test"]),
                        xytext=(-6, -15) if abajo else (9, 0), textcoords="offset points", va="center",
                        ha="left", fontsize=8.5, color=TINTA)
    ax.set_xlabel("imágenes/s en inferencia (CPU, lote 64, log)")
    ax.set_ylabel("exactitud de test (%)")
    guardar_fig(fig, "f_throughput.png")
    plt.close(fig)



# ----------------------------------------------------------------------------- 10. corrida en GPU (Colab)
def leer_colab():
    """Resultado de la réplica en GPU: results/colab_gpu.json si la corrida terminó; si no, el log parcial."""
    import re
    f = RES / "colab_gpu.json"
    if f.exists():
        d = json.loads(f.read_text(encoding="utf8"))
        return {"acc": [h["acc_test"] for h in d["historia"]], "final": d["acc_test_final"], "completa": True,
                "epocas": d["args"]["epochs"], "seg": np.mean([h["segundos_epoca"] for h in d["historia"]])}
    f = RES / "colab_gpu_corrida1_log.txt"
    if not f.exists():
        return None
    filas = re.findall(r"época\s+(\d+)/(\d+) \| entr ([\d.]+) \| test ([\d.]+) \| (\d+) s", f.read_text(encoding="utf8"))
    acc = [float(r[3]) for r in filas]
    return {"acc": acc, "final": acc[-1], "completa": False, "epocas": int(filas[0][1]), "hechas": len(acc),
            "seg": np.mean([float(r[4]) for r in filas])}


def leer_kaggle():
    """Corrida con la receta del artículo (200 épocas, mixup/CutMix, wd 0) en Kaggle: results/kaggle_200ep.json."""
    f = RES / "kaggle_200ep.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text(encoding="utf8"))
    return {"acc": [h["acc_test"] for h in d["historia"]], "final": d["acc_test_final"], "epocas": d["args"]["epochs"],
            "seg": d["segundos_epoca_media"], "minutos": d["minutos"]}


def fig_colab():
    c, k = leer_colab(), leer_kaggle()
    if not c:
        return
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    curvas = [(c, AZUL, "-", "Colab: receta del repositorio, 100 épocas")]
    if k:
        curvas.append((k, NARANJA, "--", "Kaggle: receta del artículo, 200 épocas"))
    for d, col, ls, et in curvas:
        ep = np.arange(1, len(d["acc"]) + 1)
        ax.plot(ep, 100 * np.array(d["acc"]), color=col, lw=1.6, ls=ls, label=et)
        ax.annotate(f"{100 * d['final']:.1f} %".replace(".", ","), (ep[-1], 100 * d["final"]), xytext=(6, -10),
                    textcoords="offset points", fontsize=9, color=col)
    ax.axhline(95.21, color=ROJO, lw=1, ls=(0, (4, 3)))
    ax.text(2, 95.21 + 0.6, "Trockman y Kolter (2023), Tabla 4: 95,21 % (k=5, p=2, 200 épocas)", color=ROJO, fontsize=8.5)
    if k:
        ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    ax.set_xlabel("época")
    ax.set_ylabel("exactitud de test (%)")
    ax.set_xlim(0, max(c["epocas"], k["epocas"] if k else 0) + 12)
    ax.set_ylim(50, 100)
    guardar_fig(fig, "f_colab.png")
    plt.close(fig)


# ----------------------------------------------------------------------------- números para LaTeX
def resultados():
    R = {}
    nombres = {"cm128_k9": "CMnueve", "cm128_k3": "CMtres", "cm128_p4": "CMpcuatro", "resnet": "ResNet"}
    for n, m in nombres.items():
        r = cnn.get(n)
        R["acc" + m] = fmt(100 * r["acc_test"]) if r else "--"
        R["params" + m] = miles(r["parametros"]) if r else "--"
        R["tput" + m] = f"{r['throughput_img_s']:.0f}" if r else "--"
        R["min" + m] = f"{r['minutos']:.0f}" if r else "--"
        R["val" + m] = fmt(100 * r["historia"][-1]["acc_val"]) if r else "--"
        R["entr" + m] = fmt(100 * r["historia"][-1]["acc_entr"]) if r else "--"
    r0 = next((r for r in cnn.values() if r), None)
    R["epocasCNN"] = str(r0["config"]["epocas"]) if r0 else "10"
    g = lambda k, f: (f(cla[k]) if k in cla else "--")
    R["accLineal"] = g("lineal_seudoinversa", lambda d: fmt(100 * d["acc_test"]))
    R["lamCV"] = g("lineal_seudoinversa", lambda d: f"{d['lambda_cv']:g}")
    R["minLineal"] = g("lineal_seudoinversa", lambda d: f"{d['segundos'] / 60:.0f}")
    R["accPerceptron"] = g("perceptron", lambda d: fmt(100 * d["acc_test"]))
    R["minPerceptron"] = g("perceptron", lambda d: f"{d['segundos'] / 60 / 3:.0f}")
    R["accMLP"] = g("mlp", lambda d: fmt(100 * d["acc_test"]))
    R["paramsMLP"] = g("mlp", lambda d: miles(d["parametros"]))
    R["tputMLP"] = g("mlp", lambda d: f"{d['throughput_img_s']:.0f}")
    R["minMLP"] = g("mlp", lambda d: f"{d['segundos'] / 60 / 3:.0f}")
    R["accELM"] = g("elm", lambda d: fmt(100 * d["acc_test"]))
    R["LELM"] = g("elm", lambda d: str(d["L_mejor"]))
    R["paramsELM"] = g("elm", lambda d: miles(d["parametros"]))
    R["minELM"] = g("elm", lambda d: f"{d['segundos'] / 60 / (3 * len(d['barrido'])):.0f}")
    R["accMLELM"] = g("ml_elm", lambda d: fmt(100 * d["acc_test"]))
    R["minMLELM"] = g("ml_elm", lambda d: f"{d['segundos'] / 60 / 3:.0f}")
    # derivados
    if cnn.get("cm128_k9") and cnn.get("resnet"):
        R["ratioTput"] = fmt(cnn["resnet"]["throughput_img_s"] / cnn["cm128_k9"]["throughput_img_s"])
        R["difCMResNet"] = fmt(100 * (cnn["cm128_k9"]["acc_test"] - cnn["resnet"]["acc_test"]))
    if cnn.get("cm128_k9") and cnn.get("cm128_k3"):
        R["difKernel"] = fmt(100 * (cnn["cm128_k9"]["acc_test"] - cnn["cm128_k3"]["acc_test"]))
        R["difKernelAbs"] = fmt(abs(100 * (cnn["cm128_k9"]["acc_test"] - cnn["cm128_k3"]["acc_test"])))
    if cnn.get("cm128_k9") and cnn.get("cm128_p4"):
        R["difPatch"] = fmt(100 * (cnn["cm128_k9"]["acc_test"] - cnn["cm128_p4"]["acc_test"]))
    c = leer_colab()
    if c:
        R["accColab"] = fmt(100 * c["final"])
        R["segColab"] = f"{c['seg']:.0f}"
        R["epColab"] = str(len(c["acc"]))
        R["minColab"] = f"{c['seg'] * len(c['acc']) / 60:.0f}"
        R["minEpCPU"] = f"{cnn['cm128_k9']['minutos'] / cnn['cm128_k9']['config']['epocas']:.0f}" if cnn.get("cm128_k9") else "--"
        R["colabNota"] = (f"al cabo de las {c['epocas']} épocas" if c["completa"] else
                          f"en la época {c['hechas']} de {c['epocas']} (la sesión gratuita de Colab se cortó antes de terminar; "
                          f"en las últimas épocas la tasa de aprendizaje ya era casi nula)")
    else:
        R["accColab"], R["segColab"], R["colabNota"], R["epColab"], R["minColab"], R["minEpCPU"] = "--", "--", "", "--", "--", "--"
    k = leer_kaggle()
    if k:
        R["accKaggle"] = fmt(100 * k["final"], 2)
        R["segKaggle"] = f"{k['seg']:.0f}"
        R["epKaggle"] = str(len(k["acc"]))
        R["minKaggle"] = f"{k['minutos']:.0f}"
    else:
        R["accKaggle"], R["segKaggle"], R["epKaggle"], R["minKaggle"] = "--", "--", "--", "--"
    tex = "% Generado por src/figuras.py a partir de results/*.json — no editar a mano\n"
    tex += "".join(f"\\providecommand{{\\{k}}}{{{v}}}\n" for k, v in R.items())
    (RAIZ / "presentacion" / "resultados.tex").write_text(tex, encoding="utf8")
    (RAIZ / "presentacion" / "resultados.json").write_text(json.dumps(R, indent=1, ensure_ascii=False), encoding="utf8")
    return R


if __name__ == "__main__":
    for f in [fig_curvas, fig_comparacion, fig_barras, fig_ablaciones, fig_elm, fig_cv, fig_confusion, fig_filtros,
              fig_throughput, fig_colab]:
        f()
    R = resultados()
    print(json.dumps(R, indent=1, ensure_ascii=False))
