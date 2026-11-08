"""Experimentos de la réplica parcial de "Patches Are All You Need?" sobre CIFAR-10.

Uso:
    python experimentos.py clasicos            # U2/U3/U5: seudoinversa, perceptrón, MLP, ELM, ML-ELM (≈ 15 min)
    python experimentos.py cm128_k9            # un modelo convolucional (ver EXPERIMENTOS_CNN)
    python experimentos.py todos               # todo, en el orden de config.json
    python experimentos.py rapido              # prueba de humo: 1 época sobre 5 000 imágenes (≈ 3 min)

Cada experimento guarda results/<nombre>.json (historia, exactitud de test, matriz de confusión,
parámetros, throughput) y, para las CNN, results/<nombre>.pt (pesos finales).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

from datos import cargar_cifar10, normalizar, particion_validacion  # noqa: E402
from entrenamiento import Config, entrenar, evaluar, throughput    # noqa: E402
from modelos import (ConvMixer, ResNetChica, MLP, AsociadorLineal, PerceptronMulticlase, ELM,  # noqa: E402
                     ELMMulticapa, contar_parametros, params_convmixer)

RAIZ = AQUI.parent
RES = RAIZ / "results"
CFG = json.loads((RAIZ / "config.json").read_text(encoding="utf8"))


def cargar(n_val: int | None = None, semilla: int | None = None, n_train: int | None = None):
    n_val = CFG["n_validacion"] if n_val is None else n_val
    semilla = CFG["semilla_particion"] if semilla is None else semilla
    x_tr, y_tr, x_te, y_te = cargar_cifar10()
    i_tr, i_val = particion_validacion(len(x_tr), n_val, semilla)
    if n_train:
        i_tr = i_tr[:n_train]
    X = normalizar(x_tr)
    return X[i_tr], y_tr[i_tr], X[i_val], y_tr[i_val], normalizar(x_te), y_te


def matriz_confusion(y, p, k=10):
    m = np.zeros((k, k), dtype=int)
    np.add.at(m, (y, p), 1)
    return m.tolist()


def guardar(nombre: str, d: dict):
    RES.mkdir(exist_ok=True)
    (RES / f"{nombre}.json").write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf8")
    print(f"→ results/{nombre}.json", flush=True)


# =============================================================================================
# CNN (U4): ConvMixer y ResNet
# =============================================================================================
def construir(spec: dict):
    if spec["tipo"] == "convmixer":
        act = torch.nn.ReLU if spec.get("act") == "relu" else torch.nn.GELU
        return ConvMixer(spec["h"], spec["d"], spec["k"], spec["p"], 10, act)
    if spec["tipo"] == "resnet":
        return ResNetChica(tuple(spec["anchos"]), spec.get("bloques", 1))
    raise ValueError(spec)


def correr_cnn(nombre: str, rapido: bool = False):
    spec = CFG["cnn"][nombre]
    ent = dict(CFG["entrenamiento_cnn"], **spec.get("entrenamiento", {}))
    if rapido:
        ent["epocas"] = 1
    x_tr, y_tr, x_val, y_val, x_te, y_te = cargar(n_train=5000 if rapido else None)
    torch.manual_seed(ent.get("semilla", 0))
    modelo = construir(spec)
    cfg = Config(**ent)
    t0 = time.time()
    res = entrenar(modelo, x_tr, y_tr, x_val, y_val, cfg, nombre)
    acc_te, perd_te, pred = evaluar(modelo, x_te, y_te)
    res.update({
        "nombre": nombre, "spec": spec, "unidad": "U4",
        "parametros": contar_parametros(modelo),
        "parametros_ec4": params_convmixer(spec["h"], spec["d"], spec["k"], spec["p"]) if spec["tipo"] == "convmixer" else None,
        "acc_test": acc_te, "perdida_test": perd_te, "confusion": matriz_confusion(y_te, pred),
        "throughput_img_s": throughput(modelo), "minutos": (time.time() - t0) / 60,
        "torch": torch.__version__, "hilos": torch.get_num_threads(),
    })
    print(f"[{nombre}] TEST acc = {acc_te:.4f}  params = {res['parametros']}", flush=True)
    sufijo = "_rapido" if rapido else ""
    guardar(nombre + sufijo, res)
    if not rapido:
        torch.save(modelo.state_dict(), RES / f"{nombre}.pt")
    return res


# =============================================================================================
# Modelos "clásicos" del programa (U1–U3, U5) sobre píxeles crudos
# =============================================================================================
def correr_clasicos(rapido: bool = False):
    c = CFG["clasicos"]
    x_tr, y_tr, x_val, y_val, x_te, y_te = cargar(n_train=5000 if rapido else None)
    F = lambda x: x.reshape(len(x), -1)
    Xtr, Xval, Xte = F(x_tr), F(x_val), F(x_te)
    out = {}

    # --- U1 + U2: asociador lineal (seudoinversa) con λ elegido por validación cruzada de 5 particiones
    t0 = time.time()
    lams = c["lineal_lambdas"]
    k = c["k_folds"]
    folds = np.array_split(np.random.default_rng(0).permutation(len(Xtr)), k)
    cv = np.zeros((k, len(lams)))
    for f in range(k):
        val = folds[f]
        tr = np.concatenate([folds[j] for j in range(k) if j != f])
        H = np.hstack([Xtr[tr], np.ones((len(tr), 1), np.float32)])
        from modelos import pinv_ridge, one_hot
        Ws = pinv_ridge(H, one_hot(y_tr[tr]).astype(np.float32), lams)
        Hv = np.hstack([Xtr[val], np.ones((len(val), 1), np.float32)])
        cv[f] = [((Hv @ W).argmax(1) == y_tr[val]).mean() for W in Ws]
        print(f"  CV lineal fold {f + 1}/{k}: " + " ".join(f"{a:.3f}" for a in cv[f]), flush=True)
    lam_cv = lams[int(cv.mean(0).argmax())]
    lin = AsociadorLineal(lam_cv).fit(Xtr, y_tr)
    p = lin.predict(Xte)
    out["lineal_seudoinversa"] = {"unidad": "U2", "lambda_cv": lam_cv, "lambdas": lams, "cv_media": cv.mean(0).tolist(),
                                  "cv_desvio": cv.std(0).tolist(), "acc_test": float((p == y_te).mean()),
                                  "acc_val": float((lin.predict(Xval) == y_val).mean()),
                                  "confusion": matriz_confusion(y_te, p), "parametros": int(lin.W.size),
                                  "segundos": time.time() - t0}
    print(f"[lineal] λ_cv={lam_cv}  test={out['lineal_seudoinversa']['acc_test']:.4f}", flush=True)

    # --- U2: perceptrón simple multiclase (promediado), 3 semillas
    t0 = time.time()
    accs, hist = [], []
    for s in range(c["semillas"]):
        per = PerceptronMulticlase(epocas=c["perceptron_epocas"], semilla=s).fit(Xtr, y_tr, Xval, y_val)
        accs.append(float((per.predict(Xte) == y_te).mean()))
        hist.append(per.historia)
    out["perceptron"] = {"unidad": "U2", "acc_test_semillas": accs, "acc_test": float(np.mean(accs)),
                         "historia": hist, "parametros": 3073 * 10, "segundos": time.time() - t0}
    print(f"[perceptrón] test={np.mean(accs):.4f} ± {np.std(accs):.4f}", flush=True)

    # --- U3: MLP con retropropagación (Adam/AdamW, mismo bucle que las CNN pero sin aumentación espacial)
    t0 = time.time()
    accs, hist = [], []
    for s in range(c["semillas"]):
        torch.manual_seed(s)
        m = MLP(3072, tuple(c["mlp_ocultas"]))
        ent = dict(c["mlp_entrenamiento"], semilla=s, epocas=1 if rapido else c["mlp_entrenamiento"]["epocas"])
        r = entrenar(m, x_tr, y_tr, x_val, y_val, Config(**ent), f"MLP s{s}", verbose=False)
        a, _, pm = evaluar(m, x_te, y_te)
        accs.append(a)
        hist.append(r["historia"])
    out["mlp"] = {"unidad": "U3", "acc_test_semillas": accs, "acc_test": float(np.mean(accs)), "historia": hist,
                  "parametros": contar_parametros(m), "confusion": matriz_confusion(y_te, pm),
                  "throughput_img_s": throughput(m), "segundos": time.time() - t0}
    print(f"[MLP] test={np.mean(accs):.4f} ± {np.std(accs):.4f}", flush=True)

    # --- U5: ELM, barrido del número de neuronas ocultas; λ elegido en validación (misma SVD)
    t0 = time.time()
    barrido = []
    for L in c["elm_ocultas"]:
        r = []
        for s in range(c["semillas"]):
            elm = ELM(L, "relu", c["elm_lambdas"], semilla=s).fit(Xtr, y_tr)
            curva = elm.elegir_lambda(Xval, y_val)
            r.append({"semilla": s, "lambda": elm.lam_elegido, "acc_val_por_lambda": curva,
                      "acc_test": float((elm.predict(Xte) == y_te).mean()),
                      "acc_entr": float((elm.predict(Xtr[:10000]) == y_tr[:10000]).mean())})
        barrido.append({"L": L, "corridas": r, "acc_test": float(np.mean([q["acc_test"] for q in r])),
                        "acc_test_desvio": float(np.std([q["acc_test"] for q in r])),
                        "acc_entr": float(np.mean([q["acc_entr"] for q in r]))})
        print(f"[ELM] L={L}: test={barrido[-1]['acc_test']:.4f}  entr={barrido[-1]['acc_entr']:.4f}", flush=True)
    mejor = max(barrido, key=lambda b: b["acc_test"])
    out["elm"] = {"unidad": "U5", "barrido": barrido, "acc_test": mejor["acc_test"], "L_mejor": mejor["L"],
                  "parametros": mejor["L"] * (3072 + 1 + 10), "parametros_entrenables": mejor["L"] * 10,
                  "segundos": time.time() - t0}

    # --- U5: ELM multicapa (ML-ELM)
    t0 = time.time()
    accs = []
    for s in range(c["semillas"]):
        ml = ELMMulticapa(tuple(c["mlelm_capas"]), c["mlelm_salida"], c["mlelm_lambda_ae"], c["elm_lambdas"], semilla=s)
        ml.fit(Xtr, y_tr)
        ml.salida.elegir_lambda(ml._transformar(Xval), y_val)
        accs.append(float((ml.predict(Xte) == y_te).mean()))
        print(f"  ML-ELM semilla {s}: {accs[-1]:.4f}", flush=True)
    out["ml_elm"] = {"unidad": "U5", "acc_test_semillas": accs, "acc_test": float(np.mean(accs)),
                     "capas": c["mlelm_capas"], "segundos": time.time() - t0}
    print(f"[ML-ELM] test={np.mean(accs):.4f}", flush=True)

    guardar("clasicos_rapido" if rapido else "clasicos", out)
    return out


if __name__ == "__main__":
    torch.set_num_threads(CFG.get("hilos", 8))
    que = sys.argv[1] if len(sys.argv) > 1 else "todos"
    if que == "clasicos":
        correr_clasicos()
    elif que == "rapido":
        correr_clasicos(rapido=True)
        correr_cnn(CFG["orden_cnn"][0], rapido=True)
    elif que == "todos":
        correr_clasicos()
        for n in CFG["orden_cnn"]:
            correr_cnn(n)
    else:
        correr_cnn(que)
