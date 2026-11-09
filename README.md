# Trabajo Final Integrador — Tópicos en Aprendizaje Automático (UNAJ)

**¿Bastan los parches? Análisis crítico y réplica parcial de *Patches Are All You Need?* (ConvMixer)**
Marcos David Kantor · Doctorado en Ciencia y Tecnología, Universidad Nacional Arturo Jauretche, 2026
Docentes: Dr. Ing. Martín Morales y Dr. Ing. Marcelo Cappelletti

> Trockman, A. y Kolter, J. Z. (2023). Patches are all you need? *Transactions on Machine Learning Research*.
> https://doi.org/10.48550/arXiv.2201.09792 — PDF en [`paper/`](paper/) y en <https://arxiv.org/abs/2201.09792> (código oficial: <https://github.com/locuslab/convmixer>).

Consigna del trabajo: [`consigna/Consigna_TF_Aprendizaje_Automatico_UNAJ.pdf`](consigna/Consigna_TF_Aprendizaje_Automatico_UNAJ.pdf)

## Entregables (los que pide la consigna)

| Entregable | Archivo |
|---|---|
| Artículo científico seleccionado | [`paper/Trockman_Kolter_2023_Patches_Are_All_You_Need.pdf`](paper/Trockman_Kolter_2023_Patches_Are_All_You_Need.pdf) (versión enviada a ICLR 2022; publicado en TMLR 2023) |
| Presentación para la exposición oral (Beamer, estética UNAJ) | [`presentacion/TF_presentacion.pdf`](presentacion/TF_presentacion.pdf) (fuente `.tex`) |
| Código de la réplica (notebook) | [`notebooks/replica_convmixer.ipynb`](notebooks/replica_convmixer.ipynb) (ejecutado; copia [`.html`](notebooks/replica_convmixer.html)) |
| Informe escrito del análisis crítico (APA 7) | [`informe/TF_informe.pdf`](informe/TF_informe.pdf) |

## Datos

**CIFAR-10** (Krizhevsky, 2009): 60 000 imágenes en color de 32×32, 10 clases.
Página oficial: <https://www.cs.toronto.edu/~kriz/cifar-10.html> — archivo usado:
<https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz> (versión *python*, 170 MB).
`src/datos.py` lo descarga solo en `data/` la primera vez (la carpeta no se sube al repositorio).

## Resultados

| Unidad | Modelo (CIFAR-10, test de 10 000 imágenes) | Parámetros | Test |
|---|---|---:|---:|
| U4 | ConvMixer-128/8, k=3, p=2 | 149 642 | 84,6 % |
| U4 | ResNet (misma receta) | 308 650 | 83,0 % |
| U4 | **ConvMixer-128/8, k=9, p=2** (modelo del paper) | 223 370 | **82,7 %** |
| U4 | ConvMixer-128/8, k=9, p=4 | 227 978 | 78,7 % |
| U3 | MLP 3072-1024-512-10 | 3 676 682 | 57,4 % |
| U5 | ELM (L = 3000, SVD + Tikhonov) | 9 249 000 | 47,3 % |
| U2 | Seudoinversa (λ por validación cruzada de 5 pliegues) | 30 730 | 40,7 % |
| U5 | ML-ELM | — | 40,4 % |
| U2 | Perceptrón simple multiclase | 30 730 | 40,3 % |
| GPU | ConvMixer-256/8, k=5, p=2, receta del repositorio, 100 épocas, Colab T4 (~93 min, ~56 s/época) | 594 186 | **94,5 %** |
| GPU | ConvMixer-256/8, k=5, p=2, receta del artículo (200 épocas, mixup + CutMix, wd 0), Kaggle T4 (~85 min, ~25 s/época) | 594 186 | **95,48 %** |

CNN: 10 épocas en CPU (8 núcleos), una semilla. Modelos no convolucionales: promedio de 3 semillas.
Publicado: 95,0 % (256/8, k=9, p=2, 200 épocas) y 95,21 % (k=5, p=2).

**Sobre la réplica en GPU (todo tal cual pasó).** Una sola semilla por configuración, y se informa la última época
(no la mejor). La primera corrida de Colab se cortó en la época 84 (94,50 %) al perderse la sesión gratuita (registro en
`results/colab_gpu_corrida1_log.txt`); la segunda terminó las 100 épocas (94,47 %). Después Colab gratuito nos negó la GPU
por límite de uso y la receta del artículo (200 épocas, mixup y CutMix con alfas 0,8 y 1,0 —los valores por defecto de
timm, el artículo no publica los suyos—, sin *weight decay*) se corrió en Kaggle: **95,48 %**, frente al 95,21 % publicado.
Esos 0,27 puntos están dentro del azar esperable con una semilla, así que la réplica *iguala* el valor publicado pero no
demuestra superarlo; además, como cambiaron a la vez épocas, mixup/CutMix y *weight decay*, no se puede decir cuál pesó más.
En Kaggle cada época fue más del doble de rápida (~25 s frente a ~56 s) con la misma GPU, sin una causa verificada.
El resultado está en `results/kaggle_200ep.json` (transcripto del registro del notebook: exactitud de test por época y
tiempos; no se guardó la exactitud de entrenamiento por época, que con mixup/CutMix no es comparable).

## Cómo reproducir

```bash
pip install -r requirements.txt
cd src
python experimentos.py rapido      # prueba de humo (~10 min en CPU)
python experimentos.py todos       # todo el TF (~5 h en CPU de 8 núcleos)
python figuras.py                  # figuras + presentacion/resultados.tex
python construir_notebook.py       # regenera el notebook
cd ../presentacion && latexmk -pdf TF_presentacion.tex
cd ../informe && latexmk -pdf TF_informe.tex
```

Réplica en GPU (Google Colab, Entorno de ejecución → T4): subir `src/colab_receta_paper.py` y ejecutar
`!python colab_receta_paper.py --epochs 100 --conv-ks 5` (~1 h 30 min).

Receta del artículo (Apéndice B: 200 épocas, mixup + CutMix, sin weight decay; publicado 95,21 %), con
checkpoint en Drive para reanudar si se corta la sesión (en Colab, ~3 h 10 min estimadas a ~56 s/época; en Kaggle tardó ~85 min):

```python
from google.colab import drive; drive.mount('/content/drive')
!python colab_receta_paper.py --epochs 200 --conv-ks 5 --wd 0 --mixup 0.8 --cutmix 1.0 --ckpt /content/drive/MyDrive/convmixer_200ep_ckpt.pt --salida resultado_colab_200ep.json
```

Si la sesión se corta, se vuelve a ejecutar el mismo comando y retoma desde la última época guardada.

## Estructura

```
consigna/       consigna del Trabajo Final (cátedra)
paper/          artículo analizado
src/            datos.py, modelos.py (ConvMixer, ResNet, MLP, seudoinversa, perceptrón, ELM, ML-ELM),
                entrenamiento.py, experimentos.py, figuras.py, construir_notebook.py, colab_receta_paper.py
config.json     todas las semillas e hiperparámetros
results/        resultados (JSON), pesos (.pt) y registros
notebooks/      notebook didáctico ejecutado (+ HTML)
presentacion/   Beamer UNAJ y figuras
informe/        informe escrito (APA 7)
```

## Uso de asistentes de IA

Usé un asistente de IA (Claude Code) como apoyo para la programación y la revisión de la redacción. La
verificación de los resultados fue propia: el conteo de parámetros se comparó con la ec. (4) del artículo, el test se evaluó una sola vez y todo
número citado se genera desde `results/*.json`.
