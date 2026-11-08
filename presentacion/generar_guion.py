"""Genera el guion de la exposición (TF_guion.docx y TF_guion.pdf) con tiempos por diapositiva y preguntas probables.

Uso: python generar_guion.py   (lee los números de resultados.json, generado por src/figuras.py)
Requiere python-docx; el PDF se compila con pdflatex.
"""
import json
import subprocess
from pathlib import Path

from docx import Document
from docx.shared import Pt, RGBColor, Cm

AQUI = Path(__file__).resolve().parent
R = json.loads((AQUI / "resultados.json").read_text(encoding="utf8"))
AZUL = RGBColor(0x00, 0x33, 0x66)

# (título, segundos, párrafos). Tiempo total objetivo: 20 min de exposición + 10 de debate.
DIAPOS = [
    ("Portada", 20, [
        "Buenas tardes. Soy Marcos Kantor. Elegí analizar el artículo «Patches Are All You Need?», de Asher "
        "Trockman y Zico Kolter, publicado en Transactions on Machine Learning Research en 2023. Además del análisis, "
        "repliqué parte de sus experimentos."]),
    ("Contenido de la exposición", 30, [
        "Sigo el orden que pide la consigna: el artículo, la introducción, el modelo, la metodología, los resultados "
        "y las conclusiones de los autores. Después muestro mi réplica y cierro con una reflexión. La idea en una "
        f"frase: una red convolucional muy simple, que recibe la imagen en parches como un ViT, compite con los "
        f"transformers. En mi computadora, sin GPU, llegó a {R['accCMnueve']} % en CIFAR-10."]),
    ("El artículo seleccionado", 50, [
        "Es un artículo de Carnegie Mellon y del Bosch Center for AI. Cumple la consigna: es reciente, propone un "
        "modelo, lo compara con varios algoritmos (ResNet, ViT, DeiT, MLP-Mixer) y tiene resultados en dos conjuntos "
        "de datos. Uno de ellos es CIFAR-10, que es público, y eso me permitió replicarlo.",
        "Lo ubico en la Unidad 4, redes convolucionales. En la réplica sumé un modelo de cada una de las otras "
        "unidades para poder comparar. Lo conocí porque estaba en la bibliografía de Sistemas de Inteligencia "
        "Artificial del ITBA, que curso en paralelo."]),
    ("El problema: el ViT cambió dos cosas a la vez", 70, [
        "Hasta 2020 la visión por computadora era de las CNN, que son piramidales: bajan la resolución y suben los "
        "canales. El Vision Transformer cambió dos cosas a la vez: usa atención y, como la atención es cuadrática, "
        "corta la imagen en parches de 16 por 16. Además es isotrópico: misma resolución en todas las capas.",
        "La pregunta del artículo es cuál de las dos cosas explica la mejora. Si son los parches, podemos quedarnos "
        "con la convolución, que aprende con menos datos, sin pagar el costo de la atención. Eso importa en dominios "
        "con pocos datos, como las imágenes médicas de mi tesis."]),
    ("El ConvMixer", 90, [
        "El modelo tiene tres partes. Primero, una convolución con paso igual al núcleo, que corta la imagen en parches "
        "y proyecta cada uno a h canales. Después, d bloques iguales, cada uno con una convolución depthwise de núcleo "
        "grande, un filtro por canal, que mezcla información espacial, con una conexión residual, y una convolución "
        "uno por uno que mezcla los canales. Al final, pooling global y una capa lineal.",
        "Tiene cuatro hiperparámetros: ancho, profundidad, parche y núcleo. La cantidad de parámetros tiene una fórmula "
        "cerrada, la ecuación 4, que yo verifiqué contra PyTorch. Es aprendizaje supervisado con entropía cruzada y AdamW."]),
    ("Cómo encaja con la Unidad 4", 50, [
        "Todo lo que usa es de la Unidad 4: convoluciones, separables, SGD, BatchNorm y residuales. Lo que no vimos "
        "en el curso son los modelos con los que compara: el ViT, que usa atención; el MLP-Mixer, que mezcla con dos "
        "MLP, y Swin. Lo importante: el ConvMixer es un MLP-Mixer con convoluciones en lugar de MLP. Si rinde "
        "parecido, lo que importa es lo que tienen en común."]),
    ("Metodología de los autores", 60, [
        "Usan ImageNet, con un millón de imágenes, y CIFAR-10. No hay selección de características: los filtros se "
        "aprenden. El preprocesamiento es sobre todo aumentación de datos, y es muy fuerte. Entrenan con AdamW y una "
        "tasa triangular, entre 90 y 300 épocas, unos 9 días en 10 GPU.",
        "Dicen que no buscaron hiperparámetros en ImageNet, pero en CIFAR-10 sí probaron a mano ancho, profundidad, "
        "parche, núcleo y weight decay, y no informan un conjunto de validación. Desde la Unidad 1 eso es un problema: "
        "elegir mirando el test infla el resultado."]),
    ("Resultados del artículo en ImageNet", 60, [
        "El ConvMixer de 52 millones de parámetros llega a 81,4 %, por encima de la ResNet-152 entrenada con la misma "
        "receta y cerca de DeiT. El de 21 millones iguala a la ResNet-152 con un tercio de los parámetros. Pero es "
        "entre 4 y 10 veces más lento, y sólo la ResNet se entrenó en igualdad de condiciones."]),
    ("Ablaciones en CIFAR-10", 50, [
        "En CIFAR-10 muestran que el núcleo grande ayuda (93,6 % con k=3 y 95,9 % con k=9) y que los parches chicos "
        "rinden más. Lo que más me llamó la atención es la tabla de aumentación: sacar el escalado aleatorio cuesta casi "
        "10 puntos, mucho más que cualquier cambio de arquitectura. La receta de entrenamiento es parte del resultado."]),
    ("Conclusiones y limitaciones del artículo", 60, [
        "Los autores concluyen que el diseño isotrópico con parches es una plantilla potente por sí misma. Mi lectura "
        "crítica: no hay un experimento factorial que aísle los parches, las recetas de los modelos comparados son "
        "distintas, hay una sola corrida por modelo, sin barras de error, y el costo es enorme."]),
    ("Réplica: el experimento del Apéndice B, a escala de CPU", 70, [
        "Repliqué el experimento de CIFAR-10, bajando los datos de la página oficial de Toronto. Mantuve la "
        "arquitectura, el optimizador, la tasa triangular, el recorte de gradiente y el escalado aleatorio, y comparé "
        "con una ResNet con la misma receta. Lo que cambié fue por cómputo: 10 épocas en lugar de 200 y un modelo más "
        "chico, porque en mi computadora el modelo del paper tardaba tres horas y media por época.",
        "Y fui más estricto que el artículo: separé 5000 imágenes de validación y miré el test una sola vez."]),
    ("Resultado: CNN muy arriba; ConvMixer y ResNet empatan", 80, [
        f"El ConvMixer llegó a {R['accCMnueve']} % y la ResNet a {R['accResNet']} %: empatan. Abajo están los modelos "
        f"del curso: el perceptrón y la seudoinversa rondan el 40 %; la ELM llega a {R['accELM']} %; el MLP, a "
        f"{R['accMLP']} %. El MLP tiene 16 veces más parámetros que el ConvMixer y queda 25 puntos abajo: lo que "
        "importa es el sesgo inductivo de la convolución, no la cantidad de parámetros."]),
    ("Curvas de validación", 40, [
        "Con 10 épocas todos los modelos siguen aprendiendo. La curva se aplana al final sólo porque la tasa de "
        "aprendizaje llega a cero. Ahí está la mayor parte de la distancia con el 95 % del paper."]),
    ("Ablaciones frente a la Tabla 4", 80, [
        f"El efecto del parche se reproduce: con p=4 pierdo {R['difPatch']} puntos. El del núcleo no: con k=3 gano "
        f"{R['difKernelAbs']} puntos. Mi explicación es que con pocas épocas converge antes el modelo más simple, y los "
        "propios autores dicen que el efecto del núcleo grande podría depender de entrenar más. Con una sola semilla, "
        "lo tomo como indicio, no como conclusión. ConvMixer y ResNet difieren en 0,3 puntos, que está dentro del ruido."]),
    ("El costo sí se reproduce", 40, [
        f"La ResNet es {R['ratioTput']} veces más rápida en inferencia. En entrenamiento, 32 minutos contra 110. Esto "
        "coincide con el artículo: la convolución depthwise de núcleo grande está poco optimizada."]),
    ("Réplica en GPU (Google Colab)", 60, [
        "Para comprobar que la diferencia era de cómputo y no un error mío, corrí el modelo en una GPU gratuita de "
        f"Google Colab con la receta oficial de los autores para CIFAR-10. Llegó a {R.get('accColab', '--')} %, frente "
        "al 95,2 % del paper para esa configuración. O sea, la implementación es correcta y la brecha es de escala."]),
    ("CPU frente a GPU", 60, [
        f"Esta tabla me parece lo más interesante para discutir. En mi computadora cada época tardó {R.get('minEpCPU', '--')} "
        f"minutos; en la GPU, {R.get('segColab', '--')} segundos. En un tiempo parecido, la GPU entrenó un modelo casi "
        "tres veces más grande, con muchas más épocas y con la aumentación completa, y ganó 12 puntos con el mismo código.",
        "La pregunta que dejo para el debate: ¿cuánto de «el mejor modelo» es en realidad «el que más cómputo recibió»? "
        "Es la misma crítica que le hago al artículo cuando compara recetas distintas, y tiene un costado ético: sin GPU "
        "sólo se puede verificar una versión reducida."]),
    ("Reflexión crítica", 80, [
        "Lo elegí porque lo había trabajado en el ITBA, porque se podía replicar sin un cluster y porque en mi tesis "
        "tengo que elegir entre CNN y transformers con pocos datos. Podría usarse en segmentación médica, microscopía o "
        "señales como el ECG. Para entenderlo necesité las unidades 3 y 4 y, para criticarlo, la Unidad 1.",
        "En lo ético destaco el costo de cómputo, que concentra la investigación en quien tiene GPU; la "
        "reproducibilidad; los sesgos de ImageNet, y el doble uso para diagnóstico o vigilancia."]),
    ("Conclusiones", 60, [
        "El artículo tiene una idea fuerte con evidencia parcial. Mi réplica confirma que la implementación funciona, "
        "que el ConvMixer es lento y que el parche importa; no confirma, con poco entrenamiento, la ventaja del núcleo "
        "grande ni la ventaja sobre la ResNet. Replicar no es copiar un número: es poder explicar cada diferencia."]),
    ("Reproducibilidad y cierre", 20, [
        "Todo el código, las semillas y los resultados están en el repositorio y el notebook. Gracias; quedo abierto a "
        "preguntas."]),
]

PREGUNTAS = [
    ("¿Por qué tu número no da igual que el del paper?",
     "Por cómputo: 10 épocas en lugar de 200, un modelo 3 veces más chico y sin RandAugment. En GPU, con la receta de "
     f"los autores, llegué a {R.get('accColab', '--')} % frente a 95,2 %."),
    ("¿Por qué el núcleo chico te dio mejor?",
     "Con pocas épocas converge antes el modelo con menos parámetros y un sesgo más local. Es una sola semilla. Los "
     "autores sugieren que el efecto del núcleo grande depende de entrenar más o de regularizar más."),
    ("¿Qué es una convolución depthwise?",
     "Una convolución con un filtro por canal (groups igual a la cantidad de canales). No mezcla canales; eso lo hace "
     "la pointwise de 1x1 que viene después."),
    ("¿Por qué el ConvMixer es lento si tiene pocos parámetros?",
     "Porque mantiene una resolución interna alta en todas las capas y la convolución depthwise de núcleo grande "
     "tiene poca reutilización de datos y está poco optimizada. Los FLOPs no son el único costo."),
    ("¿Cómo elegiste los hiperparámetros sin mirar el test?",
     "Con 5000 imágenes de validación. Para la seudoinversa usé validación cruzada de 5 pliegues. El test se evaluó "
     "una sola vez."),
    ("¿Por qué la ELM rinde menos que el MLP?",
     "Porque sus pesos de entrada son aleatorios: sólo aprende la capa de salida. En imágenes crudas eso no alcanza "
     "para capturar la estructura espacial. Su ventaja es que se entrena en minutos con una SVD."),
    ("¿Por qué la seudoinversa necesitó un λ tan grande?",
     "Porque los píxeles vecinos están muy correlacionados y X^T X está mal condicionada. Tikhonov estabiliza la "
     "solución, y la validación cruzada eligió λ = 10000."),
    ("¿Qué es el sesgo inductivo?",
     "Los supuestos que trae el modelo antes de ver datos. En la convolución: localidad y pesos compartidos, que "
     "en imágenes valen más que los parámetros."),
    ("¿Usaste IA para hacer el trabajo?",
     "Usé un asistente para escribir código y redactar, y lo declaro. Verifiqué cada número: los parámetros contra "
     "la ecuación 4, el test una sola vez, todo generado desde los resultados guardados."),
]


def docx():
    d = Document()
    st = d.styles["Normal"]
    st.font.name = "Arial"
    st.font.size = Pt(11)
    for s in d.sections:
        s.left_margin = s.right_margin = Cm(2.2)
    h = d.add_heading("Guion de la exposición — Trabajo Final", 0)
    h.runs[0].font.color.rgb = AZUL
    d.add_paragraph("Tópicos en Aprendizaje Automático (UNAJ) · Marcos David Kantor · "
                    "Análisis crítico de Trockman y Kolter (2023)")
    total = sum(t for _, t, _ in DIAPOS)
    d.add_paragraph(f"Duración objetivo: {total // 60} min {total % 60:02d} s de exposición + debate (30 min en total). "
                    "Regla: el título de cada diapositiva ya es el mensaje; decirlo primero y después mostrar la prueba.")
    acum = 0
    for i, (tit, seg, pars) in enumerate(DIAPOS, 1):
        acum += seg
        p = d.add_heading(f"{i}. {tit}   [{acum // 60}:{acum % 60:02d}]", 2)
        p.runs[0].font.color.rgb = AZUL
        for t in pars:
            d.add_paragraph(t)
    h = d.add_heading("Preguntas probables y respuestas", 1)
    h.runs[0].font.color.rgb = AZUL
    for q, a in PREGUNTAS:
        p = d.add_paragraph()
        p.add_run(q).bold = True
        d.add_paragraph(a)
    d.save(AQUI / "TF_guion.docx")


def tex():
    esc = lambda t: (t.replace("\\", "\\textbackslash{}").replace("%", "\\%").replace("&", "\\&").replace("_", "\\_")
                     .replace("#", "\\#").replace("^", "\\^{}").replace("λ", "$\\lambda$"))
    total = sum(t for _, t, _ in DIAPOS)
    L = [r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage{lmodern}
\usepackage[spanish,es-noshorthands]{babel}\usepackage[margin=2.2cm]{geometry}
\usepackage{xcolor,titlesec,enumitem,graphicx}
\definecolor{unaj}{HTML}{003366}
\titleformat{\section}{\Large\bfseries\color{unaj}}{}{0em}{}
\titleformat{\subsection}{\large\bfseries\color{unaj}}{}{0em}{}
\setlength{\parskip}{0.5em}\setlength{\parindent}{0pt}
\begin{document}
\includegraphics[height=1.2cm]{figuras/logo_unaj.png}\hfill{\bfseries Doctorado en Ciencia y Tecnología}\par\medskip
{\LARGE\bfseries\color{unaj} Guion de la exposición}\par
Tópicos en Aprendizaje Automático (UNAJ) --- Marcos David Kantor --- Análisis crítico de Trockman y Kolter (2023)\par
""", f"Duración objetivo: {total // 60} min {total % 60:02d} s de exposición y debate hasta completar 30 min. "
         "Regla: el título de cada diapositiva ya es el mensaje; decirlo primero y después mostrar la prueba.\n"]
    acum = 0
    for i, (tit, seg, pars) in enumerate(DIAPOS, 1):
        acum += seg
        L.append(f"\\subsection{{{i}. {esc(tit)} \\hfill \\normalfont\\small [{acum // 60}:{acum % 60:02d}]}}\n")
        L += [esc(t) + "\n\n" for t in pars]
    L.append("\\section{Preguntas probables y respuestas}\n\\begin{itemize}[leftmargin=1.2em]\n")
    for q, a in PREGUNTAS:
        L.append(f"\\item \\textbf{{{esc(q)}}}\\\\ {esc(a)}\n")
    L.append("\\end{itemize}\n\\end{document}\n")
    (AQUI / "TF_guion.tex").write_text("".join(L), encoding="utf8")
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "TF_guion.tex"], cwd=AQUI, capture_output=True)


if __name__ == "__main__":
    docx()
    tex()
    print("TF_guion.docx y TF_guion.pdf generados;", sum(t for _, t, _ in DIAPOS) // 60, "min")
