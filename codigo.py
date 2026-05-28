import random
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# ======================================
# CONSTANTES
# ======================================

PARED  = 0
CAMINO = 1
ORIGEN = 2
META   = 3

# ======================================
# GENERAR LABERINTO
# ======================================

def generar_laberinto(ancho, alto, probabilidad_romper):

    if ancho % 2 == 0:
        ancho += 1
    if alto % 2 == 0:
        alto += 1

    laberinto = [[PARED for _ in range(ancho)] for _ in range(alto)]

    def excavar(x, y):
        laberinto[y][x] = CAMINO
        direcciones = [(0, -2), (2, 0), (0, 2), (-2, 0)]
        random.shuffle(direcciones)
        for dx, dy in direcciones:
            nx, ny = x + dx, y + dy
            if (1 <= nx < ancho - 1 and 1 <= ny < alto - 1
                    and laberinto[ny][nx] == PARED):
                laberinto[y + dy // 2][x + dx // 2] = CAMINO
                excavar(nx, ny)

    inicio_x = random.randrange(1, ancho, 2)
    inicio_y = random.randrange(1, alto, 2)
    excavar(inicio_x, inicio_y)

    for y in range(1, alto - 1):
        for x in range(1, ancho - 1):
            if laberinto[y][x] == PARED:
                horizontal = (laberinto[y][x-1] == CAMINO and laberinto[y][x+1] == CAMINO)
                vertical   = (laberinto[y-1][x] == CAMINO and laberinto[y+1][x] == CAMINO)
                if (horizontal or vertical) and random.random() < probabilidad_romper:
                    laberinto[y][x] = CAMINO

    laberinto[1][0]          = ORIGEN
    laberinto[alto-2][ancho-1] = META

    return laberinto


def obtener_posicion(laberinto, valor):
    for f, fila in enumerate(laberinto):
        for c, celda in enumerate(fila):
            if celda == valor:
                return (f, c)
    return None


# ======================================
# ACO
# ======================================

def obtener_vecinos(laberinto, nodo):
    filas    = len(laberinto)
    columnas = len(laberinto[0])
    r, c     = nodo
    vecinos  = []
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r + dr, c + dc
        if (0 <= nr < filas and 0 <= nc < columnas
                and laberinto[nr][nc] != PARED):
            vecinos.append((nr, nc))
    return vecinos


def heuristica(nodo, meta):
    r, c   = nodo
    gr, gc = meta
    return 1 / (abs(gr - r) + abs(gc - c) + 1)


def probabilidades_transicion(actual, permitidos, feromonas, meta, alpha, beta):
    valores, suma = [], 0
    for vecino in permitidos:
        tau   = feromonas[(actual, vecino)]
        eta   = heuristica(vecino, meta)
        valor = (tau ** alpha) * (eta ** beta)
        valores.append((vecino, valor))
        suma += valor
    return [(v, (val / suma if suma > 0 else 0)) for v, val in valores]


def elegir_siguiente(probabilidades):
    r, acumulado = random.random(), 0
    for nodo, p in probabilidades:
        acumulado += p
        if r <= acumulado:
            return nodo
    return probabilidades[-1][0]


def construir_solucion(laberinto, inicio, meta, feromonas, alpha, beta):
    actual    = inicio
    camino    = [actual]
    visitados = {actual}
    max_pasos = len(laberinto) * len(laberinto[0]) * 2

    while actual != meta and len(camino) < max_pasos:
        vecinos = [v for v in obtener_vecinos(laberinto, actual) if v not in visitados]
        if not vecinos:
            break
        probs    = probabilidades_transicion(actual, vecinos, feromonas, meta, alpha, beta)
        siguiente = elegir_siguiente(probs)
        camino.append(siguiente)
        visitados.add(siguiente)
        actual = siguiente

    return camino if actual == meta else None


def actualizar_feromonas(feromonas, caminos, evaporacion, Q):
    for arista in list(feromonas.keys()):
        feromonas[arista] *= (1 - evaporacion)
    for camino in caminos:
        if camino is None:
            continue
        deposito = Q / len(camino)
        for i in range(len(camino) - 1):
            feromonas[(camino[i], camino[i+1])] += deposito


# ======================================
# ANIMACIÓN CON MATPLOTLIB
# ======================================

def ejecutar_aco_visual():

    laberinto = generar_laberinto(ancho=31, alto=31, probabilidad_romper=0.15)
    inicio    = obtener_posicion(laberinto, ORIGEN)
    meta      = obtener_posicion(laberinto, META)

    hormigas    = 50
    iteraciones = 60
    alpha       = 1
    beta        = 5
    evaporacion = 0.25
    Q           = 100

    feromonas      = defaultdict(lambda: 1.0)
    mejor_camino   = None
    mejor_longitud = float("inf")

    # ---- correr todas las iteraciones primero ----
    historial_feromonas = []
    historial_caminos   = []

    for iteracion in range(iteraciones):
        caminos = []
        for _ in range(hormigas):
            camino = construir_solucion(laberinto, inicio, meta, feromonas, alpha, beta)
            caminos.append(camino)
            if camino is not None and len(camino) < mejor_longitud:
                mejor_camino   = camino
                mejor_longitud = len(camino)

        actualizar_feromonas(feromonas, caminos, evaporacion, Q)
        historial_feromonas.append(dict(feromonas))
        historial_caminos.append(mejor_camino[:] if mejor_camino else None)
        print(f"Iteración {iteracion+1}/{iteraciones}  |  Mejor longitud: {mejor_longitud}")

    # ---- construir imagen base del laberinto ----
    filas    = len(laberinto)
    columnas = len(laberinto[0])

    # imagen RGB base
    base_img = np.ones((filas, columnas, 3))
    for r in range(filas):
        for c in range(columnas):
            if laberinto[r][c] == PARED:
                base_img[r, c] = [0, 0, 0]        # negro
            elif laberinto[r][c] == ORIGEN:
                base_img[r, c] = [0, 0.8, 0]      # verde
            elif laberinto[r][c] == META:
                base_img[r, c] = [0.9, 0, 0]      # rojo

    # ---- figura ----
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, columnas)
    ax.set_ylim(0, filas)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("#1a1a2e")

    im = ax.imshow(
        base_img,
        origin="upper",
        extent=[0, columnas, 0, filas],
        zorder=0
    )

    titulo = ax.set_title(
        "",
        color="white",
        fontsize=13,
        pad=10
    )

    lineas_feromonas = []
    parches_camino   = []

    def actualizar(indice):
        nonlocal lineas_feromonas, parches_camino

        # limpiar frame anterior
        for ln in lineas_feromonas:
            ln.remove()
        for p in parches_camino:
            p.remove()
        lineas_feromonas = []
        parches_camino   = []

        fer = historial_feromonas[indice]
        if fer:
            max_val = max(fer.values())
            for (r1, c1), (r2, c2) in [arista for arista in fer]:
                # omitir feromonas débiles para no saturar visualmente
                val = fer[((r1,c1),(r2,c2))]
                t   = val / (max_val + 1e-9)
                if t < 0.05:
                    continue
                # coordenadas: imshow con origin=upper → y = filas - r
                x1 = c1 + 0.5;  y1 = filas - r1 - 0.5
                x2 = c2 + 0.5;  y2 = filas - r2 - 0.5
                color  = (t, 0.55 * t, 0)        # naranja normalizado
                grosor = max(0.5, t * 3.5)
                ln, = ax.plot([x1, x2], [y1, y2],
                              color=color, linewidth=grosor,
                              alpha=min(t + 0.2, 1.0), zorder=1)
                lineas_feromonas.append(ln)

        camino = historial_caminos[indice]
        if camino:
            for r, c in camino:
                if laberinto[r][c] in [ORIGEN, META]:
                    continue
                y_patch = filas - r - 1
                p = patches.Rectangle(
                    (c, y_patch), 1, 1,
                    linewidth=0,
                    facecolor="yellow",
                    alpha=0.75,
                    zorder=2
                )
                ax.add_patch(p)
                parches_camino.append(p)

        titulo.set_text(
            f"Iteración: {indice+1}/{iteraciones}   |   "
            f"Mejor longitud: {mejor_longitud}"
        )
        return lineas_feromonas + parches_camino + [titulo]

    anim = FuncAnimation(
        fig,
        actualizar,
        frames=iteraciones,
        interval=500,
        blit=False,
        repeat=False
    )

    plt.tight_layout()

    # En Jupyter muestra inline; fuera de Jupyter abre ventana
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            plt.close()
            return HTML(anim.to_jshtml())
        else:
            plt.show()
    except ImportError:
        plt.show()


# ======================================
# MAIN
# ======================================

resultado = ejecutar_aco_visual()

# En Jupyter: ejecuta la celda y el resultado se muestra automáticamente
try:
    from IPython.display import display
    if resultado is not None:
        display(resultado)
except ImportError:
    pass
