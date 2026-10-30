# Cotas a la conectividad crítica del modelo de Kuramoto

Búsqueda de grafos densos con mínimos espurios (equilibrios estables no sincronizados) del modelo de Kuramoto
homogéneo, para subir la cota inferior de la conectividad crítica μ_c, con μ(G) = (δ_min(G) + 1) / n.

Todo está en un único notebook, `pipeline_kuramoto.ipynb`:

**GNN → SDP por clusters → CMA-ES + SDP + pulido convexo (filtro TOPE) → compresión → certificación con intervalos.**

## Resultado

El pipeline encuentra un template de 7 clusters (tamaños y densidades racionales) que se certifica con aritmética de
intervalos a 80 dígitos: test de Krawczyk para la existencia de un equilibrio exacto, y D_a > 0 y Cholesky con
intervalos para su estabilidad estricta. Su conectividad exacta da

**μ_c ≥ 0.6919305**

por encima de 11/16 = 0.6875 (Canale) y de los grafos twisted (0.683876). El paso del template a una familia de
grafos finitos usa el Lema 2 de O'Keeffe (blow-up con conjuntos de saltos aleatorios).

## Reproducibilidad

Hay dos niveles, y los dos están en el mismo notebook:

1. **Verificar el resultado (segundos, en cualquier máquina).** La sección 0b carga `datos/template_reportado.json`
   y lo certifica con intervalos. La verificación es exacta, así que da el mismo μ en cualquier computadora.
2. **Encontrarlo (unas horas).** La corrida completa (`RAPIDO = False`, `SEMILLA = 0`) busca desde cero, certifica en
   la sección 9b y guarda el mejor template en `resultados/template_certificado.json`. Toda la aleatoriedad sale de
   `SEMILLA` y no hay límite de tiempo, así que en una misma máquina la corrida se repite igual. La búsqueda es muy
   sensible a los últimos decimales: con otro procesador o con otras versiones de numpy o Clarabel, algunos empates
   se resuelven distinto y el camino puede cambiar. Por eso el resultado se reporta junto con su certificado (paso 1),
   que no depende de la búsqueda.

## Contenido

| Archivo | Qué es |
|---|---|
| `pipeline_kuramoto.ipynb` | El pipeline completo, sin salidas. Es lo que hay que correr. |
| `pipeline_kuramoto.py` | El mismo código como script de Python (celdas `# %%`, en el mismo orden). |
| `datos/template_reportado.json` | El template certificado: tamaños X_a/D, densidades Q_ab/D, fases y μ exacto (fracción). |
| `datos/` (resto) | Punto de partida: dataset etiquetado de grafos chicos, pesos de la GNN base y dos diseños de O'Keeffe que se usan como control. |
| `requirements.txt` | Dependencias. |

## Cómo correrlo

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook pipeline_kuramoto.ipynb               # o: python pipeline_kuramoto.py
```

1. Para verificar el resultado alcanza con correr las celdas de las secciones 0 y 0b.
2. Para probar la instalación completa, poner `RAPIDO = True` en la sección 0 y correr todo (unos minutos).
3. Con `RAPIDO = False` se hace la corrida completa (unas horas con 8 núcleos). Todo lo que produce queda en
   `resultados/`, y el progreso en `resultados/registro.txt`. Al final imprime `RESULTADO: μ_c ≥ ...`.

Todos los parámetros (umbrales, tamaños, márgenes, presupuestos) están en la sección 0 del notebook, con una tabla que
explica cada uno. Si el paralelismo falla (por ejemplo, en Windows), poner `N_JOBS = 1`.
