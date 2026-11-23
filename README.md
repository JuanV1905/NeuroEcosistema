# NeuroEcosistema – Depredadores y Presas
**Universidad Santo Tomás – Seccional Tunja**  
Facultad de Ingeniería de Sistemas | Electiva I de Profundización – Inteligencia Artificial

## Descripción
Simulación 2D de un ecosistema autónomo donde dos poblaciones (**presas** y **depredadores**) toman decisiones de movimiento y gestión energética mediante **Perceptrones Multicapa (MLP)** entrenados por **neuroevolución (selección elitista y mutación gaussiana)**.

## Estructura del Proyecto
- `red_neuronal.py`: Implementación desde cero en NumPy de una red neuronal multicapa con 2 capas ocultas, activación `tanh`, mutación y serialización `.npz`.
- `agentes.py`: Definición sensorial, cinemática, consumo energético y políticas de control (`neuronal`, `reglas`, `aleatorio`) para presas, depredadores y obstáculos.
- `simulacion.py`: Motor de simulación 2D con interfaz gráfica en Pygame, telemetría en vivo, inspección de agentes y soporte para los escenarios A, B y C.
- `experimentos.py`: Suite automatizada y reproducible que ejecuta las pruebas obligatorias (3 escenarios × 3 semillas), exporta `evidencia_experimental.csv` y genera gráficas comparativas.
- `resultados/`: Carpeta con modelos entrenados (`.npz`), datos tabulares (`.csv`) y gráficas (`.png`).

## Instalación y Ejecución
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
```