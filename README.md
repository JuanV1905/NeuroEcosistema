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

## Instalación y Requisitos
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
> Si PowerShell restringe la ejecución de scripts, ejecutar antes: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

## Ejecución

### 1. Simulación Interactiva
Para iniciar la interfaz visual en Pygame:
```powershell
python simulacion.py
```

**Controles dentro de la simulación:**
- `[T]`: Activar/desactivar modo turbo (25x) para acelerar el entrenamiento evolutivo.
- `[1]`: Cargar Escenario A (abundancia de plantas, 4 depredadores).
- `[2]`: Cargar Escenario B (escasez de plantas, 5 depredadores).
- `[3]`: Cargar Escenario C (10 depredadores y 6 obstáculos).
- `[R]`: Alternar a control heurístico basado en reglas (if/else).
- `[G]`: Guardar pesos del mejor individuo en `resultados/` (.npz).
- `[Clic izquierdo]`: Inspeccionar un agente en vivo (ver energía y sus 5 sensores).

### 2. Suite Experimental
Para ejecutar las pruebas reproducibles (3 escenarios x 3 semillas x 35 generaciones) y regenerar las tablas y figuras:
```powershell
python experimentos.py
```
