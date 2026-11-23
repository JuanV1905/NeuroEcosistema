import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from simulacion import EcosistemaSimulacion


def correr_experimento_individual(escenario, modo_p, modo_d, arquitectura, semilla, generaciones=35):
    sim = EcosistemaSimulacion(
        escenario=escenario,
        modo_presas=modo_p,
        modo_depredadores=modo_d,
        semilla=semilla,
        arquitectura=arquitectura
    )
    historial_gen = []

    for gen in range(1, generaciones + 1):
        while not sim.paso():
            pass
        metricas = sim.obtener_metricas()
        historial_gen.append(metricas)

        if gen < generaciones:
            sim.evolucionar_generacion()

    return historial_gen[0], historial_gen[-1], historial_gen, sim


def ejecutar_suite_experimental():
    os.makedirs("resultados", exist_ok=True)
    semillas = [10, 42, 99]
    escenarios = ["A", "B", "C"]
    arq_base = (5, 8, 6, 2)
    arq_grande = (5, 16, 12, 2)

    registros_csv = []
    curvas_aprendizaje = {esc: [] for esc in escenarios}

    print("=== INICIANDO EXPERIMENTACIÓN REPRODUCIBLE NEUROECOSISTEMA ===")

    # 1. Evaluación en los 3 escenarios: Antes vs Después del entrenamiento (Arquitectura Base)
    for esc in escenarios:
        print(f"\n[Escenario {esc}] Evaluando Red Neuronal {arq_base} (3 semillas, 35 generaciones)...")
        for sem in semillas:
            antes, despues, hist, sim_final = correr_experimento_individual(
                esc, "neuronal", "neuronal", arq_base, sem, generaciones=35
            )
            curvas_aprendizaje[esc].append([h["supervivencia_media_pasos"] for h in hist])

            for etiqueta, m in [("Neuronal_Gen1_Antes", antes), ("Neuronal_Gen35_Despues", despues)]:
                registros_csv.append({
                    "Experimento": "Antes_vs_Despues",
                    "Escenario": esc,
                    "Configuracion": etiqueta,
                    "Arquitectura": str(arq_base),
                    "Semilla": sem,
                    "Supervivencia_Presas_Pct": m["supervivencia_presas_pct"],
                    "Supervivencia_Media_Pasos": m["supervivencia_media_pasos"],
                    "Alimento_Consumido": m["alimento_presas_total"],
                    "Capturas_Depredadores": m["capturas_depredadores"],
                    "Energia_Final_Presas": m["energia_final_presas"],
                    "Energia_Final_Deprs": m["energia_final_deprs"],
                    "Tiempo_Medio_Captura": m["tiempo_medio_captura"]
                })

            # Guardar mejores modelos entrenados del Escenario C (más exigente) con semilla 42
            if esc == "C" and sem == 42:
                mejor_p = max(sim_final.presas, key=lambda p: p.calcular_fitness())
                mejor_d = max(sim_final.depredadores, key=lambda d: d.calcular_fitness())
                mejor_p.cerebro.guardar("resultados/modelo_presa_entrenado.npz")
                mejor_d.cerebro.guardar("resultados/modelo_depredador_entrenado.npz")

    # 2. Comparación entre dos arquitecturas: (5, 8, 6, 2) vs (5, 16, 12, 2)
    print(f"\n[Comparación Arquitecturas] Evaluando Red Amplia {arq_grande} en los 3 escenarios...")
    for esc in escenarios:
        for sem in semillas:
            _, despues_grande, _, _ = correr_experimento_individual(
                esc, "neuronal", "neuronal", arq_grande, sem, generaciones=35
            )
            registros_csv.append({
                "Experimento": "Comparacion_Arquitecturas",
                "Escenario": esc,
                "Configuracion": "Neuronal_Amplia_Gen35",
                "Arquitectura": str(arq_grande),
                "Semilla": sem,
                "Supervivencia_Presas_Pct": despues_grande["supervivencia_presas_pct"],
                "Supervivencia_Media_Pasos": despues_grande["supervivencia_media_pasos"],
                "Alimento_Consumido": despues_grande["alimento_presas_total"],
                "Capturas_Depredadores": despues_grande["capturas_depredadores"],
                "Energia_Final_Presas": despues_grande["energia_final_presas"],
                "Energia_Final_Deprs": despues_grande["energia_final_deprs"],
                "Tiempo_Medio_Captura": despues_grande["tiempo_medio_captura"]
            })

    # 3. Comparación contra Agentes de Referencia (Reglas y Aleatorio)
    print("\n[Líneas Base] Evaluando Agente Basado en Reglas y Agente Aleatorio...")
    for esc in escenarios:
        for modo_ref in ["reglas", "aleatorio"]:
            for sem in semillas:
                m_ref, _, _, _ = correr_experimento_individual(
                    esc, modo_ref, modo_ref, arq_base, sem, generaciones=1
                )
                registros_csv.append({
                    "Experimento": "Linea_Base_Referencia",
                    "Escenario": esc,
                    "Configuracion": f"Referencia_{modo_ref.capitalize()}",
                    "Arquitectura": "N/A",
                    "Semilla": sem,
                    "Supervivencia_Presas_Pct": m_ref["supervivencia_presas_pct"],
                    "Supervivencia_Media_Pasos": m_ref["supervivencia_media_pasos"],
                    "Alimento_Consumido": m_ref["alimento_presas_total"],
                    "Capturas_Depredadores": m_ref["capturas_depredadores"],
                    "Energia_Final_Presas": m_ref["energia_final_presas"],
                    "Energia_Final_Deprs": m_ref["energia_final_deprs"],
                    "Tiempo_Medio_Captura": m_ref["tiempo_medio_captura"]
                })

    # Guardar CSV completo de evidencia experimental
    ruta_csv = "resultados/evidencia_experimental.csv"
    with open(ruta_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(registros_csv[0].keys()))
        writer.writeheader()
        writer.writerows(registros_csv)
    print(f"\n[OK] Datos guardados en: {ruta_csv}")

    # Generar Gráfica 1: Curvas de evolución por escenario
    plt.figure(figsize=(10, 5))
    for esc in escenarios:
        matriz = np.array(curvas_aprendizaje[esc])
        media = np.mean(matriz, axis=0)
        std = np.std(matriz, axis=0)
        gens = np.arange(1, len(media) + 1)
        plt.plot(gens, media, label=f"Escenario {esc} (Media 3 semillas)", linewidth=2)
        plt.fill_between(gens, media - std, media + std, alpha=0.15)

    plt.title("Evolución de la Supervivencia Media de Presas por Generación")
    plt.xlabel("Generación")
    plt.ylabel("Supervivencia Media (pasos)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("resultados/grafica_evolucion_escenarios.png", dpi=150)
    plt.close()

    # Generar Gráfica 2: Comparativa final entre estrategias (Escenario C)
    categorias = ["Neuronal_Gen1_Antes", "Neuronal_Gen35_Despues", "Neuronal_Amplia_Gen35", "Referencia_Reglas", "Referencia_Aleatorio"]
    etiquetas_cortas = ["Red Gen 1\n(Sin entrenar)", "Red Base\n(Gen 35)", "Red Amplia\n(Gen 35)", "Agente\nReglas", "Agente\nAleatorio"]
    medias_sup = []
    medias_alim = []

    for cat in categorias:
        filas = [r for r in registros_csv if r["Escenario"] == "C" and r["Configuracion"] == cat]
        medias_sup.append(np.mean([r["Supervivencia_Media_Pasos"] for r in filas]))
        medias_alim.append(np.mean([r["Alimento_Consumido"] for r in filas]))

    x = np.arange(len(etiquetas_cortas))
    ancho_barra = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(x - ancho_barra / 2, medias_sup, ancho_barra, label="Supervivencia Media (pasos)", color="#3b82f6")
    ax1.set_ylabel("Supervivencia Media (pasos)", color="#1d4ed8")
    ax1.set_xticks(x)
    ax1.set_xticklabels(etiquetas_cortas)

    ax2 = ax1.twinx()
    ax2.bar(x + ancho_barra / 2, medias_alim, ancho_barra, label="Alimento Consumido", color="#10b981")
    ax2.set_ylabel("Alimento Consumido (plantas)", color="#047857")

    plt.title("Comparación de Desempeño en Escenario C (Alta Depredación y Obstáculos)")
    fig.tight_layout()
    plt.savefig("resultados/grafica_comparacion_estrategias.png", dpi=150)
    plt.close()
    print("[OK] Gráficas guardadas en resultados/grafica_evolucion_escenarios.png y grafica_comparacion_estrategias.png")


if __name__ == "__main__":
    ejecutar_suite_experimental()
