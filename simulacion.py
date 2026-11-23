import math
import random
import numpy as np
import pygame
from agentes import Presa, Depredador, Obstaculo
from red_neuronal import RedNeuronalMLP


class EcosistemaSimulacion:
    def __init__(self, ancho=900, alto=620, escenario="A", modo_presas="neuronal",
                 modo_depredadores="neuronal", semilla=42, arquitectura=(5, 8, 6, 2)):
        self.ancho = ancho
        self.alto = alto
        self.escenario = escenario
        self.modo_presas = modo_presas
        self.modo_depredadores = modo_depredadores
        self.arquitectura = arquitectura
        self.semilla = semilla

        self.fijar_semilla(semilla)
        self.generacion = 1
        self.paso_actual = 0
        self.max_pasos_episodio = 600

        # Configurar parámetros según los 3 escenarios obligatorios
        self._configurar_escenario()
        self.inicializar_poblaciones()

    def fijar_semilla(self, semilla):
        random.seed(semilla)
        np.random.seed(semilla)

    def _configurar_escenario(self):
        # Escenario A: recursos suficientes y pocos depredadores
        if self.escenario == "A":
            self.n_presas = 24
            self.n_depredadores = 4
            self.comida_inicial = 45
            self.tasa_regeneracion_comida = 0.35
            self.n_obstaculos = 0
        # Escenario B: escasez de alimento
        elif self.escenario == "B":
            self.n_presas = 24
            self.n_depredadores = 5
            self.comida_inicial = 15
            self.tasa_regeneracion_comida = 0.08
            self.n_obstaculos = 0
        # Escenario C: aumento de presión de depredación y obstáculos
        elif self.escenario == "C":
            self.n_presas = 24
            self.n_depredadores = 10
            self.comida_inicial = 30
            self.tasa_regeneracion_comida = 0.22
            self.n_obstaculos = 6
        else:
            raise ValueError("Escenario debe ser 'A', 'B' o 'C'")

    def _generar_comida(self):
        return (random.uniform(25, self.ancho - 25), random.uniform(25, self.alto - 25))

    def inicializar_poblaciones(self, cerebros_presas=None, cerebros_depredadores=None):
        self.paso_actual = 0
        self.capturas_totales = 0
        self.tiempos_captura = []

        self.obstaculos = []
        for _ in range(self.n_obstaculos):
            ox = random.uniform(120, self.ancho - 120)
            oy = random.uniform(100, self.alto - 100)
            self.obstaculos.append(Obstaculo(ox, oy, radio=random.uniform(24, 38)))

        self.comidas = [self._generar_comida() for _ in range(self.comida_inicial)]

        self.presas = []
        for i in range(self.n_presas):
            p = Presa(random.uniform(30, self.ancho - 30),
                      random.uniform(30, self.alto - 30),
                      self.ancho, self.alto,
                      modo=self.modo_presas, arquitectura=self.arquitectura)
            if cerebros_presas and i < len(cerebros_presas) and p.modo == "neuronal":
                p.cerebro = cerebros_presas[i]
            self.presas.append(p)

        self.depredadores = []
        for i in range(self.n_depredadores):
            d = Depredador(random.uniform(30, self.ancho - 30),
                           random.uniform(30, self.alto - 30),
                           self.ancho, self.alto,
                           modo=self.modo_depredadores, arquitectura=self.arquitectura)
            if cerebros_depredadores and i < len(cerebros_depredadores) and d.modo == "neuronal":
                d.cerebro = cerebros_depredadores[i]
            self.depredadores.append(d)

    def paso(self):
        self.paso_actual += 1

        # Regeneración periódica de plantas
        if random.random() < self.tasa_regeneracion_comida and len(self.comidas) < 70:
            self.comidas.append(self._generar_comida())

        # 1. Mover presas y verificar consumo de alimento
        for p in self.presas:
            if not p.vivo:
                continue
            p.decidir_y_actuar(self.comidas, self.depredadores, self.obstaculos)
            for idx in range(len(self.comidas) - 1, -1, -1):
                cx, cy = self.comidas[idx]
                if math.hypot(p.x - cx, p.y - cy) < 11.0:
                    p.energia = min(p.energia_max, p.energia + 32.0)
                    p.alimento_consumido += 1
                    self.comidas.pop(idx)
                    break

        # 2. Mover depredadores y verificar capturas
        for d in self.depredadores:
            if not d.vivo:
                continue
            d.decidir_y_actuar(self.presas, self.depredadores, self.obstaculos)
            for p in self.presas:
                if p.vivo and math.hypot(d.x - p.x, d.y - p.y) < 13.0:
                    p.vivo = False
                    p.energia = 0.0
                    d.energia = min(d.energia_max, d.energia + 55.0)
                    d.alimento_consumido += 1
                    self.capturas_totales += 1
                    self.tiempos_captura.append(self.paso_actual)
                    break

        presas_vivas = sum(1 for p in self.presas if p.vivo)
        deprs_vivos = sum(1 for d in self.depredadores if d.vivo)

        # Fin de episodio si se acaba el tiempo o muere toda una especie
        episodio_terminado = (
            self.paso_actual >= self.max_pasos_episodio
            or presas_vivas == 0
            or deprs_vivos == 0
        )
        return episodio_terminado

    def obtener_metricas(self):
        presas_vivas = [p for p in self.presas if p.vivo]
        deprs_vivos = [d for d in self.depredadores if d.vivo]

        return {
            "generacion": self.generacion,
            "pasos": self.paso_actual,
            "supervivencia_presas_pct": round(100.0 * len(presas_vivas) / self.n_presas, 2),
            "supervivencia_media_pasos": round(np.mean([p.tiempo_vivo for p in self.presas]), 2),
            "alimento_presas_total": sum(p.alimento_consumido for p in self.presas),
            "capturas_depredadores": self.capturas_totales,
            "energia_final_presas": round(np.mean([p.energia for p in self.presas]), 2),
            "energia_final_deprs": round(np.mean([d.energia for d in self.depredadores]), 2),
            "tiempo_medio_captura": round(np.mean(self.tiempos_captura), 2) if self.tiempos_captura else self.max_pasos_episodio
        }

    def evolucionar_generacion(self):
        """Aplica selección elitista y mutación (neuroevolución) para la siguiente generación."""
        nuevos_cerebros_presas = None
        if self.modo_presas == "neuronal":
            ordenadas = sorted(self.presas, key=lambda p: p.calcular_fitness(), reverse=True)
            elite = ordenadas[:max(2, self.n_presas // 4)]
            nuevos_cerebros_presas = [e.cerebro.clonar_y_mutar(0.0, 0.0) for e in elite[:2]]
            while len(nuevos_cerebros_presas) < self.n_presas:
                padre = random.choice(elite)
                nuevos_cerebros_presas.append(padre.cerebro.clonar_y_mutar(0.18, 0.35))

        nuevos_cerebros_deprs = None
        if self.modo_depredadores == "neuronal":
            ordenados = sorted(self.depredadores, key=lambda d: d.calcular_fitness(), reverse=True)
            elite = ordenados[:max(2, self.n_depredadores // 2)]
            nuevos_cerebros_deprs = [elite[0].cerebro.clonar_y_mutar(0.0, 0.0)]
            while len(nuevos_cerebros_deprs) < self.n_depredadores:
                padre = random.choice(elite)
                nuevos_cerebros_deprs.append(padre.cerebro.clonar_y_mutar(0.18, 0.35))

        self.generacion += 1
        self.inicializar_poblaciones(nuevos_cerebros_presas, nuevos_cerebros_deprs)


def ejecutar_visualizacion():
    pygame.init()
    ancho_mapa, alto_mapa = 900, 620
    panel_lateral = 310
    pantalla = pygame.display.set_mode((ancho_mapa + panel_lateral, alto_mapa))
    pygame.display.set_caption("NeuroEcosistema USTA - Depredadores y Presas")
    reloj = pygame.time.Clock()
    fuente = pygame.font.SysFont("Consolas", 15)
    fuente_titulo = pygame.font.SysFont("Consolas", 18, bold=True)

    sim = EcosistemaSimulacion(ancho=ancho_mapa, alto=alto_mapa, escenario="A",
                               modo_presas="neuronal", modo_depredadores="neuronal", semilla=42)
    modo_turbo = False
    agente_inspeccionado = None

    ejecutando = True
    while ejecutando:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                ejecutando = False
            elif evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_t:
                    modo_turbo = not modo_turbo
                elif evento.key == pygame.K_1:
                    sim = EcosistemaSimulacion(ancho_mapa, alto_mapa, "A", "neuronal", "neuronal", 42)
                elif evento.key == pygame.K_2:
                    sim = EcosistemaSimulacion(ancho_mapa, alto_mapa, "B", "neuronal", "neuronal", 42)
                elif evento.key == pygame.K_3:
                    sim = EcosistemaSimulacion(ancho_mapa, alto_mapa, "C", "neuronal", "neuronal", 42)
                elif evento.key == pygame.K_r:
                    sim = EcosistemaSimulacion(ancho_mapa, alto_mapa, sim.escenario, "reglas", "reglas", 42)
                elif evento.key == pygame.K_g:
                    # Guardar los mejores cerebros entrenados
                    mejor_p = max(sim.presas, key=lambda p: p.calcular_fitness())
                    mejor_d = max(sim.depredadores, key=lambda d: d.calcular_fitness())
                    if mejor_p.cerebro:
                        mejor_p.cerebro.guardar("resultados/mejor_presa.npz")
                    if mejor_d.cerebro:
                        mejor_d.cerebro.guardar("resultados/mejor_depredador.npz")
                    print("Modelos guardados en resultados/mejor_presa.npz y mejor_depredador.npz")
            elif evento.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                todos = [a for a in (sim.presas + sim.depredadores) if a.vivo]
                if todos:
                    agente_inspeccionado = min(todos, key=lambda a: math.hypot(a.x - mx, a.y - my))

        pasos_por_frame = 25 if modo_turbo else 1
        for _ in range(pasos_por_frame):
            if sim.paso():
                sim.evolucionar_generacion()

        # Renderizado del mapa
        pantalla.fill((22, 26, 33))
        pygame.draw.rect(pantalla, (32, 38, 48), (ancho_mapa, 0, panel_lateral, alto_mapa))

        # Obstáculos
        for obs in sim.obstaculos:
            pygame.draw.circle(pantalla, (100, 110, 125), (int(obs.x), int(obs.y)), int(obs.radio))

        # Comida (plantas)
        for cx, cy in sim.comidas:
            pygame.draw.circle(pantalla, (90, 220, 120), (int(cx), int(cy)), 4)

        # Presas (azul/cian)
        for p in sim.presas:
            if p.vivo:
                pygame.draw.circle(pantalla, (65, 185, 255), (int(p.x), int(p.y)), 6)
                fx = p.x + math.cos(p.angulo) * 11
                fy = p.y + math.sin(p.angulo) * 11
                pygame.draw.line(pantalla, (200, 240, 255), (p.x, p.y), (fx, fy), 2)

        # Depredadores (rojo/naranja)
        for d in sim.depredadores:
            if d.vivo:
                pygame.draw.circle(pantalla, (255, 85, 85), (int(d.x), int(d.y)), 8)
                fx = d.x + math.cos(d.angulo) * 14
                fy = d.y + math.sin(d.angulo) * 14
                pygame.draw.line(pantalla, (255, 210, 180), (d.x, d.y), (fx, fy), 2)

        # Resaltar agente seleccionado con clic
        if agente_inspeccionado and agente_inspeccionado.vivo:
            pygame.draw.circle(pantalla, (255, 230, 80),
                               (int(agente_inspeccionado.x), int(agente_inspeccionado.y)), 13, 2)

        # Panel lateral de telemetría y controles
        m = sim.obtener_metricas()
        lineas = [
            ("NEUROECOSISTEMA USTA", fuente_titulo, (255, 220, 90)),
            (f"Escenario: {sim.escenario} | Semilla: {sim.semilla}", fuente, (220, 220, 220)),
            (f"Modo P/D: {sim.modo_presas[:4]}/{sim.modo_depredadores[:4]}", fuente, (220, 220, 220)),
            (f"Generacion: {m['generacion']} (Paso {m['pasos']})", fuente, (120, 230, 180)),
            (f"Turbo [T]: {'ON (25x)' if modo_turbo else 'OFF (1x)'}", fuente, (255, 180, 100)),
            ("-" * 32, fuente, (120, 130, 145)),
            (f"Presas vivas: {m['supervivencia_presas_pct']}%", fuente, (100, 200, 255)),
            (f"Superv. media: {m['supervivencia_media_pasos']} pasos", fuente, (220, 220, 220)),
            (f"Plantas comidas: {m['alimento_presas_total']}", fuente, (120, 235, 140)),
            (f"Capturas depr.: {m['capturas_depredadores']}", fuente, (255, 110, 110)),
            (f"Energia presas: {m['energia_final_presas']}", fuente, (220, 220, 220)),
            (f"Energia depr.: {m['energia_final_deprs']}", fuente, (220, 220, 220)),
            ("-" * 32, fuente, (120, 130, 145)),
            ("CONTROLES EN VIVO:", fuente_titulo, (200, 210, 230)),
            ("[T] Acelerar entrenamiento", fuente, (190, 195, 205)),
            ("[1] Escenario A (Abundancia)", fuente, (190, 195, 205)),
            ("[2] Escenario B (Escasez)", fuente, (190, 195, 205)),
            ("[3] Escenario C (Obstaculos)", fuente, (190, 195, 205)),
            ("[R] Cambiar a Modo Reglas", fuente, (190, 195, 205)),
            ("[G] Guardar pesos (.npz)", fuente, (190, 195, 205)),
            ("[Clic] Inspeccionar agente", fuente, (190, 195, 205)),
        ]

        if agente_inspeccionado and agente_inspeccionado.vivo:
            tipo = "Presa" if isinstance(agente_inspeccionado, Presa) else "Depredador"
            sens = (agente_inspeccionado.percibir(sim.comidas, sim.depredadores)
                    if tipo == "Presa"
                    else agente_inspeccionado.percibir(sim.presas, sim.depredadores))
            lineas.extend([
                ("-" * 32, fuente, (120, 130, 145)),
                (f"INSPECCION ({tipo}):", fuente_titulo, (255, 230, 80)),
                (f"Energia: {agente_inspeccionado.energia:.1f}", fuente, (240, 240, 240)),
                (f"Alimento/Capturas: {agente_inspeccionado.alimento_consumido}", fuente, (240, 240, 240)),
                (f"S1(distObj): {sens[0]:.2f} S2(ang): {sens[1]:.2f}", fuente, (180, 220, 255)),
                (f"S3(dist2): {sens[2]:.2f} S4(ang2): {sens[3]:.2f}", fuente, (180, 220, 255)),
            ])

        y_texto = 16
        for texto, fnt, color in lineas:
            surf = fnt.render(texto, True, color)
            pantalla.blit(surf, (ancho_mapa + 14, y_texto))
            y_texto += 22

        pygame.display.flip()
        reloj.tick(60)

    pygame.quit()


if __name__ == "__main__":
    ejecutar_visualizacion()
