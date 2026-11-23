import math
import numpy as np
from red_neuronal import RedNeuronalMLP

def normalizar_angulo(angulo):
    """Lleva cualquier ángulo en radianes al rango [-pi, pi]."""
    while angulo > math.pi:
        angulo -= 2 * math.pi
    while angulo < -math.pi:
        angulo += 2 * math.pi
    return angulo

class AgenteBase:
    def __init__(self, x, y, ancho_mapa, alto_mapa, energia_inicial=100.0, 
                 vel_max=4.0, radio_vision=200.0, modo="neuronal", arquitectura=(5, 8, 6, 2)):
        self.x = float(x)
        self.y = float(y)
        self.ancho_mapa = ancho_mapa
        self.alto_mapa = alto_mapa
        
        self.angulo = np.random.uniform(-math.pi, math.pi)
        self.velocidad = 0.0
        self.vel_max = vel_max
        self.radio_vision = radio_vision
        
        self.energia = energia_inicial
        self.energia_max = 150.0
        self.vivo = True
        self.modo = modo
        
        self.tiempo_vivo = 0
        self.alimento_consumido = 0
        
        if self.modo == "neuronal":
            self.cerebro = RedNeuronalMLP(*arquitectura)
        else:
            self.cerebro = None

    def _buscar_mas_cercano(self, lista_objetivos, excluir_self=False):
        menor_dist = self.radio_vision
        objetivo_cercano = None
        
        for obj in lista_objetivos:
            if excluir_self and obj is self:
                continue
            ox, oy = (obj.x, obj.y) if hasattr(obj, 'x') else (obj[0], obj[1])
            dist = math.hypot(ox - self.x, oy - self.y)
            if dist < menor_dist:
                menor_dist = dist
                objetivo_cercano = (ox, oy, dist)
                
        if objetivo_cercano is None:
            return 1.0, 0.0
            
        ox, oy, dist = objetivo_cercano
        angulo_global = math.atan2(oy - self.y, ox - self.x)
        angulo_rel = normalizar_angulo(angulo_global - self.angulo)
        
        return dist / self.radio_vision, angulo_rel / math.pi

    def mover(self, giro, aceleracion, obstaculos=None):
        if not self.vivo:
            return
            
        self.angulo = normalizar_angulo(self.angulo + giro * 0.25)
        factor_vel = (aceleracion + 1.0) / 2.0
        self.velocidad = (0.2 + 0.8 * factor_vel) * self.vel_max
        
        nuevo_x = self.x + math.cos(self.angulo) * self.velocidad
        nuevo_y = self.y + math.sin(self.angulo) * self.velocidad
        
        if nuevo_x < 10 or nuevo_x > self.ancho_mapa - 10:
            self.angulo = normalizar_angulo(math.pi - self.angulo)
            nuevo_x = np.clip(nuevo_x, 10, self.ancho_mapa - 10)
        if nuevo_y < 10 or nuevo_y > self.alto_mapa - 10:
            self.angulo = normalizar_angulo(-self.angulo)
            nuevo_y = np.clip(nuevo_y, 10, self.alto_mapa - 10)
            
        if obstaculos:
            for obs in obstaculos:
                if math.hypot(nuevo_x - obs.x, nuevo_y - obs.y) < (obs.radio + 6):
                    self.angulo = normalizar_angulo(self.angulo + math.pi * 0.75)
                    nuevo_x, nuevo_y = self.x, self.y
                    break

        self.x = nuevo_x
        self.y = nuevo_y
        
        gasto = 0.12 + 0.18 * (factor_vel ** 2)
        self.energia -= gasto
        self.tiempo_vivo += 1
        
        if self.energia <= 0:
            self.energia = 0
            self.vivo = False


class Presa(AgenteBase):
    def __init__(self, x, y, ancho_mapa, alto_mapa, modo="neuronal", arquitectura=(5, 8, 6, 2)):
        super().__init__(x, y, ancho_mapa, alto_mapa, energia_inicial=100.0,
                         vel_max=4.2, radio_vision=180.0, modo=modo, arquitectura=arquitectura)

    def percibir(self, comidas, depredadores):
        dist_comida, ang_comida = self._buscar_mas_cercano(comidas)
        deprs_vivos = [d for d in depredadores if d.vivo]
        dist_dep, ang_dep = self._buscar_mas_cercano(deprs_vivos)
        energia_norm = self.energia / self.energia_max
        return [dist_comida, ang_comida, dist_dep, ang_dep, energia_norm]

    def decidir_y_actuar(self, comidas, depredadores, obstaculos=None):
        if not self.vivo:
            return
        sensores = self.percibir(comidas, depredadores)
        
        if self.modo == "neuronal":
            giro, acel = self.cerebro.forward(sensores)
        elif self.modo == "reglas":
            dist_comida, ang_comida, dist_dep, ang_dep, _ = sensores
            if dist_dep < 0.55:
                giro = -1.0 if ang_dep > 0 else 1.0
                acel = 1.0
            elif dist_comida < 1.0:
                giro = np.clip(ang_comida * 2.5, -1.0, 1.0)
                acel = 0.5
            else:
                giro = np.random.uniform(-0.2, 0.2)
                acel = 0.0
        else:
            giro = np.random.uniform(-1.0, 1.0)
            acel = np.random.uniform(-0.5, 0.5)
            
        self.mover(giro, acel, obstaculos)

    def calcular_fitness(self):
        return self.tiempo_vivo * 0.5 + self.alimento_consumido * 40.0


class Depredador(AgenteBase):
    def __init__(self, x, y, ancho_mapa, alto_mapa, modo="neuronal", arquitectura=(5, 8, 6, 2)):
        super().__init__(x, y, ancho_mapa, alto_mapa, energia_inicial=120.0,
                         vel_max=4.6, radio_vision=240.0, modo=modo, arquitectura=arquitectura)

    def percibir(self, presas, depredadores):
        presas_vivas = [p for p in presas if p.vivo]
        deprs_vivos = [d for d in depredadores if d.vivo]
        dist_presa, ang_presa = self._buscar_mas_cercano(presas_vivas)
        dist_aliado, ang_aliado = self._buscar_mas_cercano(deprs_vivos, excluir_self=True)
        energia_norm = self.energia / self.energia_max
        return [dist_presa, ang_presa, dist_aliado, ang_aliado, energia_norm]

    def decidir_y_actuar(self, presas, depredadores, obstaculos=None):
        if not self.vivo:
            return
        sensores = self.percibir(presas, depredadores)
        
        if self.modo == "neuronal":
            giro, acel = self.cerebro.forward(sensores)
        elif self.modo == "reglas":
            dist_presa, ang_presa, _, _, _ = sensores
            if dist_presa < 1.0:
                giro = np.clip(ang_presa * 3.0, -1.0, 1.0)
                acel = 0.9 if dist_presa < 0.5 else 0.4
            else:
                giro = np.random.uniform(-0.2, 0.2)
                acel = -0.3
        else:
            giro = np.random.uniform(-1.0, 1.0)
            acel = np.random.uniform(-0.5, 0.5)
            
        self.mover(giro, acel, obstaculos)

    def calcular_fitness(self):
        return self.alimento_consumido * 100.0 + self.tiempo_vivo * 0.2


class Obstaculo:
    def __init__(self, x, y, radio=25):
        self.x = float(x)
        self.y = float(y)
        self.radio = float(radio)
