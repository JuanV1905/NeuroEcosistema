import numpy as np

class RedNeuronalMLP:
    def __init__(self, n_entradas, n_oculta1, n_oculta2, n_salidas):
        self.arquitectura = (n_entradas, n_oculta1, n_oculta2, n_salidas)
        # Inicialización de pesos (Xavier simplificado)
        self.w1 = np.random.randn(n_entradas, n_oculta1) * 0.5
        self.b1 = np.zeros((1, n_oculta1))
        
        self.w2 = np.random.randn(n_oculta1, n_oculta2) * 0.5
        self.b2 = np.zeros((1, n_oculta2))
        
        self.w3 = np.random.randn(n_oculta2, n_salidas) * 0.5
        self.b3 = np.zeros((1, n_salidas))

    def forward(self, entradas):
        x = np.array(entradas, dtype=float).reshape(1, -1)
        
        # Capa oculta 1
        z1 = np.dot(x, self.w1) + self.b1
        a1 = np.tanh(z1)
        
        # Capa oculta 2
        z2 = np.dot(a1, self.w2) + self.b2
        a2 = np.tanh(z2)
        
        # Capa de salida: [giro (-1 a 1), aceleración (-1 a 1)]
        z3 = np.dot(a2, self.w3) + self.b3
        salida = np.tanh(z3)
        
        return salida.flatten()

    def clonar_y_mutar(self, tasa_mutacion=0.15, fuerza_mutacion=0.35):
        hijo = RedNeuronalMLP(*self.arquitectura)
        for attr in ['w1', 'b1', 'w2', 'b2', 'w3', 'b3']:
            matriz = getattr(self, attr).copy()
            mascara = np.random.rand(*matriz.shape) < tasa_mutacion
            ruido = np.random.randn(*matriz.shape) * fuerza_mutacion
            matriz[mascara] += ruido[mascara]
            setattr(hijo, attr, matriz)
        return hijo

    def guardar(self, ruta):
        np.savez(
            ruta,
            arquitectura=np.array(self.arquitectura),
            w1=self.w1, b1=self.b1,
            w2=self.w2, b2=self.b2,
            w3=self.w3, b3=self.b3
        )

    @classmethod
    def cargar(cls, ruta):
        datos = np.load(ruta)
        arq = tuple(datos['arquitectura'])
        red = cls(*arq)
        for attr in ['w1', 'b1', 'w2', 'b2', 'w3', 'b3']:
            setattr(red, attr, datos[attr])
        return red
