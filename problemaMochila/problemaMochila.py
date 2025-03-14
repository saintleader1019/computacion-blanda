import numpy as np
import matplotlib.pyplot as plt

def repair(x, b, p, C, m):
  pi = np.dot(p, x)
  while pi > C:
      pos = np.random.randint(0, m)
      x[pos] = 0
      pi = np.dot(p, x)
  foi = np.dot(b, x)
  return x, pi, foi

def upx(p1, p2, m):
  mask = np.random.randint(0, 2, m)
  m1 = mask == 1
  m0 = mask == 0
  h1 = np.zeros(m, dtype=int)
  h1[m1] = p1[m1]
  h1[m0] = p2[m0]
  return h1

# Datos del problema
b = np.array([51, 36, 83, 65, 88, 54, 26, 36, 36, 40])
p = np.array([30, 38, 54, 21, 32, 33, 68, 30, 32, 38])
m = 10
C = 220
x1 = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0])

fx1 = np.dot(b, x1)
px1 = np.dot(p, x1)

# Inicialización de la población
np.random.seed(0)  # Fijamos la semilla solo una vez
N = 20
pop = np.random.randint(0, 2, [N, m])

fos = np.dot(b, pop.T)
ps = np.dot(p, pop.T)

# Reparación de individuos infactibles
for i in range(N):
  if ps[i] > C:
    pop[i, :], ps[i], fos[i] = repair(pop[i, :], b, p, C, m)

# Incumbente inicial (mejor individuo)
incumbente = np.argmax(fos)

# Selección de padres (por torneo)
maxGen = 1000
plt.scatter(0, fos[incumbente], c='blue')

for gen in range(maxGen):
  idxcandidatos = np.random.choice(N, 4, replace=False)
  
  # Torneo para el primer padre
  if fos[idxcandidatos[0]] > fos[idxcandidatos[1]]:
    idxp1 = idxcandidatos[0]
  else:
    idxp1 = idxcandidatos[1]
  
  # Torneo para el segundo padre
  if fos[idxcandidatos[2]] > fos[idxcandidatos[3]]:
    idxp2 = idxcandidatos[2]
  else:
    idxp2 = idxcandidatos[3]
  
  padre1 = pop[idxp1]
  padre2 = pop[idxp2]
  
  # Cruce: se genera un hijo usando uniform crossover
  hijo1 = upx(padre1, padre2, m)
  print('hijo1 inicial:', hijo1, 'fo:', np.dot(b, hijo1), 'p:', np.dot(p, hijo1))
  
  # Mutación: probabilidad de mutar (por ejemplo, 0.1 o 10%)
  tm = 0.1
  if np.random.rand() <= tm:
    pos = np.random.randint(0, m)
    hijo1[pos] = 0 if hijo1[pos] == 1 else 1
  
  # Reparar la nueva solución
  hijo1, ph1, foh1 = repair(hijo1, b, p, C, m)
  foh1 = np.dot(b, hijo1)
  ph1 = np.dot(p, hijo1)
  print('hijo1 reparado:', hijo1, 'foh1:', foh1, 'ph1:', ph1)
  
  # Reemplazar el peor de la población si el hijo es mejor
  idxPeor = np.argmin(fos)
  if foh1 > fos[idxPeor]:
    fos[idxPeor] = foh1
    ps[idxPeor] = ph1
    pop[idxPeor, :] = hijo1
  
  # Actualizar el mejor individuo (incumbente) si se encuentra uno mejor
  idxincum = np.argmax(fos)
  if fos[idxincum] > fos[incumbente]:
    incumbente = idxincum
    plt.scatter(gen, fos[idxincum], c='red')

plt.grid()
plt.show()
print('Mejor solución:', pop[incumbente], 'fo:', fos[incumbente], 'p:', ps[incumbente])
