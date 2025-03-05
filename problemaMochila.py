import numpy as np
import matplotlib.pyplot as plt

def repair(x, b,p,C,m):
  pi=np.dot(p,x)
  while pi>C:
    pos=np.random.randint(0,m)
    x[pos]=0
    pi=np.dot(p,x)
  foi=np.dot(b,x)
  return x, pi, foi

def upx(p1,p2,m):
  mask = np.random.randint(0,2,m)
  m1 = mask == 1
  m0 = mask == 0
  h1 = np.zeros(m,dtype=int)
  h1[m1]=p1[m1]
  h1[m0]=p2[m0]
  return h1

b=np.array([51, 36, 83, 65, 88, 54, 26, 36,36,40])
p=np.array([30, 38, 54, 21, 32, 33, 68, 30, 32 ,38])
m=10
C=220
x1=np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0,])

fx1=np.dot(b,x1)
px1=np.dot(p,x1)
#ejercicio 3
np.random.seed(0)
N=20
pop=np.random.randint(0,2, [N,m])

fos= np.dot(b,np.transpose(pop))
ps= np.dot(p,np.transpose(pop))

# Reparación de individuos infactibles
for i  in range(N):
  if ps[i]>C:
    pop[i,: ], ps[i,], fos[i,] =repair(pop[i,:], b,p,C,m)

#incertidumbre inicial
incertidumbre = np.argmax(fos)

#selección de padres
total = np.sum(fos)
fas = fos/total

# np.random.seed(0)
maxGen = 1000
plt.scatter(0,fos[incertidumbre],c='blue')
for gen in range(maxGen):
  idxcandidatos = np.random.choice(N,4,replace=False)

  #torneo de los 4 candidatos
  if fos[idxcandidatos[0]]>fos[idxcandidatos[1]]:
    idxp1=idxcandidatos[0]
  else:
    idxp1=idxcandidatos[1]

  if fos[idxcandidatos[2]]>fos[idxcandidatos[3]]:
    idxp2=idxcandidatos[2]
  else:
    idxp2=idxcandidatos[3]

    #funcion upx, para cruzar dos padres y obtener dos hijos que deberian ser mejor que los padre
    np.random.seed(0)
    padre1 = pop[idxp1]
    padre2 = pop[idxp2]


    # padre1=np.array([1,1,0,0,1,1],dtype=int)  
    # padre2=np.array([1,0,1,0,1,0],dtype=int)
    hijo1 = upx(padre1,padre2,m)
    print('hijo1',hijo1,'fo',np.dot(b,hijo1),'p',np.dot(p,hijo1))

    #mutacion 
    tm = 1

    if np.random.rand() <= tm:
      pos = np.random.randint(0,m)
      if hijo1[pos]==1:
        hijo1[pos]=0
      else:
        hijo1[pos]=1



    #reparar la nueva solucion
    hijo1, ph1, foh1 = repair(hijo1,b,p,C,m)


    foh1 = np.dot(b,hijo1)
    ph1 = np.dot(p,hijo1)
    print('hijo1',hijo1, 'foh1', foh1, 'ph1', ph1)

    #si tengo una mutaccion del 10%, el individuo se altera un gen aleatoreamente,calcular cuantas cambios hay, toca meter un for o while que refleje ese porcentaje, ´parte de la primera evaluacion, este el core del algoritmo, saber si ese hijo se puede integrar a la poblacion, vamos a comporar el nuevo con el peor de la poblacion, si es mejor, se reemplaza, si no, se descarta
    #acualizar la poblacion, se reemplaza el peor de la poblacion por el hijo
    idxPeor = np.argmin(fos)
    if foh1 > fos[idxPeor]:
      fos[idxPeor] = foh1
      ps[idxPeor] = ph1
      pop[idxPeor,]=hijo1


    #actualizar a sebas(incumbente)
    idxincum = np.argmax(fos)
    if fos[idxincum]>fos[incertidumbre]:
      incertidumbre = idxincum
      plt.scatter(gen,fos[idxincum],c='red')

plt.grid()      
plt.show()
print('mejor solucion', pop[incertidumbre], 'fo', fos[idxincum], 'p', ps[idxincum])


#adaptacion , funcion matematica
#seleecion no por torneo sino por ruleta
#primer parcial, dentro de 15 dias