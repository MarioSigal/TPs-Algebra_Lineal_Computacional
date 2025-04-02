import numpy as np
matriz = np.array([[1, 2, 3, 4, 5],
                   [6, 7, 8, 9, 10],
                   [11, 12, 13, 14, 15],
                   [16, 17, 18, 19, 20]])
bloque1 = matriz[0, 0]
bloque2 = matriz[0, 1:]
bloque3 = matriz[1:, 0]
bloque4 = matriz[1:, 1:]

matriz1 = np.concatenate((bloque3.reshape(-1,1), bloque4), axis=1)
matriz2 = np.concatenate(([bloque1], bloque2))
matriz3 = np.concatenate(([matriz2], matriz1))
# Imprimir los bloques

"""print("Bloque 1:")
print([bloque1])
print(bloque1.shape == ())
print("Bloque 2:")
print(bloque2)

print("Bloque 3:")
print(bloque3)
print(bloque3.reshape(-1, 1))

print("Bloque 4:")
print(bloque4)

print("Matriz 1")
print(matriz1)

print("Matriz 2")
print(matriz2)

print("Matriz 3")
print(matriz3)"""
n= 10
A=np.random.rand(n,n)
print(A.shape)
print(np.zeros(A.shape[0]-1))

def descompLU (A):
    # asumimos que la matriz es cuadrada, entoces A.shape[0] == A.shape[1]
    Ashape0= A.shape[0]

    a11 = A[0,0]
    A12 = A[0,1:]
    A21 = A[1:, 0]
    A22 = A[1:, 1:]
    u11 = a11
    U12 = A12
    L21 = A21/u11
    L22U22 = A22 - L21.reshape(-1,1)@U12.reshape(1,-1)
    
    if L22U22.shape != (2,2): #paso recursivo
        L22,U22= descompLU(L22U22)
        A = L22U22
        Ashape0= A.shape[0]  
        a11 = A[0,0]
        A12 = A[0,1:]
        A21 = A[1:, 0]
        A22 = A[1:, 1:]
        u11 = a11
        U12 = A12
        L21 = A21/u11
        O = np.zeros(Ashape0 - 1)
        Ot = O.reshape(-1,1)
        L1 = np.concatenate(([1], O))
        print(L21.reshape(-1,1), L22)
        L2 = np.concatenate((L21.reshape(-1, 1), L22), axis=1)
        L  = np.concatenate(([L1],L2))
        U1 = np.concatenate(([u11], U12))
        U2 = np.concatenate((Ot,U22), axis = 1)
        U  = np.concatenate(([U1],U2))
        return L, U
    else: #caso base
        A = L22U22
        A22 = A[1][1]
        L21 = A[1][0]/A[0][0]
        U12 = A[0][1]
        L   = np.array([[1,0],[L21,1]])
        U22 = A22 - L21*U12 #en realidad seria L22U22 pero L22=1
        U   = np.array([A[0],[0,U22]])
      
        return L,U 

A = np.array([[1, 2, 0, 1], 
              [4, 3, 2, 5], 
              [2, 1, 0, 4], 
              [3, 2, 1, 0]])
L, U = descompLU(A)
print("L:")
print(L)
print("U:")
print(U)



#chat GPT
def descompLU(A):
    n = A.shape[0]
    
    if n == 1:
        if A[0, 0] == 0:
            print("Error: A tiene un 0 en la diagonal.")
            return np.eye(1), A  # Devuelve la matriz identidad y A original
        else:
            return np.array([[1]]), A  # Matriz L es 1x1 con valor 1 y U es A original
 
 #idea para lo del cero en la diagonal 
    
    a11 = A[0, 0]
    A12 = A[0, 1:]
    A21 = A[1:, 0]
    A22 = A[1:, 1:]
    
  #idea para no tener que concatenar 
    L = np.eye(n)  # Matriz identidad de nxn
    U = np.zeros((n, n))  # Matriz de ceros de nxn
    
    # Calculamos u11 y U12
    if a11 == 0:
        print("Error: A tiene un 0 en la diagonal.")
        return L, A  # Devuelve la matriz identidad y A original
    
    u11 = a11
    U12 = A12
    
    # Calculamos L21
    L21 = A21 / u11
    
    # Actualizamos las matrices L y U
    L[1:, 0] = L21
    U[0, 1:] = U12
    
    # Calculamos L22 y U22 recursivamente
    A22_minus_L21_U12 = A22 - np.outer(L21, U12)
    L22, U22 = descompLU(A22_minus_L21_U12)
    
    # Actualizamos las matrices L y U con los bloques calculados recursivamente
    L[1:, 1:] = L22
    U[1:, 1:] = U22
    
    return L, U

# Ejemplo de uso:
A = np.array([[1, 2, 0, 1], [4, 3, 2, 5], [2, 1, 0, 4], [3, 2, 1, 0]])
L, U = descompLU(A)
print("L:")
print(L)
print("U:")
print(U)
    
#A = np.array([[1,2,0,1],[4,3,2,5],[2,1,0,4],[3,2,1,0]])
#print(descompLU(A))
#L   = np.array([[1,0],[2,1]])
#print(L)

A21 = A[1:, 0]
a11 = A[0,0]
A12 = A[0,1:]
A21 = A[1:, 0]
A22 = A[1:, 1:]
u11 = a11
U12 = A12
L21 = A21/u11
L22U22 = A22 - L21.reshape(-1,1)@U12.reshape(1,-1)
print(A21)
print(L21)
print(U12)
print(L21.reshape(-1,1))
print(U12.reshape(1,-1))
print(A22)
print(L21@U12)
print(L21.reshape(-1,1)@U12.reshape(1,-1))
print(L22U22)


#print(descompLU(A))
A = np.array([[1,2],[3,4]])
a11 = A[0,0]
A12 = A[0,1:]
A21 = A[1:, 0]
A22 = A[1:, 1:]
print( a11, A12, A21, A22)
def reordenar_matrices(matriz):
    # Dividir la matriz en bloques
    bloque1 = matriz[:2, :2]
    bloque2 = matriz[:2, 2:]
    bloque3 = matriz[2:, :2]
    bloque4 = matriz[2:, 2:]

    # Reordenar los bloques según el orden deseado
    matriz1 = np.concatenate((bloque1, bloque3), axis=0)
    matriz2 = np.concatenate((bloque2, bloque4), axis=1)

    return matriz1, matriz2, bloque1, bloque2, bloque3, bloque4

# Ejemplo de uso:
matriz_original = np.array([[1, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12],
                            [13, 14, 15, 16]])

matriz_resultante1, matriz_resultante2, bloque1, bloque2, bloque3, bloque4 = reordenar_matrices(matriz_original)

print("Matriz Resultante 1:")
print(matriz_resultante1)

print("Matriz Resultante 2:")
print(matriz_resultante2)

print("Bloque 1:")
print(bloque1)

print("Bloque 2:")
print(bloque2)

print("Bloque 3:")
print(bloque3)

print("Bloque 4:")
print(bloque4)

# Supongamos que tienes una matriz
matriz = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Supongamos que quieres tomar el elemento en la coordenada (i, j)
i = 1
j = 1

# Supongamos que tienes una fila de destino a la que deseas agregar el elemento
fila_destino = np.array([10, 11, 12])

# Tomar el elemento en (i, j)
elemento = matriz[i, j]

# Agregar una nueva dimensión a la fila de destino

# Concatenar el elemento a la fila de destino
fila_destino = np.concatenate((fila_destino, [[elemento]]), axis=0)

print(fila_destino)

# ej 2 https://www.youtube.com/watch?v=KaZ9T7czN6U
  #np.concatenate(np.array([[1,0]]),np.concatenate([L21],np.array([[1]]))) 
    #np.concatenate(np.concatenate([u11],U12),np.concatenate(np.array([[0]]),U22)) 
