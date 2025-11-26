import numpy as np

# Création d'un tableau NumPy
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

# Opérations élément par élément
addition = a + b
multiplication = a * b
division = b / a

print("Tableau a :", a)
print("Tableau b :", b)
print("Addition :", addition)
print("Multiplication :", multiplication)
print("Division :", division)

# Matrices 2D
mat1 = np.array([[1, 2], [3, 4]])
mat2 = np.array([[5, 6], [7, 8]])

# Produit matriciel
produit_mat = np.dot(mat1, mat2)

print("\nMatrice 1 :\n", mat1)
print("Matrice 2 :\n", mat2)
print("Produit matriciel :\n", produit_mat)
