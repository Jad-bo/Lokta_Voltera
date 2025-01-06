import numpy as np
import matplotlib.pyplot as plt

# Définition des paramètres du modèle
time = [0]
lapin = [1]
renard = [2]
alpha, beta, delta, gama = 2/3, 4/3, 1, 1

step = 0.001

for _ in range(0,100_000):
	new_value_time = time[-1] + step
	new_value_lapin = (lapin[-1] * (alpha - beta * renard[-1])) * step +lapin[-1]
	new_value_renard = (renard[-1] * (delta * lapin[-1] - gama)) * step +renard[-1]

	time.append(new_value_time)
	lapin.append(new_value_lapin)
	renard.append(new_value_renard)
	
renard = np.array(renard)
renard *= 2000
lapin = np.array(lapin)
lapin *= 1000

plt.figure(figsize=(15, 6))
plt.plot(time, lapin, "b-")
plt.plot(time, renard, "r-")
plt.show()	