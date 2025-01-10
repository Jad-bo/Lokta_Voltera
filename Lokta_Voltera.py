import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
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
plt.plot(time, lapin, "b-", label="Lapins") 
plt.plot(time, renard, "r-", label="Renards")
plt.xlabel("Temps") 
plt.ylabel("Population") 
plt.legend() 
plt.title("Dynamique des populations selon le modèle de Lotka-Volterra")
plt.show()	

# Lecture et analyse des données 
df = pd.read_csv("populations_lapins_renards.csv", delimiter=",")
# Affichage des premières lignes du fichier CSV pour vérifier son contenu
print(df.head())
# Conversion des dates
df["date"] = pd.to_datetime(df["date"], yearfirst=True, errors="coerce")
# Résumé des données
print(df.info())

# Visualisation des données réelles
plt.figure(figsize=(15, 6))
plt.plot(df["date"], df["lapin"], "b--", label="Lapins (réels)")
plt.plot(df["date"], df["renard"], "r--", label="Renards (réels)")
plt.xlabel("Date")
plt.ylabel("Population")
plt.title("Données réelles des populations Proie-Prédateur")
plt.legend()
plt.show()

#Comparaison modèle vs données réelles
plt.figure(figsize=(15, 6))
plt.plot(time[: len(df)], lapin[: len(df)], "b-", label="Lapins (modèle)")
plt.plot(time[: len(df)], renard[: len(df)], "r-", label="Renards (modèle)")
plt.plot(df["date"], df["lapin"], "b--", label="Lapins (réels)")
plt.plot(df["date"], df["renard"], "r--", label="Renards (réels)")
plt.xlabel("Temps")
plt.ylabel("Population")
plt.title("Comparaison : Modèle Lotka-Volterra vs Données Réelles")
plt.legend()
plt.show()
# Definition de la Fonction Lotka-Volterra
def lotka_volterra_step(lapin, renard, alpha, beta, delta, gamma, step):
    new_lapin = (lapin * (alpha - beta * renard)) * step + lapin
    new_renard = (renard * (delta * lapin - gamma)) * step + renard
    return new_lapin, new_renard

# Fonction pour générer des prédictions avec le modèle Lotka-Volterra
def predict_lotka_volterra(time_steps, lapin_init, renard_init, alpha, beta, delta, gamma, step):
    time = [0]
    lapins = [lapin_init]
    renards = [renard_init]
    
    for _ in range(time_steps):
        new_time = time[-1] + step
        new_lapin, new_renard = lotka_volterra_step(lapins[-1], renards[-1], alpha, beta, delta, gamma, step)
        time.append(new_time)
        lapins.append(new_lapin)
        renards.append(new_renard)
    
    return np.array(time), np.array(lapins), np.array(renards)
# MSE
def mse_objective(real_lapins, real_renards, predicted_lapins, predicted_renards):
    mse_lapins = np.mean((real_lapins - predicted_lapins) ** 2)
    mse_renards = np.mean((real_renards - predicted_renards) ** 2)
    return mse_lapins + mse_renards

# Exemple d'utilisation
# Données réelles 
time_real = np.linspace(0, 20, 100)
real_lapins = np.sin(time_real) + 0.1
real_renards = np.cos(time_real) + 0.1

# Générer des prédictions
alpha, beta, delta, gamma = 1/3, 4/3, 4/3, 1/3
time_steps = len(time_real) - 1
step = (time_real[-1] - time_real[0]) / time_steps

_, predicted_lapins, predicted_renards = predict_lotka_volterra(
    time_steps, real_lapins[1], real_renards[2], alpha, beta, delta, gamma, step
)
# Calcul de la MSE
error = mse_objective(real_lapins, real_renards, predicted_lapins, predicted_renards)
print(f"Erreur quadratique moyenne (MSE) après GridSearch: {error:.4f}")    
# Paramètres initiaux pour la simulation
lapin_init = real_lapins[0]
renard_init = real_renards[0]
time_steps = len(real_lapins) - 1
step = 0.01

# Valeurs possibles pour les paramètres (grid search)
alpha_values = [1/3, 2/3, 1, 4/3]
beta_values = [1/3, 2/3, 1, 4/3]
delta_values = [1/3, 2/3, 1, 4/3]
gamma_values = [1/3, 2/3, 1, 4/3]

# Grid search pour optimiser les paramètres
results = []
for alpha, beta, delta, gamma in product(alpha_values, beta_values, delta_values, gamma_values):
    simulated_lapins, simulated_renards = predict_lotka_volterra(
        time_steps, lapin_init, renard_init, alpha, beta, delta, gamma, step
    )
    error = mse(real_lapins, real_renards, simulated_lapins[:len(real_lapins)], simulated_renards[:len(real_renards)])
    results.append({"alpha": alpha, "beta": beta, "delta": delta, "gamma": gamma, "mse": error})

# Trouver les meilleurs paramètres
best_params = min(results, key=lambda x: x["mse"])
alpha, beta, delta, gamma = best_params["alpha"], best_params["beta"], best_params["delta"], best_params["gamma"]

# Simuler avec les meilleurs paramètres
simulated_lapins, simulated_renards = predict_lotka_volterra(
    time_steps, lapin_init, renard_init, alpha, beta, delta, gamma, step
)

# Visualisation des résultats
plt.figure(figsize=(12, 6))
plt.plot(real_dates, real_lapins, "b--", label="Lapins (réels)")
plt.plot(real_dates, real_renards, "r--", label="Renards (réels)")
plt.plot(real_dates, simulated_lapins[:len(real_lapins)], "b-", label="Lapins (modèle)")
plt.plot(real_dates, simulated_renards[:len(real_renards)], "r-", label="Renards (modèle)")
plt.xlabel("Temps")
plt.ylabel("Population")
plt.title("Comparaison : Modèle Lotka-Volterra vs Données réelles")
plt.legend()
plt.show()

# Résultats finaux
print("Meilleurs paramètres trouvés :")
print(f"Alpha : {alpha}, Beta : {beta}, Delta : {delta}, Gamma : {gamma}")
print(f"Erreur quadratique moyenne (MSE) : {best_params['mse']:.4f}")

