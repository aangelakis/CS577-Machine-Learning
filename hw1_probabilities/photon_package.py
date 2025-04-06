import matplotlib.pyplot as plt
import numpy as np

# Part 2.a
FP = 0.1 # False Positive Rate
TP = 0.85 # True Positive Rate
prior = 1e-7 # Prior probability of a photon package emmited by a star passing through the Earth's atmosphere and reaching the detector
evidence = TP*prior + FP*(1-prior) # Evidence
posterior_prob = (TP*prior) / evidence # Bayes' Theorem
print('Posterior Probability:', posterior_prob)


# Part 2.b
photons = 100 # Number of photons in the package
photon_energies = [10, 20, 30, 40] # Probable energy of a photon (equal probability)
N = 2000 # Number of experiments

total_energies = []

for _ in range(N):
    energy_samples = np.random.choice(photon_energies, photons) # Sample the energies of the photons from the package uniformly
    total_energies.append(np.sum(energy_samples)) # Calculate the total energy of the package

total_energies = np.array(total_energies)
plt.figure()
plt.hist(total_energies, bins=30)
plt.xlabel('Total Energy (eV)')
plt.ylabel('Occurences')
plt.show()


# Part 2.c
mu = 1e-7
sigma = 9e-8
N = 10000
posterior_probs = []

while len(posterior_probs) < N:
    prior_prob = np.random.normal(mu, sigma) # Sample the prior distribution
    if prior_prob < 0:
        continue
    evidence = TP*prior_prob + FP*(1-prior_prob) # Calculate the evidence
    posterior_prob = (TP*prior_prob) / evidence # Calculate the posterior probability
    posterior_probs.append(posterior_prob)

posterior_probs = np.array(posterior_probs)
plt.figure()
plt.hist(posterior_probs, bins=30)
plt.xlabel('Posterior Probability')
plt.ylabel('Occurences')
plt.show()
