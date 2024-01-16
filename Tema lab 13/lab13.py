import arviz as az
import matplotlib.pyplot as plt


#1
#centered
centered_eight_data = az.load_arviz_data("centered_eight")

nr_chains_centered = centered_eight_data.posterior.chain.size
total_samples_centered = centered_eight_data.posterior.draw.size

az.plot_posterior(centered_eight_data)
plt.suptitle("Centered")
plt.show()

print(f"Centered Model: Numar de lanturi = {nr_chains_centered}, Marime totala esantion = {total_samples_centered}")

#non-centered
non_centered_eight_data = az.load_arviz_data("non_centered_eight")

nr_chains_non_centered = non_centered_eight_data.posterior.chain.size
total_samples_non_centered = non_centered_eight_data.posterior.draw.size

az.plot_posterior(non_centered_eight_data)
plt.suptitle("Non-Centered")
plt.show()

print(f"Non-Centered Model: Numar de lanturi = {nr_chains_non_centered}, Marime totala esantion = {total_samples_non_centered}")

#2
#centered
mu_centered = centered_eight_data.posterior["mu"].values
tau_centered = centered_eight_data.posterior["tau"].values

rhat_mu_centered = az.rhat(mu_centered)
rhat_tau_centered = az.rhat(tau_centered)
autocorr_mu_centered = az.autocorr(mu_centered)
autocorr_tau_centered = az.autocorr(tau_centered)

print("Modelul Centrat:")
print(f"Rhat pentru mu: {rhat_mu_centered}")
print(f"Rhat pentru tau: {rhat_tau_centered}")
print(f"Autocorelatie pentru mu: {autocorr_mu_centered}")
print(f"Autocorelatie pentru tau: {autocorr_tau_centered}")

#non-centered
mu_non_centered = non_centered_eight_data.posterior["mu"].values
tau_non_centered = non_centered_eight_data.posterior["tau"].values

rhat_mu_non_centered = az.rhat(mu_non_centered)
rhat_tau_non_centered = az.rhat(tau_non_centered)
autocorr_mu_non_centered = az.autocorr(mu_non_centered)
autocorr_tau_non_centered = az.autocorr(tau_non_centered)

print("\nModelul Necentrat:")
print(f"Rhat pentru mu: {rhat_mu_non_centered}")
print(f"Rhat pentru tau: {rhat_tau_non_centered}")
print(f"Autocorelatie pentru mu: {autocorr_mu_non_centered}")
print(f"Autocorelatie pentru tau: {autocorr_tau_non_centered}")

#3
#centered
divergences_centered = centered_eight_data.sample_stats["diverging"].sum()
print(f"Numarul de divergente pentru modelul centrat: {divergences_centered}")
az.plot_pair(centered_eight_data, var_names=["mu", "tau"], divergences=True)
plt.suptitle("Model Centrat- divergente")
plt.show()

#non-centered
divergences_non_centered = non_centered_eight_data.sample_stats["diverging"].sum()
print(f"Numarul de divergente pentru modelul necentrat: {divergences_non_centered}")
az.plot_pair(non_centered_eight_data, var_names=["mu", "tau"], divergences=True)
plt.suptitle("Model Necentrat- divergente")
plt.show()
