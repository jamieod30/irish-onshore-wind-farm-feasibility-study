hFirst, WIND SPEED DISTRIBUTION:
Code:
import matplotlib.pyplot as plt

def plot_histogram(df, site):
    plt.figure()
    plt.hist(df["ws_100"], bins=50, density=True)
    plt.xlabel("Wind Speed at 100 m (m/s)")
    plt.ylabel("Probability Density")
    plt.title(f"{site.capitalize()} Wind Speed Distribution")
    plt.savefig(f"figures/wind_maps/{site}_histogram.png")
    plt.close()

Second, weibull fit
code:

from scipy.stats import weibull_min

def fit_weibull(df):
    shape, loc, scale = weibull_min.fit(df["ws_100"], floc=0)
    k = shape
    c = scale
    return k, c

third, weibull vs histogram
import numpy as np

def plot_weibull(df, site, k, c):
    plt.figure()
    
    # histogram
    plt.hist(df["ws_100"], bins=50, density=True, alpha=0.5, label="Data")

    # Weibull curve
    x = np.linspace(0, df["ws_100"].max(), 100)
    y = weibull_min.pdf(x, k, scale=c)

    plt.plot(x, y, label=f"Weibull (k={k:.2f}, c={c:.2f})")

    plt.xlabel("Wind Speed (m/s)")
    plt.ylabel("Probability Density")
    plt.title(f"{site.capitalize()} Weibull Fit")
    plt.legend()

    plt.savefig(f"figures/wind_maps/{site}_weibull.png")
    plt.close()


fourth, wind rose:
code:
from windrose import WindroseAxes

def plot_wind_rose(df, site):
    ax = WindroseAxes.from_ax()
    ax.bar(df["wd_100"], df["ws_100"], normed=True, opening=0.8)
    ax.set_title(f"{site.capitalize()} Wind Rose")
    plt.savefig(f"figures/wind_maps/{site}_wind_rose.png")
    plt.close()


fifth, wind shear:
using power law
code:

import numpy as np

def compute_shear(df):
    alpha = np.log(df["ws_100"] / df["ws_10"]) / np.log(100 / 10)
    return alpha.mean()

sixth, then we need to save those results to correct files:

results = []

for site, df in datasets.items():
    k, c = fit_weibull(df)
    shear = compute_shear(df)

    results.append({
        "site": site,
        "weibull_k": k,
        "weibull_c": c,
        "shear_exponent": shear
    })

results_df = pd.DataFrame(results)
results_df.to_csv("data/era5_wind_data/metadata/weibull_parameters.csv", index=False)




make sure to avoid:
fitting Weibull on ws_10
not forcing loc=0
forgetting density=True in histogram
using raw counts instead of probability density
ignoring shear