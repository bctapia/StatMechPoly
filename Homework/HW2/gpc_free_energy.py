import numpy as np
import matplotlib.pyplot as plt

def summation(k, Rg, L):
    if k % 2 == 0:
        RuntimeWarning(f"k was supplied as even (={k}). Skipping this k...")
        return 0
    else:
        return (1 / k**2) * np.exp(-(k**2 * np.pi**2 * Rg**2) / L**2)


def run(Rg_vals, L):
    x_vals = []
    betaF_vals = []
    scaled_vals = []

    for Rg in Rg_vals:
        sum_val = 0
        for k in range(1, 20, 2):   # 10 odd terms
            sum_val += summation(k, Rg, L)

        S = (8 / np.pi**2) * sum_val
        betaF = -np.log(S)

        x = Rg / L
        x_vals.append(x)
        betaF_vals.append(betaF)
        scaled_vals.append(betaF / x)

        #print(f"Rg/L = {x:.4f}, betaF = {betaF:.6f}")

    return np.array(x_vals), np.array(betaF_vals), np.array(scaled_vals)


if __name__ == "__main__":

    L = 1
    Rg_vals = np.logspace(-4, 1.2, 100)   # keeps Rg/L < 1

    x_vals, betaF_vals, scaled_vals = run(Rg_vals, L)

    logx = np.log(x_vals)
    logy = np.log(betaF_vals)

    #slope, intercept = np.polyfit(logx, logy, 1)

    # instantaneous scaling exponent:
    # d log(betaF) / d log(x)
    inst_slope = np.gradient(logy, logx)

    # Main log-log plot
    plt.figure()
    plt.loglog(x_vals, betaF_vals)
    plt.xlabel(r"$R_g/L$")
    plt.ylabel(r"$\beta A_{\mathrm{conf}}$")

    # Instantaneous exponent plot
    plt.figure()
    plt.semilogx(x_vals, inst_slope)
    plt.xlabel(r"$R_g/L$")
    plt.ylabel(r"$d\log(\beta A_{\mathrm{conf}})/d\log(R_g/L)$")

    plt.axhline(0, linestyle="--", color="black")
    plt.axhline(1, linestyle="--", color="black")
    plt.axhline(2, linestyle="--", color="black")

    # Optional: compare to fitted constant slope
    #plt.axhline(slope, linestyle='--')

    plt.show()