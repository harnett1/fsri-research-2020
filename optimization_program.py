import numpy as np
from scipy.integrate import odeint
import pandas as pd
from lmfit import Model
import matplotlib.pyplot as plt

def fit_funct(t, beta_r, k_r, rho, alpha, epsilon, k_m, delta, beta_p, M, gamma):
    # Store extra args in an array
    args = (beta_r, k_r, rho, alpha, epsilon, k_m, delta, beta_p, M, gamma)
    
    def dX_dt(X, t, *args):
        """Returns an array of the differential equations with parameters."""
        beta_r, k_r, rho, alpha, epsilon, k_m, delta, beta_p, M, gamma = args
        r = X[0]
        m_p = X[1]
        p = X[2]
        p_m = X[3]
        # Constant concentration for I
        I = 10e-6
        drdt = (beta_r * I / (k_r + I)) - rho * r
        dmpdt = alpha + (epsilon * r / (k_m + r)) - delta * m_p
        dpdt = beta_p * m_p - M * p - gamma * p
        dpmdt = M * p - gamma * p_m
        return np.array([drdt, dmpdt, dpdt, dpmdt])
    
    # Integrate ODEs
    timepoints = np.linspace(0, 720, 62)
    X0 = np.array([0, 0, 0, 0])
    X, infodict = odeint(dX_dt, X0, timepoints, args, full_output = True)
    r, m_p, p, p_m = X.T
    return p_m


# Load data
df = pd.read_excel('combined_data.xlsx')
t=df.iloc[302:364, 0] - 100
y=df.iloc[302:364, 2]

# Initiate function and parameter ranges
pmodel = Model(fit_funct)
pmodel.set_param_hint('beta_r', min=0)
pmodel.set_param_hint('k_r', min=0)
pmodel.set_param_hint('rho', min=0)
pmodel.set_param_hint('alpha', min=0)
pmodel.set_param_hint('epsilon', min=0)
pmodel.set_param_hint('k_m', min=0)
pmodel.set_param_hint('delta', min=0, max=0.5)
pmodel.set_param_hint('beta_p', min=0)
pmodel.set_param_hint('M', min=0)
pmodel.set_param_hint('gamma', min=0, max=0.5)
pmodel.set_param_hint('k_m', min=0)

# Calculate line of best fit with initial values
result = pmodel.fit(y, t=t, beta_r=7.986, k_r=7.980, rho=1111, alpha=0.6922, epsilon=8930, k_m=7.980, delta=0.008396, beta_p=7.980, M=0.001588, gamma=0.008488)
print(result.fit_report())

# Plot data, initial fit, and best fit
plt.plot(t, y, 'bo', label='data')
plt.plot(t, result.init_fit, 'k--', label='initial fit')
plt.plot(t, result.best_fit, 'r-', label='best fit')
plt.ylabel('YFP (AU)')
plt.xlabel('Time (minutes)')
plt.xticks(range(0, 620 + 1, 100))
plt.title('Yellow Fluorescent Protein (YFP) vs Time')
plt.legend(loc="lower right", fontsize='small')
plt.show()