import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import approx_fprime
from numpy.linalg import solve
from scipy.linalg import eigh
import pandas as pd
import os
import time  


# Funzione di Nelson-Siegel per la curva dei rendimenti
def nelson_siegel(t, beta0, beta1, beta2, tau):
    term1 = beta0
    term2 = beta1 * (1 - np.exp(-t / tau)) / (t / tau)
    term3 = beta2 * ((1 - np.exp(-t / tau)) / (t / tau) - np.exp(-t / tau))
    return term1 + term2 + term3


# Funzione di errore (loss function) per minimizzare
def objective_function(params, t, yields):
    beta0, beta1, beta2, tau = params
    model_yields = nelson_siegel(t, beta0, beta1, beta2, tau)
    return np.sum((yields - model_yields)**2)


# Funzione di Nelson-Siegel-Svensson per la curva dei rendimenti
def nelson_siegel_svensson_nss(t, beta0, beta1, beta2, beta3, tau1, tau2):
    term1 = beta0
    term2 = beta1 * (1 - np.exp(-t / tau1)) / (t / tau1)
    term3 = beta2 * ((1 - np.exp(-t / tau1)) / (t / tau1) - np.exp(-t / tau1))
    term4 = beta3 * ((1 - np.exp(-t / tau2)) / (t / tau2) - np.exp(-t / tau2))
    return term1 + term2 + term3 + term4

# Funzione di errore (loss function) per minimizzare
def objective_function_nss(params, t, yields):
    beta0, beta1, beta2, beta3, tau1, tau2 = params
    model_yields = nelson_siegel_svensson_nss(t, beta0, beta1, beta2, beta3, tau1, tau2)
    return np.sum((yields - model_yields)**2)


####################################################################
# Metodo di discesa gradiente (dal laboratorio)
def gradient_descent(f, x_0, alpha_0=1, apx_LS=True, diminishing_alpha=False, print_output=True, N=100, eps=1e-4): 
    x_values = [x_0]
    f_values = [f(x_0)]
    alpha = alpha_0

    for i in range(N):
        d = -approx_fprime(x_values[-1], f, epsilon=1e-5)

        if apx_LS:
            alpha = apx_line_search(f, x_values[-1], d, alpha_0=alpha_0)
        elif diminishing_alpha:
            alpha = alpha * 0.9
        else:
            alpha = alpha_0

        # Update x
        x_new = x_values[-1] + alpha * d
        x_values.append(x_new)
        f_values.append(f(x_new))

        # Stopping criterion
        if np.linalg.norm(d) < eps:
            print(f"Gradient Descent: convergenza raggiunta dopo {i+1} iterazioni.")
            break

    if print_output:
        print('Gradient descent method performed ' + str(i+1) + ' iterations')
    return x_values[-1], f_values[-1]


# Line search utilizzato nella discesa gradiente
def apx_line_search(f, x, d, c=0.1, t=0.9, alpha_0=1):
    alpha = alpha_0
    f_x = f(x)

    def phi(a):
        return f(x + a * d)

    phi_prime = approx_fprime(0, phi, epsilon=1e-5)

    while phi(alpha) > f_x + c * alpha * phi_prime:
        alpha *= t

    return alpha

##########################################################################
# Metodo di Newton-Raphson  (dal laboratorio)
def newton_method(f, x_0, N=100, damping_factor=0.6, eps=1e-6,alpha_0=1, apx_LS=True, diminishing_alpha=False):  
    x_values = [x_0]
    f_values = [f(x_0)]
    alpha = alpha_0

    for i in range(N):
        gradient = approx_fprime(x_values[-1], f, epsilon=1e-5)
        hessian = approx_hessian(x_values[-1], f)

        if not is_positive_definite(hessian):
            hessian = hessian + damping_factor * np.eye(len(x_0))

        d = solve(hessian, -gradient)
        
        if apx_LS:
            alpha = apx_line_search(f, x_values[-1], d, alpha_0=alpha_0)
        elif diminishing_alpha:
            alpha = alpha * 0.1
        else:
            alpha = alpha_0

       
        x_values.append(x_values[-1] + alpha *d)
        f_values.append(f(x_values[-1]))

        if np.linalg.norm(d) < eps:
            print(f"Newton-Raphson: convergenza raggiunta dopo {i+1} iterazioni.")
            break
        
    
    print('Newton\'s method performed ' + str(i+1) + ' iterations\n')
    return x_values[-1], f_values[-1]


def is_positive_definite(matrix, tol=1e-4):
    # Compute the eigenvalues
    eigenvalues = eigh(matrix, eigvals_only=True)
    return np.all(eigenvalues > tol)


def approx_hessian(x, f):
    n = len(x)
    hessian = np.zeros((n, n))

    for i in range(n):
        def grad_i(y):
            return approx_fprime(y, f, epsilon=1e-5)[i]
        hess_i = approx_fprime(x, grad_i, epsilon=1e-5)
        
        for j in range(n):
            if i <= j:
                hessian[i, j] = hess_i[j]
                hessian[j, i] = hessian[i, j]
    return hessian

##############################################################
##############################################################

# carico dati SHORT - TERM
os.chdir(r'C:\Users\User\OneDrive\Desktop')
d_st = pd.read_csv('Italy 1-Year Bond Yield Historical Data.csv', header=0) #Italy
#d_st = pd.read_csv('Japan 1-Year Bond Yield Historical Data.csv', header=0) #Japan
#d_st = pd.read_csv('United States 1-Year Bond Yield Historical Data.csv', header=0) #USA

observed_yields_st = d_st['Price'].values[::-1]
observed_yields_st = observed_yields_st/ 100 #trasformo 2.5% --> 0.025

t_st = np.arange(1, len(d_st) + 1)
print('\n SHORT - TERM DATASET')
print('Frequency: Weekly')
print('Length of historical series:', len(d_st), 'data, which is more or less', round(len(d_st)/52), 'years')
print(d_st.head())


# Parametri iniziali: [beta0, beta1, beta2, tau]
#x_0_st = [0.10, -0.20, 0.10, 0.1]  #starting point 'non-realistic'
x_0_st= [0.0242, -0.00141, 0.05, 1.0]  # Italy
#x_0_st= [0.00016, 0.00001, 0.00002, 1.0]  # Japan
#x_0_st= [0.0416, -0.0009, 0.0003, 1.0]  # USA

def f_to_minimize(params):
    return objective_function(params, t_st, observed_yields_st)

### Gradient Descent method
start_time_gd = time.time()  # Inizio cronometro
gd_params_st, gd_loss_st = gradient_descent(f_to_minimize, x_0_st)
end_time_gd = time.time()  # Fine cronometro
gd_time_st = end_time_gd - start_time_gd  # Tempo di esecuzione in secondi

# Newton-Raphson method
start_time_nr = time.time()  # Inizio cronometro
nr_params_st, nr_loss_st = newton_method(f_to_minimize, x_0_st)
end_time_nr = time.time()  # Fine cronometro
nr_time_st = end_time_nr - start_time_nr  # Tempo di esecuzione in secondi


# Gradient Descent
model_yields_gd_st = nelson_siegel(t_st, *gd_params_st)
# Newton-Raphson
model_yields_nr_st = nelson_siegel(t_st, *nr_params_st)

# Residui
residuals_nr_st = observed_yields_st - model_yields_nr_st
residuals_gd_st = observed_yields_st - model_yields_gd_st

# Parametri ottimali trovati
print('NELSON-SIEGEL SHORT - TERM')
print("Gradient Descent:")
print(f"  beta0: {gd_params_st[0]:.4f}, beta1: {gd_params_st[1]:.4f}, beta2: {gd_params_st[2]:.4f}, tau: {gd_params_st[3]:.4f}")
print("Newton-Raphson:")
print(f"  beta0: {nr_params_st[0]:.4f}, beta1: {nr_params_st[1]:.4f}, beta2: {nr_params_st[2]:.4f}, tau: {nr_params_st[3]:.4f}")

# Confronto
results_table = pd.DataFrame({
    'Method': ['Gradient Descent', 'Newton-Raphson'],
    'Loss': [gd_loss_st, nr_loss_st],
    'Running time(s)': [gd_time_st, nr_time_st]
})
print(results_table)


# Definire un range di maturità per la yield curve 
maturity_st = np.linspace(0.1, 15, 100)  # Maturità da 0.1 a 15 anni (esempio)

# Calcolo delle yield curve per maturità con i parametri ottimizzati
yield_curve_gd_st = nelson_siegel(maturity_st, *gd_params_st)
yield_curve_nr_st = nelson_siegel(maturity_st, *nr_params_st)


#####################################################################
plt.figure(figsize=(8, 6))

# Grafico 1: Curva dei rendimenti - Gradient Descent vs Newton-Raphson
plt.subplot(2, 1, 1)
plt.plot(t_st, observed_yields_st, label='Observed Yields', linestyle='', color='black', marker='.')
plt.plot(t_st, model_yields_nr_st, label='Modeled Yields (Newton-Raphson)', color='green')
plt.plot(t_st, model_yields_gd_st, label='Modeled Yields (Gradient Descent)', color='purple')
plt.title('ST: Yield Curve - Gradient Descent vs Newton-Raphson')
plt.xlabel('Time (t)')
plt.ylabel('Yields')
plt.legend()
plt.grid()


# Residuals Plot - Gradient Descent vs Newton-Raphson
plt.subplot(2, 1, 2)
plt.plot(t_st, residuals_nr_st, label='Residuals (Newton-Raphson)', color='green')
plt.plot(t_st, residuals_gd_st, label='Residuals (Gradient Descent)', color='purple')
plt.axhline(0, color='black')
plt.title('ST: Residuals - Gradient Descent vs Newton-Raphson')
plt.xlabel('Time (t)')
plt.ylabel('Residuals')
plt.legend()
plt.grid()


plt.tight_layout()  
plt.show()
"""
# Yield Curve as a Function of Maturity
plt.figure(figsize=(8, 6))
plt.plot(maturity_st, yield_curve_gd_st, label='Yield Curve (Gradient Descent)', color='purple')
plt.plot(maturity_st, yield_curve_nr_st, label='Yield Curve (Newton-Raphson)', color='green')
plt.title('ST: Yield Curve as a Function of Maturity')
plt.xlabel('Maturity (Years)')
plt.ylabel('Yield (%)')
plt.legend()
plt.grid()
plt.show()
"""

#############################################################################
# Parametri iniziali per NS-Svensson: [beta0, beta1, beta2, beta3, tau1, tau2]
#x_0_nss_st = [0.10, -0.20, 0.10, 0.20, 0.1, 50.0]  #starting point 'non-realistic'
x_0_nss_st = [0.0242, -0.00141, 0.05, 0.025, 1.0, 2.0]  #Italy
#x_0_nss_st = [0.00016, 0.00001, 0.00002, 0.00001, 1.0, 2.0]  #Japan
#x_0_nss_st = [0.0416, -0.0009, 0.0003, 0.00015, 1.0, 2.0]  #USA

def f_to_minimize_nss(params):
    return objective_function_nss(params, t_st, observed_yields_st)

### Gradient Descent method per NS-Svensson
start_time_gd_nss = time.time()  # Inizio cronometro
gd_params_nss_st, gd_loss_nss_st = gradient_descent(f_to_minimize_nss, x_0_nss_st)
end_time_gd_nss = time.time()  # Fine cronometro
gd_time_nss_st = end_time_gd_nss - start_time_gd_nss  # Tempo di esecuzione in secondi

# Newton-Raphson method per NS-Svensson
start_time_nr_nss = time.time()  # Inizio cronometro
nr_params_nss_st, nr_loss_nss_st = newton_method(f_to_minimize_nss, x_0_nss_st)
end_time_nr_nss = time.time()  # Fine cronometro
nr_time_nss_st = end_time_nr_nss - start_time_nr_nss  # Tempo di esecuzione in secondi

# Gradient Descent
model_yields_gd_nss_st = nelson_siegel_svensson_nss(t_st, *gd_params_nss_st)
# Newton-Raphson
model_yields_nr_nss_st = nelson_siegel_svensson_nss(t_st, *nr_params_nss_st)

# Residui
residuals_nr_nss_st = observed_yields_st - model_yields_nr_nss_st
residuals_gd_nss_st = observed_yields_st - model_yields_gd_nss_st

# Parametri ottimali trovati
print('NELSON-SIEGEL-SVESSON SHORT - TERM')
print("Gradient Descent:")
print(f"  beta0: {gd_params_nss_st[0]:.4f}, beta1: {gd_params_nss_st[1]:.4f}, beta2: {gd_params_nss_st[2]:.4f}, beta3: {gd_params_nss_st[3]:.4f}, tau1: {gd_params_nss_st[4]:.4f}, tau2: {gd_params_nss_st[5]:.4f}")
print("Newton-Raphson:")
print(f"  beta0: {nr_params_nss_st[0]:.4f}, beta1: {nr_params_nss_st[1]:.4f}, beta2: {nr_params_nss_st[2]:.4f}, beta3: {nr_params_nss_st[3]:.4f}, tau1: {nr_params_nss_st[4]:.4f}, tau2: {nr_params_nss_st[5]:.4f}")

# Confronto
results_table_nss = pd.DataFrame({
    'Method': ['Gradient Descent', 'Newton-Raphson'],
    'Loss': [gd_loss_nss_st, nr_loss_nss_st],
    'Running time (s)': [gd_time_nss_st, nr_time_nss_st]
})
print(results_table_nss)

# Definire un range di maturità per la yield curve 
maturity_nss_st = np.linspace(0.1, 15, 100)  # Maturità da 0.1 a 15 anni (esempio)

# Calcolo delle yield curve per maturità con i parametri ottimizzati
yield_curve_gd_nss_st = nelson_siegel_svensson_nss(maturity_nss_st, *gd_params_nss_st)
yield_curve_nr_nss_st = nelson_siegel_svensson_nss(maturity_nss_st, *nr_params_nss_st)


#####################################################################
plt.figure(figsize=(8, 6))

# Newton-Svensson Model Yield Curve
plt.subplot(2, 1, 1)
plt.plot(t_st, observed_yields_st, label='Observed Yields', linestyle='', color='black', marker='.')
plt.plot(t_st, model_yields_nr_nss_st, label='Modeled Yields (Newton-Raphson)', color='green')
plt.plot(t_st, model_yields_gd_nss_st, label='Modeled Yields (Gradient Descent)', color='purple')
plt.title('ST: Yield Curve - Gradient Descent vs Newton-Raphson (NS-Svensson)')
plt.xlabel('Time (t)')
plt.ylabel('Yields')
plt.legend()
plt.grid()
plt.subplots_adjust(top=0.9, hspace=0.4)

# Residuals - Newton-Svensson Model
plt.subplot(2, 1, 2)
plt.plot(t_st, residuals_nr_nss_st, label='Residuals (Newton-Raphson)', color='green')
plt.plot(t_st, residuals_gd_nss_st, label='Residuals (Gradient Descent)', color='purple')
plt.axhline(0, color='black')
plt.title('ST: Residuals - Gradient Descent vs Newton-Raphson (NS-Svensson)')
plt.xlabel('Time (t)')
plt.ylabel('Residuals')
plt.legend()
plt.grid()
plt.show()

plt.tight_layout()  
plt.show()
"""
# Yield Curve as a Function of Maturity (NS-Svensson)
plt.figure(figsize=(8, 6))
plt.plot(maturity_nss_st, yield_curve_gd_nss_st, label='Yield Curve (Gradient Descent)', color='purple')
plt.plot(maturity_nss_st, yield_curve_nr_nss_st, label='Yield Curve (Newton-Raphson)', color='green')
plt.title('ST: Yield Curve as a Function of Maturity (NS-Svensson)')
plt.xlabel('Maturity (Years)')
plt.ylabel('Yield (%)')
plt.legend()
plt.grid()
plt.show()
"""

##############################################################
##############################################################

# carico dati MEDIUM - TERM
d_mt = pd.read_csv('Italy 5-Year Bond Yield Historical Data.csv', header=0) #Italy
#d_mt = pd.read_csv('Japan 5-Year Bond Yield Historical Data.csv', header=0) #Japan
#d_mt = pd.read_csv('United States 5-Year Bond Yield Historical Data.csv', header=0) #USA


observed_yields_mt = d_mt['Price'].values
observed_yields_mt = observed_yields_mt/ 100 #trasformo 2.5% --> 0.025
t_mt = np.arange(1, len(d_mt) + 1)
print('\n MEDIUM - TERM DATASET')
print('Frequency: Weekly')
print('Length of historical series:', len(d_mt), 'data, which is more or less', round(len(d_mt)/52), 'years')
print(d_mt.head())

# Parametri iniziali: [beta0, beta1, beta2, tau]
#x_0_mt = [0.10, -0.20, 0.10, 0.1]   #starting point 'non-realistic'
x_0_mt= [0.0240, -0.00133, 0.05, 1.5]  # Italy
#x_0_mt= [0.00021, 0.00001, 0.00002, 1.5]  # Japan
#x_0_mt= [0.0275, 0.00003, 0.0004, 1.5]  # USA

def f_to_minimize(params):
    return objective_function(params, t_mt, observed_yields_mt)

### Gradient Descent method
start_time_gd = time.time()  # Inizio cronometro
gd_params_mt, gd_loss_mt = gradient_descent(f_to_minimize, x_0_mt)
end_time_gd = time.time()  # Fine cronometro
gd_time_mt = end_time_gd - start_time_gd  # Tempo di esecuzione in secondi

# Newton-Raphson method
start_time_nr = time.time()  # Inizio cronometro
nr_params_mt, nr_loss_mt = newton_method(f_to_minimize, x_0_mt)
end_time_nr = time.time()  # Fine cronometro
nr_time_mt = end_time_nr - start_time_nr  # Tempo di esecuzione in secondi


# Gradient Descent
model_yields_gd_mt = nelson_siegel(t_mt, *gd_params_mt)
# Newton-Raphson
model_yields_nr_mt = nelson_siegel(t_mt, *nr_params_mt)

# Residui
residuals_nr_mt = observed_yields_mt - model_yields_nr_mt
residuals_gd_mt = observed_yields_mt - model_yields_gd_mt

# Parametri ottimali trovati
print('NELSON-SIEGEL MEDIUM - TERM')
print("Gradient Descent:")
print(f"  beta0: {gd_params_mt[0]:.4f}, beta1: {gd_params_mt[1]:.4f}, beta2: {gd_params_mt[2]:.4f}, tau: {gd_params_mt[3]:.4f}")
print("Newton-Raphson:")
print(f"  beta0: {nr_params_mt[0]:.4f}, beta1: {nr_params_mt[1]:.4f}, beta2: {nr_params_mt[2]:.4f}, tau: {nr_params_mt[3]:.4f}")

# Confronto
results_table = pd.DataFrame({
    'Method': ['Gradient Descent', 'Newton-Raphson'],
    'Loss': [gd_loss_mt, nr_loss_mt],
    'Running time (s)': [gd_time_mt, nr_time_mt]
})
print(results_table)


# Definire un range di maturità per la yield curve 
maturity_mt = np.linspace(0.1, 15, 100)  # Maturità da 0.1 a 15 anni (esempio)

# Calcolo delle yield curve per maturità con i parametri ottimizzati
yield_curve_gd_mt = nelson_siegel(maturity_mt, *gd_params_mt)
yield_curve_nr_mt = nelson_siegel(maturity_mt, *nr_params_mt)

#####################################################################
plt.figure(figsize=(8, 6))
plt.subplot(2, 1, 1)
plt.plot(t_mt, observed_yields_mt, label='Observed Yields', linestyle='', color='black', marker='.')
plt.plot(t_mt, model_yields_nr_mt, label='Modeled Yields (Newton-Raphson)', color='green')
plt.plot(t_mt, model_yields_gd_mt, label='Modeled Yields (Gradient Descent)', color='purple')
plt.title('MT: Yield Curve - Gradient Descent vs Newton-Raphson')
plt.xlabel('Time (t)')
plt.ylabel('Yields')
plt.legend()
plt.grid()


# Residuals Plot - Gradient Descent vs Newton-Raphson
plt.subplot(2, 1, 2)
plt.plot(t_mt, residuals_nr_mt, label='Residuals (Newton-Raphson)', color='green')
plt.plot(t_mt, residuals_gd_mt, label='Residuals (Gradient Descent)', color='purple')
plt.axhline(0, color='black')
plt.title('MT: Residuals - Gradient Descent vs Newton-Raphson')
plt.xlabel('Time (t)')
plt.ylabel('Residuals')
plt.legend()
plt.grid()

plt.tight_layout()  
plt.show()

"""
# Yield Curve as a Function of Maturity
plt.figure(figsize=(8, 6))
plt.plot(maturity_mt, yield_curve_gd_mt, label='Yield Curve (Gradient Descent)', color='purple')
plt.plot(maturity_mt, yield_curve_nr_mt, label='Yield Curve (Newton-Raphson)', color='green')
plt.title('MT: Yield Curve as a Function of Maturity')
plt.xlabel('Maturity (Years)')
plt.ylabel('Yield (%)')
plt.legend()
plt.grid()
plt.show()
"""
#####################################################################

# Parametri iniziali per NS-Svensson: [beta0, beta1, beta2, beta3, tau1, tau2]
#x_0_nss_mt = [0.10, -0.20, 0.10, 0.20, 0.1, 50.0]  #starting point 'non-realistic'
x_0_nss_mt = [0.0240, -0.00133, 0.05, 0.025, 1.5, 3.0]  # Italy
#x_0_nss_mt = [0.00021, 0.00001, 0.00002, 0.00001, 1.5, 3.0]  # Japan
#x_0_nss_mt = [0.0275, 0.00003, 0.0004, 0.0002, 1.5, 3.0]  # USA


def f_to_minimize_nss(params):
    return objective_function_nss(params, t_mt, observed_yields_mt)

### Gradient Descent method per NS-Svensson
start_time_gd_nss = time.time()  # Inizio cronometro
gd_params_nss_mt, gd_loss_nss_mt = gradient_descent(f_to_minimize_nss, x_0_nss_mt)
end_time_gd_nss = time.time()  # Fine cronometro
gd_time_nss_mt = end_time_gd_nss - start_time_gd_nss  # Tempo di esecuzione in secondi

# Newton-Raphson method per NS-Svensson
start_time_nr_nss = time.time()  # Inizio cronometro
nr_params_nss_mt, nr_loss_nss_mt = newton_method(f_to_minimize_nss, x_0_nss_mt)
end_time_nr_nss = time.time()  # Fine cronometro
nr_time_nss_mt = end_time_nr_nss - start_time_nr_nss  # Tempo di esecuzione in secondi

# Gradient Descent
model_yields_gd_nss_mt = nelson_siegel_svensson_nss(t_mt, *gd_params_nss_mt)
# Newton-Raphson
model_yields_nr_nss_mt = nelson_siegel_svensson_nss(t_mt, *nr_params_nss_mt)

# Residui
residuals_nr_nss_mt = observed_yields_mt - model_yields_nr_nss_mt
residuals_gd_nss_mt = observed_yields_mt - model_yields_gd_nss_mt

# Parametri ottimali trovati
print('NELSON-SIEGEL-SVESSON MEDIUM - TERM')
print("Gradient Descent:")
print(f"  beta0: {gd_params_nss_mt[0]:.4f}, beta1: {gd_params_nss_mt[1]:.4f}, beta2: {gd_params_nss_mt[2]:.4f}, beta3: {gd_params_nss_mt[3]:.4f}, tau1: {gd_params_nss_mt[4]:.4f}, tau2: {gd_params_nss_mt[5]:.4f}")
print("Newton-Raphson:")
print(f"  beta0: {nr_params_nss_mt[0]:.4f}, beta1: {nr_params_nss_mt[1]:.4f}, beta2: {nr_params_nss_mt[2]:.4f}, beta3: {nr_params_nss_mt[3]:.4f}, tau1: {nr_params_nss_mt[4]:.4f}, tau2: {nr_params_nss_mt[5]:.4f}")

# Confronto
results_table_nss = pd.DataFrame({
    'Method': ['Gradient Descent', 'Newton-Raphson'],
    'Loss': [gd_loss_nss_mt, nr_loss_nss_mt],
    'Running time (s)': [gd_time_nss_mt, nr_time_nss_mt]
})
print(results_table_nss)

# Definire un range di maturità per la yield curve 
maturity_nss_mt = np.linspace(0.1, 15, 100)  # Maturità da 0.1 a 15 anni (esempio)

# Calcolo delle yield curve per maturità con i parametri ottimizzati
yield_curve_gd_nss_mt = nelson_siegel_svensson_nss(maturity_nss_mt, *gd_params_nss_mt)
yield_curve_nr_nss_mt = nelson_siegel_svensson_nss(maturity_nss_mt, *nr_params_nss_mt)


plt.figure(figsize=(8, 6))
plt.subplot(2, 1, 1)
plt.plot(t_mt, observed_yields_mt, label='Observed Yields', linestyle='', color='black', marker='.')
plt.plot(t_mt, model_yields_nr_nss_mt, label='Modeled Yields (Newton-Raphson)', color='green')
plt.plot(t_mt, model_yields_gd_nss_mt, label='Modeled Yields (Gradient Descent)', color='purple')
plt.title('MT: Yield Curve - Gradient Descent vs Newton-Raphson (NS-Svensson)')
plt.xlabel('Time (t)')
plt.ylabel('Yields')
plt.legend()
plt.grid()


# Residuals Plot - Gradient Descent vs Newton-Raphson (NS-Svensson)
plt.subplot(2, 1, 2)
plt.plot(t_mt, residuals_nr_nss_mt, label='Residuals (Newton-Raphson)', color='green')
plt.plot(t_mt, residuals_gd_nss_mt, label='Residuals (Gradient Descent)', color='purple')
plt.axhline(0, color='black')
plt.title('MT: Residuals - Gradient Descent vs Newton-Raphson (NS-Svensson)')
plt.xlabel('Time (t)')
plt.ylabel('Residuals')
plt.legend()
plt.grid()

plt.tight_layout()  
plt.show()

"""
# Yield Curve as a Function of Maturity (NS-Svensson)
plt.figure(figsize=(8, 6))
plt.plot(maturity_nss_mt, yield_curve_gd_nss_mt, label='Yield Curve (Gradient Descent)', color='purple')
plt.plot(maturity_nss_mt, yield_curve_nr_nss_mt, label='Yield Curve (Newton-Raphson)', color='green')
plt.title('MT: Yield Curve as a Function of Maturity (NS-Svensson)')
plt.xlabel('Maturity (Years)')
plt.ylabel('Yield (%)')
plt.legend()
plt.grid()
plt.show()
"""

# Calcolo delle curve per il modello Nelson-Siegel
curve_ns_st_gd = nelson_siegel(maturity_st, *gd_params_st)
curve_ns_st_nr = nelson_siegel(maturity_st, *nr_params_st)
curve_ns_10y_gd = nelson_siegel(maturity_mt, *gd_params_mt)
curve_ns_10y_nr = nelson_siegel(maturity_mt, *nr_params_mt)

# Calcolo delle curve per il modello Nelson-Siegel-Svensson
curve_nss_st_gd = nelson_siegel_svensson_nss(maturity_st, *gd_params_nss_st)
curve_nss_st_nr = nelson_siegel_svensson_nss(maturity_st, *nr_params_nss_st)
curve_nss_10y_gd = nelson_siegel_svensson_nss(maturity_mt, *gd_params_nss_mt)
curve_nss_10y_nr = nelson_siegel_svensson_nss(maturity_mt, *nr_params_nss_mt)

# Nelson-Siegel Model Yield Curve Comparison
plt.figure(figsize=(10, 6))
plt.plot(maturity_st, curve_ns_st_gd, label='NS ST (Gradient Descent)', linestyle='-', color='blue')
plt.plot(maturity_st, curve_ns_st_nr, label='NS ST (Newton-Raphson)', linestyle='--', color='blue')
plt.plot(maturity_mt, curve_ns_10y_gd, label='NS MT (Gradient Descent)', linestyle='-', color='green')
plt.plot(maturity_mt, curve_ns_10y_nr, label='NS MT (Newton-Raphson)', linestyle='--', color='green')
plt.title('Nelson-Siegel Yield Curve Comparison')
plt.xlabel('Maturity (Years)')
plt.ylabel('Yield (%)')
plt.legend()
plt.grid()
plt.show()

# Nelson-Siegel-Svensson Model Yield Curve Comparison
plt.figure(figsize=(10, 6))
plt.plot(maturity_st, curve_nss_st_gd, label='NSS ST (Gradient Descent)', linestyle='-', color='red')
plt.plot(maturity_st, curve_nss_st_nr, label='NSS ST (Newton-Raphson)', linestyle='--', color='red')
plt.plot(maturity_mt, curve_nss_10y_gd, label='NSS MT (Gradient Descent)', linestyle='-', color='purple')
plt.plot(maturity_mt, curve_nss_10y_nr, label='NSS MT (Newton-Raphson)', linestyle='--', color='purple')
plt.title('Nelson-Siegel-Svensson Yield Curve Comparison')
plt.xlabel('Maturity (Years)')
plt.ylabel('Yield (%)')
plt.legend()
plt.grid()
plt.show()

"""
#Ho implementato questo confronto quindi in base ai due metodi
# Confronto
print('\nFinal Comparison - Nelson-Siegel Model')
results_table = pd.DataFrame({
    'Method': ['Short-term: Gradient Descent', 'Short-term: Newton-Raphson', 'Medium-term: Gradient Descent', 'Medium-term: Newton-Raphson'],
    'Loss': [gd_loss_st, nr_loss_st, gd_loss_mt, nr_loss_mt],
    'Running time (s)': [gd_time_st, nr_time_st, gd_time_mt, nr_time_mt],
    'beta0': [gd_params_st[0], nr_params_st[0], gd_params_mt[0], nr_params_mt[0]],
    'beta1': [gd_params_st[1], nr_params_st[1], gd_params_mt[1], nr_params_mt[1]],
    'beta2': [gd_params_st[2], nr_params_st[2], gd_params_mt[2], nr_params_mt[2]],
    'tau': [gd_params_st[3], nr_params_st[3], gd_params_mt[3], nr_params_mt[3]]
})
pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', None)  
pd.set_option('display.float_format', '{:.4f}'.format)  
print(results_table)

# Confronto
print('\nFinal Comparison - Nelson-Siegel Svesson Model')
results_table_sv = pd.DataFrame({
    'Method': ['Short-term: Gradient Descent', 'Short-term: Newton-Raphson', 'Medium-term: Gradient Descent', 'Medium-term: Newton-Raphson'],
    'Loss': [gd_loss_nss_st, nr_loss_nss_st, gd_loss_nss_mt, nr_loss_nss_mt],
    'Running time (s)': [gd_time_nss_st, nr_time_nss_st, gd_time_nss_mt, nr_time_nss_mt],
    'beta0': [gd_params_nss_st[0], nr_params_nss_st[0], gd_params_nss_mt[0], nr_params_nss_mt[0]],
    'beta1': [gd_params_nss_st[1], nr_params_nss_st[1], gd_params_nss_mt[1], nr_params_nss_mt[1]],
    'beta2': [gd_params_nss_st[2], nr_params_nss_st[2], gd_params_nss_mt[2], nr_params_nss_mt[2]],
    'beta3': [gd_params_nss_st[3], nr_params_nss_st[3], gd_params_nss_mt[3], nr_params_nss_mt[3]],
    'tau1': [gd_params_nss_st[4], nr_params_nss_st[4], gd_params_nss_mt[4], nr_params_nss_mt[4]],
    'tau2': [gd_params_nss_st[5], nr_params_nss_st[5], gd_params_nss_mt[5], nr_params_nss_mt[5]]
})
pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', None)  
pd.set_option('display.float_format', '{:.4f}'.format)  
print(results_table_sv)
"""

#Ma penso si più utile confrontare le du durate
print('\nFinal Comparison - Short Term')
results_table = pd.DataFrame({
    'Method': ['Gradient Descent NS', 'Newton-Raphson NS', 'Gradient Descent NSS', 'Newton-Raphson NSS'],
    'Loss': [gd_loss_st, nr_loss_st, gd_loss_nss_st, nr_loss_nss_st],
    'Running time (s)': [gd_time_st, nr_time_st, gd_time_nss_st, nr_time_nss_st],
    'beta0': [gd_params_st[0], nr_params_st[0], gd_params_nss_st[0], nr_params_nss_st[0]],
    'beta1': [gd_params_st[1], nr_params_st[1], gd_params_nss_st[1], nr_params_nss_st[1]],
    'beta2': [gd_params_st[2], nr_params_st[2], gd_params_nss_st[2], nr_params_nss_st[2]],
    'beta3': ['-', '-', gd_params_nss_st[3], nr_params_nss_st[3]],
    'tau1': [gd_params_st[3], nr_params_st[3], gd_params_nss_st[4], nr_params_nss_st[4]],
    'tau2': ['-', '-', gd_params_nss_st[5], nr_params_nss_st[5]]
})
pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', None)  
pd.set_option('display.float_format', '{:.4f}'.format)  
print(results_table)

print('\nFinal Comparison - Medium Term')
results_table = pd.DataFrame({
    'Method': ['Gradient Descent NS', 'Newton-Raphson NS', 'Gradient Descent NSS', 'Newton-Raphson NSS'],
    'Loss': [gd_loss_mt, nr_loss_mt, gd_loss_nss_mt, nr_loss_nss_mt],
    'Running time (s)': [gd_time_mt, nr_time_mt, gd_time_nss_mt, nr_time_nss_mt],
    'beta0': [gd_params_mt[0], nr_params_mt[0], gd_params_nss_mt[0], nr_params_nss_mt[0]],
    'beta1': [gd_params_mt[1], nr_params_mt[1], gd_params_nss_mt[1], nr_params_nss_mt[1]],
    'beta2': [gd_params_mt[2], nr_params_mt[2], gd_params_nss_mt[2], nr_params_nss_mt[2]],
    'beta3': ['-', '-', gd_params_nss_mt[3], nr_params_nss_mt[3]],
    'tau1': [gd_params_mt[3], nr_params_mt[3], gd_params_nss_mt[4], nr_params_nss_mt[4]],
    'tau2': ['-', '-', gd_params_nss_mt[5], nr_params_nss_mt[5]]
})
pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', None)  
pd.set_option('display.float_format', '{:.4f}'.format)  
print(results_table)
