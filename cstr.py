import sympy as sp
from sympy import Symbol, Eq, Function, exp, Number
from modulus.sym.eq.pde import PDE
import math

# Constants
V = 100 # L
q = 100# L/min
C_Ai = 1 # mol/L
T_i = 350 #K
C_A0 = 0.5 # mol/L
T_0 = 350 # K
T_c0 = 300 # K
rho = 1000 # g/L
C = 0.239  # J/kg.K
k_0 = 7.2E10  # min^-1
EoverR = 8750  #k
delta_H_r = -50000  # J/mol
UA = 50000   #J/min K
#k = k_0 * math.exp(-(EoverR/350.))

def k(T):
    return k_0 * sp.exp(-(EoverR/T))

class CSTR(PDE):
    name="cstr"

    def __init__(self):
        # Define symbols
        #self.T_c = T_c
        t, C_A, T, T_c = Symbol("t"), Symbol("C_A"), Symbol("T"), Symbol("T_c")

        # make input variables
        input_variables = {"t":t}
        #if type(T_ci) is str:
        #    T_c = Function("T_c")(t)           # cooling water temp is a function of time
        #elif type(T_ci) in [float, int]:
        #    T_c = Number(T_ci)
        #C_A = Function("C_A")(t)    # Concentration of the reactant is a function of time and cooling water temp
        #T = Function("T")(t)        # Reactor temp is a function of time and cooling water temp
        #T_c = Function("T_c")(t)
        #k = Function("k")(t)

        # set equations
        self.equations = {}

        # k 
        #self.equations["k_equation"] = k - (k_0 * sp.exp(-(EoverR/T)))

        # Energy balance W=q * rho
        self.equations["energy_balance"] = ((T.diff(t,1)) - (((q / V) * (T_i - T)) + ((-delta_H_r * k(T) * C_A) / (rho * C)) + ((UA *(T_c - T)) / (rho * C * V))))

        # Material balance        
        self.equations["material_balance"] = ((C_A.diff(t,1)) - ((q / V) * (C_Ai - C_A) - (k(T) * C_A)))

if __name__ == '__main__':
    for tx in [290,300,310]:
        print("T_c=",tx)
        #cstr=CSTR(T_ci=tx)
        cstr=CSTR()
        cstr.pprint()


