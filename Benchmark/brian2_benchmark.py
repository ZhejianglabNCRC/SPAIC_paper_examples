from brian2 import *
import sympy

seed(11922)  # to get identical figures for repeated runs

################################################################################
# Model parameters
################################################################################
### General parameters
# duration = 0.1*second  # Total simulation time
sim_dt = 1.0*ms        # Integrator/sampling step
N_e = 5000           # Number of excitatory neurons
N_i = 5000              # Number of inhibitory neurons

### Neuron parameters
E_l = -60*mV           # Leak reversal potential
g_l = 9.99*nS          # Leak conductance
E_e = 0*mV             # Excitatory synaptic reversal potential
E_i = -80*mV           # Inhibitory synaptic reversal potential
C_m = 198*pF           # Membrane capacitance
tau_e = 5*ms           # Excitatory synaptic time constant
tau_i = 10*ms          # Inhibitory synaptic time constant
tau_r = 5*ms           # Refractory period
I_ex = 150*pA          # External current
V_th = -50*mV          # Firing threshold
V_r = E_l              # Reset potential

### Synapse parameters
w_e = 0.05*nS          # Excitatory synaptic conductance
w_i = 1.0*nS           # Inhibitory synaptic conductance
U_0 = 0.6              # Synaptic release probability at rest
Omega_d = 2.0/second   # Synaptic depression rate
Omega_f = 3.33/second  # Synaptic facilitation rate

################################################################################
# Model definition
################################################################################
# Set the integration time (in this case not strictly necessary, since we are
# using the default value)
defaultclock.dt = sim_dt

### Neurons
eqs_neurons = '''
    dv/dt = (ge * (-60 * mV) + (-74 * mV) - v) / (10 * ms) : volt
    dge/dt = -ge / (5 * ms) : 1
'''

input = PoissonGroup(100, rates=30.0 * Hz)
layer1 = NeuronGroup(
    10**7, eqs_neurons, threshold='v > (-54 * mV)', reset='v = -60 * mV', method='exact'
)
layer1.v = -60*mV

### Synapses

S1 = Synapses(
    input, layer1, '''w: 1''', on_pre='v += w * mV'
)
S1.connect(p=1)
S1.w = 'rand() * 10'


# ##############################################################################
# # Simulation run
# ##############################################################################
import time
t1 = time.time()
for i in range(100):
    run(100 * ms)

print(time.time()-t1)
