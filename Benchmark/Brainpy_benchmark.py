import brainpy as bp
import brainpy.math as bpm
import brainmodels as bm

device = 'gpu'
# device = 'cpu'
scale = 6
bp.math.set_platform(device)
bp.math.set_dt(1.0)

# %%
# set parameters

num = 1000
num_inh = int(num * 0.2)
num_exc = num - num_inh
prob = 0.25

tau_E = 15.
tau_I = 10.
V_reset = 0.
V_threshold = 15.
f_E = 3.
f_I = 2.
mu_f = 0.1

tau_Es = 6.
tau_Is = 5.
JEE = 0.25
JEI = -1.
JIE = 0.4
JII = -1.


class BNet(bp.Network):
  def __init__(self):
    # neurons
    pars = dict(V_rest=-52., V_th=-50., V_reset=-60., tau=10., tau_ref=0.)

    self.layer1 = bm.neurons.LIF(100, **pars)
    self.layer2 = bm.neurons.LIF(10**scale, **pars)

    self.layer1.V[:] = 0
    self.layer2.V[:] = 0

    if device == 'gpu':
        # gpu
        c1 = bm.synapses.AMPA(pre=self.layer1, post=self.layer2, conn=bp.conn.All2All())
    elif device == 'cpu':
        #cpu
        c1 = bp.models.ExpCUBA(pre=self.layer1, post=self.layer2, conn=bp.conn.All2All())
    # c2 = bm.synapses.AMPA(pre=self.layer2, post=self.layer3, conn=bp.conn.All2All())
    # c3 = bm.synapses.AMPA(pre=self.layer3, post=self.layer4, conn=bp.conn.All2All())
    # c1 = bm.synapses.AMPA(pre=self.layer1, post=self.layer2, conn=bp.conn.FixedProb(1.0))

    # # synapses
    # c1 = bp.models.ExpCUBA(pre=self.layer1, post=self.layer2, conn=bp.conn.FixedProb(1.0), tau=tau_Es, g_max=JIE)
    # c2 = bp.models.ExpCUBA(pre=layer2, post=layer3, conn=bp.conn.FixedProb(1.0), tau=tau_Es, g_max=JEE)
    # I2I = bp.models.ExpCUBA(pre=I, post=I, conn=bp.conn.FixedProb(prob), tau=tau_Is, g_max=JII)
    # I2E = bp.models.ExpCUBA(pre=I, post=E, conn=bp.conn.FixedProb(prob), tau=tau_Is, g_max=JEI)


    super(BNet, self).__init__(c1, self.layer1, self.layer2)


# %%
net = BNet()

# %%
runner = bp.StructRunner(net,
                         # monitors=['E.spike', 'I.spike'],
                         # inputs=[('layer1.input', f_E * bpm.sqrt(num) * mu_f)],
                        inputs=[('layer1.input', 30.0)],
                         jit=True)

# runner = bp.dyn.runners.DSRunner(net,
#                          # monitors=['E.spike', 'I.spike'],
#                          inputs=[('layer1.input', f_E * bpm.sqrt(num) * mu_f)],
#                          jit=True)

import time
a = time.time()
for i in range(100):
    t = runner.run(100.)
print(time.time() - a)