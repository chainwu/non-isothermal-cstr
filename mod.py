import numpy as np
import sympy as sp
from sympy import Symbol, Eq, Number
import torch

import modulus
from modulus.sym.hydra import instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_1d import Point1D
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
)
from modulus.sym.domain.monitor.pointwise import PointwiseMonitor
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.domain.inferencer.pointwise import PointwiseInferencer
from modulus.sym.key import Key
from modulus.sym.node import Node
from modulus.sym.utils.io.plotter import ValidatorPlotter, InferencerPlotter
from modulus.sym.geometry.parameterization import Parameterization
import matplotlib.pyplot as plt
import sys
import os
from cstr import CSTR, k_0, EoverR

class ReactorPlotter(InferencerPlotter):
    def __call__(self, invar, outvar):
        try:
            # get input variables
            t = invar["t"][:,0]
            #print("shape", t.shape, T_c.shape)
            C_A, T, T_c = outvar["C_A"], outvar["T"], outvar["T_c"]
            #print(C_A, T)
        
            # make plot
            plt.figure(figsize=(20,5), dpi=100)
            plt.suptitle("CSTR Reactor")
            fig, ax=plt.subplots(3,1)
            ax[0].plot(t, C_A)
            ax[0].set_xlim([0, 10])
            #ax[0].set_ylim([0, 1])
            ax[0].set_title("Reactant Concentration (mol/L)")
        
            ax[1].plot(t, T)
            ax[1].set_title("Reactor Temp (K)")
            ax[1].set_xlim([0, 10])
            #ax[1].set_ylim([300, 450])

            ax[2].plot(t, T_c)
            ax[2].set_title("Cooling water Temp (K)")
            ax[2].set_xlim([0, 10])
            #ax[1].set_ylim([300, 450])
            plt.tight_layout()

            return [(fig, "custom_plot")]
        except:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


#def add_inferencer(nodes, domain):
    # add inferencer data
#    tdict=np.linspace(0, 10, 1000).reshape(-1,1)
#    tcdict=np.full((1000), 290.).reshape(-1,1)
#    inf290 = PointwiseInferencer(
#        nodes=nodes,
#        invar={'t': tdict, 'T_c': tcdict},
#        output_names=["C_A", "T"],
#        batch_size=1000,
#        requires_grad=False,
#        plotter=InferencerPlotter(),
#    )
#    domain.add_inferencer(inf290, "Reactor-290")

#    tcdict=np.full((1000), 300.).reshape(-1,1)
#    inf300 = PointwiseInferencer(
#        nodes=nodes,
#        invar={'t': tdict, 'T_c': tcdict},
#        output_names=["C_A", "T"],
#        requires_grad=False,
#        batch_size=1000,
#        #plotter=ReactorPlotter(),
#    )
#    domain.add_inferencer(inf300, "Reactor-300")
        
#    tcdict=np.full((1000), 305.).reshape(-1,1)
#    inf305 = PointwiseInferencer(
#        nodes=nodes,
#        invar={'t': tdict, 'T_c': tcdict},
#        output_names=["C_A", "T"],
#        batch_size=1000,
#        requires_grad=False,
#        plotter=InferencerPlotter(),
#    )
#    domain.add_inferencer(inf305, "Reactor-305")


# takes in a module and applies the specified weight initialization
def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-1, 0)
        m.bias.data.fill_(0)

@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    # make list of nodes to unroll graph on
    T_c=300
    reactor = CSTR(T_c)
    reactor_net = instantiate_arch(
        input_keys=[Key("t")],
        output_keys=[Key("T"), Key("C_A"), Key("T_c")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = reactor.make_nodes()  + [reactor_net.make_node(name="reactor_network")]

    #weights_init_uniform_rule(nodes)

    # add constraints to solver
    # make geometry
    geo = Point1D(0)
    t_max = 10.0  #min
    t = Symbol("t")
    param_range = Parameterization({t: (0, t_max), "T_c": 300})
    param_range.sample(10000)
    #k = Symbol("k")
    # make domain
    domain = Domain()

    # initial conditions
    IC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"T": 350, "C_A": 0.5}, 
        batch_size=cfg.batch_size.IC,
        lambda_weighting={
            "T": 1.0,
            "C_A": 1.0,
        },
        parameterization={t: 0},
    )
    domain.add_constraint(IC, name="IC")

    # solve over given time period
    tc_cond = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"T_c":300},
        batch_size=cfg.batch_size.Interior,
        parameterization=param_range,
    )
    domain.add_constraint(tc_cond, "tc_cond")

    # add monitors
    #tnpy=np.linspace(0, 10, 1000).reshape(-1,1)
    #tcnpy=np.full((1000), 300.).reshape(-1,1)
    #gmon = PointwiseMonitor(
    #    invar={'t':tnpy, 'T_c':tcnpy},
    #    output_names=["T", "C_A"],
    #    metrics={
    # solve over given time period
    interior = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"material_balance": 0.0, "energy_balance": 0.0},
        batch_size=cfg.batch_size.Interior,
        parameterization=param_range,
    )
    domain.add_constraint(interior, "interior")
    #        "Temp": lambda var: torch.mean(var["T"]),
    #        "Temp_Std": lambda var: torch.std(var["T"]),
    #        "Conc": lambda var: torch.mean(var["C_A"]),
    #        "Conc_Std": lambda var: torch.std(var["C_A"]),
    #    },
    #    nodes=nodes,
    #)
    #domain.add_monitor(gmon)

    tdict=np.linspace(0, 10, 10000).reshape(-1,1)
    #tcdict=np.full((1000), 300.).reshape(-1,1)
    inf300 = PointwiseInferencer(
        nodes=nodes,
        invar={'t': tdict}, #, 'T_c': tcdict},
        output_names=["C_A", "T", "T_c"],
        requires_grad=False,
        batch_size=10000,
        plotter=ReactorPlotter()
    )
    domain.add_inferencer(inf300, "Reactor-300")

# make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()

