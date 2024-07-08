import numpy as np
import sympy as sp
from sympy import Symbol, Eq

import modulus
from modulus.sym.hydra import instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_1d import Point1D
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseBoundaryConstraint,
)
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.key import Key
from modulus.sym.node import Node

from cstr import CSTR, k_0, EoverR

@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    # make list of nodes to unroll graph on
    reactor = CSTR()
    reactor_net = instantiate_arch(
        input_keys=[Key("t"), Key("T_c")],
        output_keys=[Key("T"), Key("C_A")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = reactor.make_nodes()  + [reactor_net.make_node(name="reactor_network")]

    # add constraints to solver
    # make geometry
    geo = Point1D(0)
    t_max = 10.0  #min
    t = Symbol("t")
    param_range = {t: (0, t_max), "T_c": 300}
    k = Symbol("k")

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
        parameterization={t: 0, "T_c":300, "T_c__t":0},
    )
    domain.add_constraint(IC, name="IC")

    # solve over given time period
    interior = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"material_balance": 0.0, "energy_balance": 0.0},
        batch_size=cfg.batch_size.Interior,
        parameterization=param_range,
    )
    domain.add_constraint(interior, "interior")

# make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()

