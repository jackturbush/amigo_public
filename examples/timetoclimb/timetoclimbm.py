# Unscaled, multiple components for the dynamics, then linked together
import amigo as am
import numpy as np
import argparse
import json
import sys

num_time_steps = 100

"""
Min Time to Climb 
==============================

This is example from Betts' book "Practical Methods for Optimal Control
Using NonlinearProgramming", 3rd edition, Chapter 10: Test Problems.

This is the timetoclimb example with multiple components: one to solve the dynamics and the other one for calculating the aerodynamic properties
"""


class TrapezoidRule(am.Component):
    def __init__(self):
        super().__init__()

        self.add_constant("num_time_steps", value=num_time_steps, type=int)
        # Scalar inputs
        self.add_input("tf")  # final time
        self.add_input("q1")  # state value at the t1
        self.add_input("q2")  # state value at the t2
        self.add_input("q1dot")  # state derivative at the t1
        self.add_input("q2dot")  # state derivative at the t2

        # Add the residual
        self.add_constraint("res")

        return

    def compute(self):
        tf = self.inputs["tf"]
        N = self.constants["num_time_steps"]
        dt = tf / N

        q1 = self.inputs["q1"]
        q2 = self.inputs["q2"]
        q1dot = self.inputs["q1dot"]
        q2dot = self.inputs["q2dot"]

        self.constraints["res"] = q2 - q1 - 0.5 * dt * (q1dot + q2dot)


class AircraftDynamics(am.Component):
    def __init__(self):
        super().__init__()

        # Add Constants
        self.add_constant("m0", value=19030.468)
        self.add_constant("g", value=9.81)
        self.add_constant("Isp", value=1600.0)
        self.add_constant("TtoW", value=0.9)
        self.add_constant("S_ref", value=49.2386)

        # Add Inputs (AeroForces)
        self.add_input("L")
        self.add_input("D")

        # Add the control variable
        self.add_input("alpha")

        # Add the state variables
        self.add_input("q", shape=5)
        self.add_input("qdot", shape=5)

        # Add the residual output
        self.add_constraint("res", shape=5)

        return

    def compute(self):
        m0 = self.constants["m0"]
        g = self.constants["g"]
        Isp = self.constants["Isp"]
        TtoW = self.constants["TtoW"]
        S_ref = self.constants["S_ref"]

        L = self.inputs["L"]
        D = self.inputs["D"]
        alpha = self.inputs["alpha"]
        q = self.inputs["q"]
        qdot = self.inputs["qdot"]

        # State variables
        v = q[0]
        gamma = q[1]
        h = q[2]
        r = q[3]
        m = q[4]

        # Compute the thrust
        T = self.constants["TtoW"] * self.constants["m0"] * self.constants["g"]

        # Residual Equations
        self.constraints["res"] = [
            qdot[0] - ((T / m) * am.cos(alpha) - (D / m) - g * am.sin(gamma)),
            qdot[1]
            - ((T / (m * v)) * am.sin(alpha) + L / (m * v) - (g / v) * am.cos(gamma)),
            qdot[2] - v * am.sin(gamma),
            qdot[3] - v * am.cos(gamma),
            qdot[4] + T / (g * Isp),
        ]
        return


class AeroModel(am.Component):
    def __init__(self):
        super().__init__()

        # Add Constants
        self.add_constant("CL_alpha", value=3.44)
        self.add_constant("CD0", value=0.013)
        self.add_constant("kappa", value=0.54)
        self.add_constant("rho", value=1.225)
        self.add_constant("v_inf", value=136.0)
        self.add_constant("S_ref", value=49.2386)

        # Add Inputs
        self.add_input("alpha")
        self.add_input("v")
        self.add_input("L")
        self.add_input("D")

        # Add Constraints we're enforcing
        self.add_constraint("L_res")
        self.add_constraint("D_res")

        return

    def compute(self):
        CL_alpha = self.constants["CL_alpha"]
        CD0 = self.constants["CD0"]
        kappa = self.constants["kappa"]
        rho = self.constants["rho"]
        S_ref = self.constants["S_ref"]

        L = self.inputs["L"]
        D = self.inputs["D"]
        alpha = self.inputs["alpha"]
        v = self.inputs["v"]

        CL = CL_alpha * alpha
        CD = CD0 + kappa * CL**2
        q_dyn = 0.5 * rho * v * v

        # Residual Equations
        self.constraints["L_res"] = L - q_dyn * S_ref * CL
        self.constraints["D_res"] = D - q_dyn * S_ref * CD

        return


class TimeObjective(am.Component):
    def __init__(self):
        super().__init__()

        # design variable scalar (same one is fed to all trapezoid residuals)
        self.add_input("tf")

        # objective
        self.add_objective("obj")

    def compute(self):
        self.objective["obj"] = self.inputs["tf"]


class InitialConditions(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("q", shape=5)  # [v, γ, h, r, m]
        self.add_constraint("res", shape=5)  # residual = q − q0

    def compute(self):
        v0, gam0, h0, r0, m0 = 135.964, 0.0, 100.0, 0.0, 19030.468
        q = self.inputs["q"]
        self.constraints["res"] = [
            q[0] - v0,
            q[1] - gam0,
            q[2] - h0,
            q[3] - r0,
            q[4] - m0,
        ]


class FinalConditions(am.Component):
    def __init__(self):
        super().__init__()
        self.add_constant("T", value=288.15)  # temperature Kelvin
        self.add_constant("R", value=287.058)  # gas constant for air
        self.add_input("q", shape=5)  # final state
        self.add_constraint("res", shape=3)  # h_f, γ_f, v_f (Mach=1)

    def compute(self):
        T = self.constants["T"]
        R = self.constants["R"]
        q = self.inputs["q"]
        a = (1.4 * R * T) ** 0.5
        hf, gam_f, vf = 20000.0, 0.0, a  # v = 1 Mach
        self.constraints["res"] = [
            q[2] - hf,  # altitude
            q[1] - gam_f,  # flight-path angle
            q[0] - vf,
        ]  # velocity


# Build and link the model
def create_climb_time_model(module_name="climb_time"):
    # Build the components
    ac = AircraftDynamics()
    aero = AeroModel()
    trap = TrapezoidRule()
    obj = TimeObjective()
    ic = InitialConditions()
    fc = FinalConditions()

    model = am.Model("min_time_climb")

    model.add_component("ac", num_time_steps + 1, ac)
    model.add_component("aero", num_time_steps + 1, aero)
    model.add_component("trap", 5 * num_time_steps, trap)

    # scalar components
    model.add_component("obj", 1, obj)
    model.add_component("ic", 1, ic)
    model.add_component("fc", 1, fc)

    # Linking:
    # Link the state variables with the trapezoid rule
    for i in range(5):
        s, e = i * num_time_steps, (i + 1) * num_time_steps
        # Link the state variables
        model.link(f"ac.q[:{num_time_steps}, {i}]", f"trap.q1[{s}:{e}]")
        model.link(f"ac.q[1:, {i}]", f"trap.q2[{s}:{e}]")

        # Link the state rates
        model.link(f"ac.qdot[:-1, {i}]", f"trap.q1dot[{s}:{e}]")
        model.link(f"ac.qdot[1:, {i}]", f"trap.q2dot[{s}:{e}]")

    # Link objective to the trapezoid rule
    model.link("obj.tf", "trap.tf[:]")

    # Link aero outputs -> dynamics inputs  (THIS direction)
    model.link("aero.L", "ac.L")
    model.link("aero.D", "ac.D")

    # Link initial conditions and final conditions
    model.link("ac.q[0, :]", "ic.q[0, :]")
    model.link(f"ac.q[{num_time_steps}, :]", "fc.q[0, :]")

    # Link control and velocity
    model.link("ac.alpha", "aero.alpha")
    model.link("ac.q[:,0]", "aero.v")
    return model


parser = argparse.ArgumentParser()
parser.add_argument(
    "--build", dest="build", action="store_true", default=False, help="Enable building"
)
parser.add_argument(
    "--with-openmp",
    dest="use_openmp",
    action="store_true",
    default=False,
    help="Enable OpenMP",
)
parser.add_argument(
    "--show-sparsity",
    dest="show_sparsity",
    action="store_true",
    default=False,
    help="Show the sparsity pattern",
)
args = parser.parse_args()

model = create_climb_time_model()

if args.build:
    compile_args = []
    link_args = []
    define_macros = []
    if args.use_openmp:
        compile_args = ["-fopenmp"]
        link_args = ["-fopenmp"]
        define_macros = [("AMIGO_USE_OPENMP", "1")]

    model.build_module(
        compile_args=compile_args, link_args=link_args, define_macros=define_macros
    )

model.initialize(order_type=am.OrderingType.NESTED_DISSECTION)

with open("climb_time_model.json", "w") as fp:
    json.dump(model.get_serializable_data(), fp, indent=2)


print(f"Num variables:              {model.num_variables}")
print(f"Num constraints:            {model.num_constraints}")

prob = model.get_opt_problem()

# Get the design variables
x = model.create_vector()
x[:] = 0.0

# Initial values for the state variables
tf_guess = 300.0  # in seconds
x["obj.tf"] = tf_guess

# Set initial guess for states
t_guess = np.linspace(0, tf_guess, num_time_steps + 1)
x["ac.q[:, 0]"] = 136.0 + (340.0 - 136.0) * t_guess / tf_guess  # velocity
x["ac.q[:, 1]"] = np.radians(
    5.0 * np.sin(np.pi * t_guess / tf_guess)
)  # flight path angle
x["ac.q[:, 2]"] = 100.0 + (20000.0 - 100.0) * t_guess / tf_guess  # altitude
x["ac.q[:, 3]"] = 5000.0 * t_guess  # range
x["ac.q[:, 4]"] = 19030.0 - 200.0 * t_guess / tf_guess  # mass decrease

# Set initial guess for control (constant small angle)
x["ac.alpha"] = np.radians(1.0)  # degrees
# alpha_guess = np.radians(3.0 * np.sin(np.pi * t_guess / tf_guess))

# # Initial values for the aero forces
# CL_alpha = 3.44  # per rad (already rad-based)
# CD0 = 0.013
# kappa = 0.54
# rho = 1.225
# S = 49.2386

# v_traj = 136.0 + (340.0 - 136.0) * t_guess / tf_guess

# CL = CL_alpha * alpha_guess
# CD = CD0 + kappa * CL**2
# q_dyn = 0.5 * rho * v_traj**2
# L_guess = q_dyn * S * CL
# D_guess = q_dyn * S * CD

# x["aero.L"] = L_guess
# x["aero.D"] = D_guess

# Set the bounds
lower = model.create_vector()
upper = model.create_vector()

# α bounds
lower["ac.alpha"] = -8.0 * np.pi / 180.0
upper["ac.alpha"] = 8.0 * np.pi / 180.0

# Velocity
lower["ac.q[:,0]"] = 10.0
upper["ac.q[:,0]"] = 1000.0

# Flight-path angle
lower["ac.q[:, 1]"] = -1.5
upper["ac.q[:, 1]"] = 1.5

# Altitude between 0 and 30 000 m
lower["ac.q[:,2]"] = 0.0
upper["ac.q[:,2]"] = 30000.0

# Range
lower["ac.q[:,3]"] = 0.0
upper["ac.q[:,3]"] = 1.0e6

# Mass
lower["ac.q[:,4]"] = 50.0
upper["ac.q[:,4]"] = 19030.0

# Flight-time bounds
lower["obj.tf"] = 100.0
upper["obj.tf"] = 1000.0

# Create and run optimizer
opt = am.Optimizer(model, x, lower=lower, upper=upper)
data = opt.optimize({"max_iterations": 200, "record_components": ["obj.tf[0]"]})

# Save optimization data
with open("time_to_climb_opt_data.json", "w") as fp:
    json.dump(data, fp, indent=2)

# Extract final flight time
tf = x["obj.tf[0]"]
print(f"\nOptimal time to climb: {tf:.2f} seconds")
