import amigo as am
import numpy as np
import sys
import matplotlib.pylab as plt
import niceplots
import argparse
import json

# Problem parameters
num_time_steps = 100

"""
Min Time to Climb 
==============================

This is example from Betts' book "Practical Methods for Optimal Control
Using NonlinearProgramming", 3rd edition, Chapter 10: Test Problems.

This is a non-scaled problem with a single component dynamics code
"""


class TrapezoidRule(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("tf")  # Final time (back as design variable)
        self.add_input("q1")
        self.add_input("q2")
        self.add_input("q1dot")
        self.add_input("q2dot")

        self.add_constraint("res")

        return

    def compute(self):
        tf = self.inputs["tf"]
        q1 = self.inputs["q1"]
        q2 = self.inputs["q2"]
        q1dot = self.inputs["q1dot"]
        q2dot = self.inputs["q2dot"]

        dt = tf / num_time_steps  # Variable time step based on final time
        self.constraints["res"] = q2 - q1 - 0.5 * dt * (q1dot + q2dot)

        return


class AircraftDynamics(am.Component):
    def __init__(self):
        super().__init__()

        # Physical constants matching the original code
        self.add_constant("S", value=49.2386)  # m^2
        self.add_constant("CL_alpha", value=3.44)
        self.add_constant("CD0", value=0.013)
        self.add_constant("kappa", value=0.54)
        self.add_constant("Isp", value=1600.0)  # s
        self.add_constant("TtoW", value=0.9)
        self.add_constant("m0", value=19030.0)  # kg
        self.add_constant("gamma_air", value=1.4)
        self.add_constant("R", value=287.058)
        self.add_constant("g", value=9.81)
        self.add_constant("conv", value=np.pi / 180.0)

        # Inputs
        self.add_input("alpha", label="angle of attack (degrees)")
        self.add_input("q", shape=5, label="state variables")
        self.add_input("qdot", shape=5, label="state derivatives")

        # Constraint residuals
        self.add_constraint("res", shape=5, label="dynamics residual")

        return

    def compute(self):
        # Get constants
        S = self.constants["S"]
        CL_alpha = self.constants["CL_alpha"]
        CD0 = self.constants["CD0"]
        kappa = self.constants["kappa"]
        Isp = self.constants["Isp"]
        TtoW = self.constants["TtoW"]
        m0 = self.constants["m0"]
        gamma_air = self.constants["gamma_air"]
        R = self.constants["R"]
        g = self.constants["g"]
        conv = self.constants["conv"]

        # Get inputs
        alpha = self.inputs["alpha"]
        q = self.inputs["q"]
        qdot = self.inputs["qdot"]

        # State variables
        v = q[0]  # velocity [m/s]
        gamma = q[1]  # flight path angle [degrees]
        h = q[2]  # altitude [m]
        r = q[3]  # range [m]
        m = q[4]  # mass [kg]

        # Atmospheric properties (simplified)
        T_atm = 288.15  # K
        rho = 1.225  # kg/m^3

        # Aerodynamics matching original code
        CL = CL_alpha * conv * alpha  # Convert alpha from degrees to radians

        # Drag coefficient (simplified - no compressibility for now)
        CD = CD0 + kappa * CL**2

        # Dynamic pressure and forces
        qinfty = 0.5 * rho * v * v
        D = qinfty * S * CD
        L = qinfty * S * CL

        # Thrust
        T = TtoW * m0 * g

        # Convert angles to radians for dynamics
        alpha_rad = conv * alpha
        gamma_rad = conv * gamma

        # Aircraft dynamics equations (matching original computeSystemResidual)
        res = [
            qdot[0] - ((T / m) * am.cos(alpha_rad) - (D / m) - g * am.sin(gamma_rad)),
            qdot[1]
            - (
                T / (m * v) * am.sin(alpha_rad)
                + L / (m * v)
                - (g / v) * am.cos(gamma_rad)
            )
            / conv,
            qdot[2] - v * am.sin(gamma_rad),
            qdot[3] - v * am.cos(gamma_rad),
            qdot[4] + T / (g * Isp),
        ]

        self.constraints["res"] = res

        return


class Objective(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("tf", label="final time")
        self.add_objective("obj")

        return

    def compute(self):
        tf = self.inputs["tf"]
        self.objective["obj"] = tf  # Minimize final time

        return


class InitialConditions(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("q", shape=5)
        self.add_constraint("res", shape=5)

    def compute(self):
        q = self.inputs["q"]

        # Initial conditions matching original getInitConditions
        # [v0, gamma0, h0, r0, m0] = [136.0, 0.0, 100.0, 0.0, 19030.0]
        self.constraints["res"] = [
            q[0] - 136.0,  # velocity [m/s]
            q[1] - 0.0,  # flight path angle [degrees]
            q[2] - 100.0,  # altitude [m]
            q[3] - 0.0,  # range [m]
            q[4] - 19030.0,  # mass [kg]
        ]


class FinalConditions(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("q", shape=5)  # final state
        self.add_constraint("res", shape=3)  # h_f, γ_f, v_f (Mach=1)

    def compute(self):
        q = self.inputs["q"]
        hf, gam_f, vf = 20000.0, 0.0, 340.0  # v = 1 Mach
        self.constraints["res"] = [
            q[0] - vf,
            q[1] - gam_f,
            q[2] - hf,
        ]


def create_time_to_climb_model(module_name="time_to_climb"):
    # Create component instances
    ac = AircraftDynamics()
    trap = TrapezoidRule()
    obj = Objective()
    ic = InitialConditions()
    fc = FinalConditions()

    model = am.Model(module_name)

    # Add components to the model
    model.add_component("ac", num_time_steps + 1, ac)
    model.add_component("trap", 5 * num_time_steps, trap)  # 5 states
    model.add_component("obj", 1, obj)  # Only 1 objective instance for final time
    model.add_component("ic", 1, ic)
    model.add_component("fc", 1, fc)

    # Link the trapezoidal rule for each state
    for i in range(5):
        start = i * num_time_steps
        end = (i + 1) * num_time_steps

        # Link state variables
        model.link(f"ac.q[:{num_time_steps}, {i}]", f"trap.q1[{start}:{end}]")
        model.link(f"ac.q[1:, {i}]", f"trap.q2[{start}:{end}]")

        # Link state derivatives
        model.link(f"ac.qdot[:-1, {i}]", f"trap.q1dot[{start}:{end}]")
        model.link(f"ac.qdot[1:, {i}]", f"trap.q2dot[{start}:{end}]")

    # Link final time from objective to all trapezoidal rule components
    model.link("obj.tf[0]", f"trap.tf[:]")

    # Link boundary conditions
    model.link("ac.q[0, :]", "ic.q[0, :]")
    model.link(f"ac.q[{num_time_steps}, :]", "fc.q[0, :]")  # Temporarily remove

    return model


def plot_results(t, q, alpha):
    """Plot the optimization results"""

    with plt.style.context(niceplots.get_style()):
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))

        # State variables
        axes[0, 0].plot(t, q[:, 0])
        axes[0, 0].set_ylabel("Velocity (m/s)")
        axes[0, 0].set_xlabel("Time (s)")
        axes[0, 0].grid(True)

        axes[0, 1].plot(t, q[:, 1])
        axes[0, 1].set_ylabel("Flight path angle (deg)")
        axes[0, 1].set_xlabel("Time (s)")
        axes[0, 1].grid(True)

        axes[1, 0].plot(t, q[:, 2] / 1000)
        axes[1, 0].set_ylabel("Altitude (km)")
        axes[1, 0].set_xlabel("Time (s)")
        axes[1, 0].grid(True)

        axes[1, 1].plot(t, q[:, 3] / 1000)
        axes[1, 1].set_ylabel("Range (km)")
        axes[1, 1].set_xlabel("Time (s)")
        axes[1, 1].grid(True)

        axes[2, 0].plot(t, q[:, 4])
        axes[2, 0].set_ylabel("Mass (kg)")
        axes[2, 0].set_xlabel("Time (s)")
        axes[2, 0].grid(True)

        axes[2, 1].plot(t, alpha)
        axes[2, 1].set_ylabel("Angle of attack (deg)")
        axes[2, 1].set_xlabel("Time (s)")
        axes[2, 1].grid(True)

        plt.tight_layout()
        plt.savefig("time_to_climb_results.png", dpi=300, bbox_inches="tight")
        plt.show()


# Command line argument parsing
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
args = parser.parse_args()

# Create the model
model = create_time_to_climb_model()

# Build the module if requested
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

# Initialize the model
model.initialize(order_type=am.OrderingType.NESTED_DISSECTION)

print(f"Num variables:              {model.num_variables}")
print(f"Num constraints:            {model.num_constraints}")

# Create the design vector
x = model.create_vector()
x[:] = 0.0

# Set initial guess for final time
tf_guess = 300.0
x["obj.tf"] = tf_guess

# Set initial guess for states (reasonable trajectory)
t_guess = np.linspace(0, tf_guess, num_time_steps + 1)
x["ac.q[:, 0]"] = 136.0 + (340.0 - 136.0) * t_guess / tf_guess  # velocity
x["ac.q[:, 1]"] = 5.0 * np.sin(np.pi * t_guess / tf_guess)  # flight path angle
x["ac.q[:, 2]"] = 100.0 + (20000.0 - 100.0) * t_guess / tf_guess  # altitude
x["ac.q[:, 3]"] = 5000.0 * t_guess  # range
x["ac.q[:, 4]"] = 19030.0 - 200.0 * t_guess / tf_guess  # mass decrease

# Set initial guess for control (constant small angle)
x["ac.alpha"] = 1.0  # degrees
alpha_guess = 3.0 * np.sin(np.pi * t_guess / tf_guess)

# Set up bounds
lower = model.create_vector()
upper = model.create_vector()

# Final time bounds
lower["obj.tf"] = 100.0
upper["obj.tf"] = 1000.0

# Control bounds (matching original -5 to 5 degrees)
lower["ac.alpha"] = -5.0
upper["ac.alpha"] = 5.0

# State bounds
lower["ac.q[:, 0]"] = 50.0  # minimum velocity
upper["ac.q[:, 0]"] = 500.0  # maximum velocity
lower["ac.q[:, 1]"] = -85.0  # flight path angle
upper["ac.q[:, 1]"] = 85.0
lower["ac.q[:, 2]"] = 0.0  # altitude
upper["ac.q[:, 2]"] = 25000.0
lower["ac.q[:, 3]"] = 0.0  # range
lower["ac.q[:, 4]"] = 10000.0  # minimum mass
upper["ac.q[:, 4]"] = 20000.0  # maximum mass

# Optimize
opt = am.Optimizer(model, x, lower=lower, upper=upper)
data = opt.optimize({"max_iterations": 100, "rtol": 1e-6})

# Save optimization data
with open("time_to_climb_opt_data.json", "w") as fp:
    json.dump(data, fp, indent=2)

# Extract results
tf_opt = x["obj.tf"][0]  # Extract scalar from array
q = x["ac.q"]
alpha = x["ac.alpha"]
t = np.linspace(0, tf_opt, num_time_steps + 1)

print(f"\nOptimization Results:")
print(f"Optimal time: {tf_opt:.2f} seconds")
print(f"Final altitude: {q[-1, 2]:.0f} m")
print(f"Final velocity: {q[-1, 0]:.1f} m/s")
print(f"Final mass: {q[-1, 4]:.0f} kg")

# Plot results
plot_results(t, q, alpha)
