import amigo as am
import numpy as np
import sys
import matplotlib.pylab as plt
import niceplots
import argparse
import json


"""
Min Time to Climb 
==============================

This is example from Betts' book "Practical Methods for Optimal Control
Using NonlinearProgramming", 3rd edition, Chapter 10: Test Problems.

This is a scaled problem with a single component dynamics code
"""

# Problem parameters
num_time_steps = 100

# Scaling factors for state variables
SCALING_FACTORS = {
    "v": 100.0,
    "gamma": 10.0,
    "h": 10000.0,  # altitude [m]
    "r": 10000.0,  # range [m]
    "m": 10000.0,  # mass [kg]
}


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

        # Declare physical constants
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

        # Declare inputs
        self.add_input("alpha", label="angle of attack (degrees)")
        self.add_input("q", shape=5, label="scaled state variables")
        self.add_input("qdot", shape=5, label="scaled state derivatives")

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

        # Get scaled inputs
        alpha = self.inputs["alpha"]
        q_scaled = self.inputs["q"]
        qdot_scaled = self.inputs["qdot"]

        # Convert scaled state variables back to physical units
        v = q_scaled[0] * SCALING_FACTORS["v"]  # velocity [m/s]
        gamma = q_scaled[1] * SCALING_FACTORS["gamma"]  # flight path angle [degrees]
        h = q_scaled[2] * SCALING_FACTORS["h"]  # altitude [m]
        r = q_scaled[3] * SCALING_FACTORS["r"]  # range [m]
        m = q_scaled[4] * SCALING_FACTORS["m"]  # mass [kg]

        # Atmospheric properties (simplified)
        T_atm = 288.15  # K
        rho = 1.225  # kg/m^3

        # Aerodynamics
        CL = CL_alpha * conv * alpha  # Convert alpha from degrees to radians

        # Drag coefficient (simplified - no compressibility)
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

        # Aircraft dynamics equations (in physical units)
        dvdt = (T / m) * am.cos(alpha_rad) - (D / m) - g * am.sin(gamma_rad)
        dgammadt = (
            T / (m * v) * am.sin(alpha_rad) + L / (m * v) - (g / v) * am.cos(gamma_rad)
        ) / conv
        dhdt = v * am.sin(gamma_rad)
        drdt = v * am.cos(gamma_rad)
        dmdt = -T / (g * Isp)

        # Convert derivatives to scaled form
        dvdt_scaled = dvdt / SCALING_FACTORS["v"]
        dgammadt_scaled = dgammadt / SCALING_FACTORS["gamma"]
        dhdt_scaled = dhdt / SCALING_FACTORS["h"]
        drdt_scaled = drdt / SCALING_FACTORS["r"]
        dmdt_scaled = dmdt / SCALING_FACTORS["m"]

        # Residuals using scaled derivatives
        res = [
            qdot_scaled[0] - dvdt_scaled,
            qdot_scaled[1] - dgammadt_scaled,
            qdot_scaled[2] - dhdt_scaled,
            qdot_scaled[3] - drdt_scaled,
            qdot_scaled[4] - dmdt_scaled,
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
        q_scaled = self.inputs["q"]

        # Initial conditions in physical units
        # [v0, gamma0, h0, r0, m0] = [136.0, 0.0, 100.0, 0.0, 19030.0]

        # Convert to scaled form
        v0_scaled = 136.0 / SCALING_FACTORS["v"]
        gamma0_scaled = 0.0 / SCALING_FACTORS["gamma"]
        h0_scaled = 100.0 / SCALING_FACTORS["h"]
        r0_scaled = 0.0 / SCALING_FACTORS["r"]
        m0_scaled = 19030.0 / SCALING_FACTORS["m"]

        self.constraints["res"] = [
            q_scaled[0] - v0_scaled,
            q_scaled[1] - gamma0_scaled,
            q_scaled[2] - h0_scaled,
            q_scaled[3] - r0_scaled,
            q_scaled[4] - m0_scaled,
        ]


class FinalConditions(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("q", shape=5)
        self.add_constraint("res", shape=3)

    def compute(self):
        q_scaled = self.inputs["q"]

        # Final conditions in physical units
        hf = 20000.0
        vf = 340.0
        gama_f = 0.0

        # Convert to scaled form
        vf_scaled = vf / SCALING_FACTORS["v"]
        gama_f_scaled = gama_f / SCALING_FACTORS["gamma"]
        hf_scaled = hf / SCALING_FACTORS["h"]

        self.constraints["res"] = [
            q_scaled[0] - vf_scaled,
            q_scaled[1] - gama_f_scaled,
            q_scaled[2] - hf_scaled,
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
    model.add_component("obj", 1, obj)
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
    model.link(f"ac.q[{num_time_steps}, :]", "fc.q[0, :]")

    return model


def plot_results(t, q_scaled, alpha):
    """Plot the optimization results"""

    # Convert scaled states back to physical units for plotting
    q = np.zeros_like(q_scaled)
    q[:, 0] = q_scaled[:, 0] * SCALING_FACTORS["v"]  # velocity
    q[:, 1] = q_scaled[:, 1] * SCALING_FACTORS["gamma"]  # flight path angle
    q[:, 2] = q_scaled[:, 2] * SCALING_FACTORS["h"]  # altitude
    q[:, 3] = q_scaled[:, 3] * SCALING_FACTORS["r"]  # range
    q[:, 4] = q_scaled[:, 4] * SCALING_FACTORS["m"]  # mass

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
tf_guess = 200.0
x["obj.tf"] = tf_guess

# Set initial guess for scaled states
t_guess = np.linspace(0, tf_guess, num_time_steps + 1)

# Create a very conservative, physics-based initial guess that respects bounds
tau = t_guess / tf_guess  # normalized time [0, 1]

# Velocity profile - linear interpolation
v_guess = 136.0 + (340.0 - 136.0) * tau  # simple linear interpolation

# Flight path angle:
gamma_guess = 5.0 * np.sin(np.pi * t_guess / tf_guess)  # flight path angle

# Altitude:
h_guess = 100.0 + (20000.0 - 100.0) * t_guess / tf_guess  # altitude

# Range:
r_guess = 5000.0 * t_guess  # range

# Mass:
m_guess = 19030.0 - 200.0 * t_guess / tf_guess  # mass decrease

# Convert to scaled form
x["ac.q[:, 0]"] = v_guess / SCALING_FACTORS["v"]  # scaled velocity
x["ac.q[:, 1]"] = gamma_guess / SCALING_FACTORS["gamma"]  # scaled flight path angle
x["ac.q[:, 2]"] = h_guess / SCALING_FACTORS["h"]  # scaled altitude
x["ac.q[:, 3]"] = r_guess / SCALING_FACTORS["r"]  # scaled range
x["ac.q[:, 4]"] = m_guess / SCALING_FACTORS["m"]  # scaled mass

# Control AOA guess
x["ac.alpha"] = 5.0 * np.exp(-2 * tau) + 1.0  # starts at 6 deg, goes to 1 deg

# Set up bounds (in scaled form)
lower = model.create_vector()
upper = model.create_vector()

# Final time bounds
lower["obj.tf"] = 100.0
upper["obj.tf"] = 1000.0

# Control bounds
lower["ac.alpha"] = -8.0
upper["ac.alpha"] = 10.0

# Scaled state bounds
lower["ac.q[:, 0]"] = 10.0 / SCALING_FACTORS["v"]  # minimum velocity
upper["ac.q[:, 0]"] = 500.0 / SCALING_FACTORS["v"]  # maximum velocity
lower["ac.q[:, 1]"] = -85.0 / SCALING_FACTORS["gamma"]  # flight path angle
upper["ac.q[:, 1]"] = 85.0 / SCALING_FACTORS["gamma"]
lower["ac.q[:, 2]"] = 0.0 / SCALING_FACTORS["h"]  # altitude
upper["ac.q[:, 2]"] = 25000.0 / SCALING_FACTORS["h"]
# lower["ac.q[:, 3]"] = 0.0 / SCALING_FACTORS["r"]  # range
lower["ac.q[:, 4]"] = 10.0 / SCALING_FACTORS["m"]  # minimum mass
upper["ac.q[:, 4]"] = 20000.0 / SCALING_FACTORS["m"]  # maximum mass


# Optimize
opt = am.Optimizer(model, x, lower=lower, upper=upper)

# Add diagnostics before optimization
print("Check the bounds violations\n")

print(f"Initial guess ranges:")
print(f"  Final time: {x['obj.tf'][0]:.2f} s")
print(
    f"  Velocity: {x['ac.q[:, 0]'].min():.3f} to {x['ac.q[:, 0]'].max():.3f} (scaled)"
)
print(f"  Gamma: {x['ac.q[:, 1]'].min():.3f} to {x['ac.q[:, 1]'].max():.3f} (scaled)")
print(
    f"  Altitude: {x['ac.q[:, 2]'].min():.3f} to {x['ac.q[:, 2]'].max():.3f} (scaled)"
)
print(f"  Range: {x['ac.q[:, 3]'].min():.3f} to {x['ac.q[:, 3]'].max():.3f} (scaled)")
print(f"  Mass: {x['ac.q[:, 4]'].min():.3f} to {x['ac.q[:, 4]'].max():.3f} (scaled)")
print(f"  Alpha: {x['ac.alpha'].min():.3f} to {x['ac.alpha'].max():.3f} deg")

# Check bounds violations
lower_viol = np.sum(np.maximum(0, lower._x.get_array() - x._x.get_array()))
upper_viol = np.sum(np.maximum(0, x._x.get_array() - upper._x.get_array()))

print(f"\nInitial bound violations:")
print(f"  Lower bound violation: {lower_viol:.3e}")
print(f"  Upper bound violation: {upper_viol:.3e}")

# Check physical values
q_physical = np.zeros_like(x["ac.q"])
q_physical[:, 0] = x["ac.q"][:, 0] * SCALING_FACTORS["v"]
q_physical[:, 1] = x["ac.q"][:, 1] * SCALING_FACTORS["gamma"]
q_physical[:, 2] = x["ac.q"][:, 2] * SCALING_FACTORS["h"]
q_physical[:, 3] = x["ac.q"][:, 3] * SCALING_FACTORS["r"]
q_physical[:, 4] = x["ac.q"][:, 4] * SCALING_FACTORS["m"]

print(f"\nPhysical values:")
print(f"  Velocity: {q_physical[0, 0]:.1f} to {q_physical[-1, 0]:.1f} m/s")
print(f"  Gamma: {q_physical[0, 1]:.1f} to {q_physical[-1, 1]:.1f} deg")
print(f"  Altitude: {q_physical[0, 2]:.0f} to {q_physical[-1, 2]:.0f} m")
print(f"  Mass: {q_physical[0, 4]:.0f} to {q_physical[-1, 4]:.0f} kg")

print("END DIAGNOSTIC OUTPUT\n")

optimizer_options = {"max_iterations": 500}

data = opt.optimize(optimizer_options)

# Save optimization data
with open("time_to_climb_opt_data.json", "w") as fp:
    json.dump(data, fp, indent=2)

# Extract results
tf_opt = x["obj.tf"][0]  # Extract scalar from array
q_scaled = x["ac.q"]
alpha = x["ac.alpha"]
t = np.linspace(0, tf_opt, num_time_steps + 1)

# Convert back to physical units for reporting
q_physical = np.zeros_like(q_scaled)
q_physical[:, 0] = q_scaled[:, 0] * SCALING_FACTORS["v"]
q_physical[:, 1] = q_scaled[:, 1] * SCALING_FACTORS["gamma"]
q_physical[:, 2] = q_scaled[:, 2] * SCALING_FACTORS["h"]
q_physical[:, 3] = q_scaled[:, 3] * SCALING_FACTORS["r"]
q_physical[:, 4] = q_scaled[:, 4] * SCALING_FACTORS["m"]

print(f"\nOptimization Results:")
print(f"Optimal time: {tf_opt:.2f} seconds")
print(f"Final altitude: {q_physical[-1, 2]:.0f} m")
print(f"Final velocity: {q_physical[-1, 0]:.1f} m/s")
print(f"Final mass: {q_physical[-1, 4]:.0f} kg")

# Plot results using scaled states
plot_results(t, q_scaled, alpha)
