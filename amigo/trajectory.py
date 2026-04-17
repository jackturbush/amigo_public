import numpy as np
import amigo as am
from abc import abstractmethod


class TrajectorySource(am.Component):
    def __init__(self, state_size: int, inputs: list[dict] = []):
        super().__init__()

        self.add_input("q", shape=state_size)

        for input in inputs:
            name = input["name"]
            self.add_input(
                f"{name}", shape=input.get("shape"), label=input.get("label")
            )

        return


class TrajectoryComponent(am.Component):
    """
    Component wrapper designed for trajectory computations using the trapezoid rule.

    The user defines the `_dynamics` method to implement the dynamics governing equations.
    The `compute` method automatically calls the dynamics based on any specified inputs
    (the state `q` is always assumed).

    Note:
    Can be paired with TrajectorySource either directly by the user or automatically if used
    to create a TrajectoryModel.

    Example Usage:
    ---------
    For a trajectory calculated at 20 points in time and a state vector of dimension 4,
    we also include two extra inputs: angle of attack and throttle. A scaling dictionary
    is also provided in this example.
    ```
    class ExampleDynamics(am.TrajectoryComponent):
        def __init__(self, scaling):
            super().__init__(20, 4, [{"name": "alpha"}, {"name": "throttle"}])

            self.scaling = scaling
            self.add_constant("g", value=9.81)
            return

        def _dynamics(self, q, alpha, throttle):
            g = self.constants["g"]
            mass = q[0] * self.scaling["mass"]
            ...
            return qdot

        def compute(self):
            super().compute()
    ```
    """

    def __init__(
        self, num_time_steps: int, state_size: int, aux_inputs: list[dict] = []
    ):
        """
        Automatically adds inputs for the state variables and final time, as well as
        any additional inputs specified in `aux_inputs`.

        The user can specify constants after calling `super().__init__()`

        Args:
            num_time_steps (int) : Number of evaluation points in the trajectory
            state_size (int) : Size of the state vector at a given point
            aux_inputs : list of auxiliary inputs besides the state vector `q`

        Inputs must be specified as a list of dictionaries, with a dictionary for every
        auxiliary input that contains at least the "name" entry.
        ```
        """
        super().__init__()

        self._input_names = [input["name"] for input in aux_inputs]
        self.num_time_steps = num_time_steps
        self._state_size = state_size
        self._input_list = aux_inputs

        self.add_input("tf", label="final time")
        self.add_input("q1", shape=state_size)
        self.add_input("q2", shape=state_size)

        for input in aux_inputs:
            name = input["name"]
            self.add_input(
                f"{name}1", shape=input.get("shape"), label=input.get("label")
            )
            self.add_input(
                f"{name}2", shape=input.get("shape"), label=input.get("label")
            )

        self.add_constraint("res", shape=state_size)

        return

    @abstractmethod
    def _dynamics(self, q, *args):
        """
        User must implement the dynamics and return qdot.

        Args:
            q : State vector

        Optional additional arguments must be specified in the same order as defined in `aux_inputs`.
        """
        pass

    def compute(self):
        """
        User must define their own `compute` method that calls `super().compute()`
        """
        dt = self.inputs["tf"] / self.num_time_steps
        input1_vars = [self.inputs[f"{name}1"] for name in self._input_names]
        input2_vars = [self.inputs[f"{name}2"] for name in self._input_names]

        q1, q2 = self.inputs["q1"], self.inputs["q2"]

        f1 = self._dynamics(q1, *input1_vars)
        f2 = self._dynamics(q2, *input2_vars)
        self.constraints["res"] = [
            q2[i] - q1[i] - 0.5 * dt * (f1[i] + f2[i]) for i in range(self._state_size)
        ]

        return


class TrajectoryModel:

    def __init__(
        self, num_time_steps: int, state_size: int, aux_inputs: list[dict] = []
    ):
        self.num_time_steps = num_time_steps
        self.state_size = state_size
        self._input_list = aux_inputs
        self._input_names = [input["name"] for input in aux_inputs]

        return

    def create_model(
        self,
        dynamics: TrajectoryComponent,
        module_name: str | None = None,
    ):
        model = am.Model(module_name)

        model.add_component(
            "source",
            self.num_time_steps + 1,
            TrajectorySource(self.state_size, self._input_list),
        )
        model.add_component("kernel", self.num_time_steps, dynamics)

        model.link(f"source.q[:-1,:]", f"kernel.q1")
        model.link(f"source.q[1:,:]", f"kernel.q2")

        for name in self._input_names:
            model.link(f"source.{name}[:-1]", f"kernel.{name}1")
            model.link(f"source.{name}[1:]", f"kernel.{name}2")

        return model

    def link_boundary_conditions(
        self, model: am.Model, traj_model_name: str, ic: str = None, fc: str = None
    ):
        if ic:
            model.link(f"{traj_model_name}.source.q[0,:]", f"{ic}.q[0,:]")
        if fc:
            model.link(
                f"{traj_model_name}.source.q[{self.num_time_steps}, :]", f"{fc}.q[0,:]"
            )


class TrajModel(am.Model):
    def __init__(self, dynamics: TrajectoryComponent, module_name: str | None = None):
        super().__init__(module_name)

        self._num_time_steps = dynamics.num_time_steps
        self._state_size = dynamics._state_size
        input_list = dynamics._input_list
        input_names = dynamics._input_names

        self.add_component(
            "source",
            self._num_time_steps + 1,
            TrajectorySource(self._state_size, input_list),
        )

        self.add_component("kernel", self._num_time_steps, dynamics)

        self.link(f"source.q[:-1,:]", f"kernel.q1")
        self.link(f"source.q[1:,:]", f"kernel.q2")

        for name in input_names:
            self.link(f"source.{name}[:-1]", f"kernel.{name}1")
            self.link(f"source.{name}[1:]", f"kernel.{name}2")

        return

    def link_boundary_conditions(
        self, model, traj_name: str, ic: str = None, fc: str = None
    ):
        if ic:
            model.link(f"{traj_name}.source.q[0,:]", f"{ic}.q[0,:]")
        if fc:
            model.link(
                f"{traj_name}.source.q[{self._num_time_steps}, :]", f"{fc}.q[0,:]"
            )
