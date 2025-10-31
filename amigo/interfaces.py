from __future__ import annotations
from typing import TYPE_CHECKING, Any
from .model import Model, ModelVector
from .optimizer import Optimizer
import numpy as np


if TYPE_CHECKING:  # type-checkers see it; users don't need runtime package
    import openmdao.api as om


def _import_openmdao():
    try:
        import openmdao.api as om

        return om
    except Exception as exc:
        raise ImportError(
            "OpenMDAO integration requires the 'openmdao' extra. Install with: "
            "pip install 'amigo[openmdao]'"
        ) from exc


def ExternalOpenMDAOComponent(om_problem, am_model):
    om = _import_openmdao()

    class _ExternalOpenMDAOComponent:
        def __init__(self, om_problem: om.Problem, am_model: Model):

            self.om_problem = om_problem
            self.am_model = am_model

            # Get the OpenMDAO names for the inputs
            self.obj = []
            self.cons = []
            self.dvs = []

            self.nvars = 0
            self.dv_mapping = {}
            self.ncon = 0
            self.con_mapping = {}

            for name in self.om_problem.model.get_objectives():
                self.obj.append(name)
            for name, meta in self.om_problem.model.get_constraints().items():
                self.cons.append(name)
                size = meta["size"]
                self.con_mapping[name] = np.arange(self.ncon, self.ncon + size)
                self.ncon += size
            for name, meta in self.om_problem.model.get_design_vars().items():
                self.dvs.append(name)
                size = meta["size"]
                self.dv_mapping[name] = np.arange(self.nvars, self.nvars + size)
                self.nvars += size

            self.rowp = np.arange(0, self.ncon * self.nvars + 1, self.nvars, dtype=int)
            self.cols = np.arange(0, self.ncon * self.nvars, dtype=int) % self.nvars

            self.jac_mapping = {}
            for con in self.cons:
                rows = self.con_mapping[con]
                for dv in self.dvs:
                    cols = self.dv_mapping[dv]

                    indices = np.zeros((len(rows), len(cols)))
                    for i in range(len(rows)):
                        indices[i, :] = rows[i] + np.arange(len(cols), dtype=int)

                    self.jac_mapping[(con, dv)] = indices.flatten()

            return

        def get_constraint_jacobian_csr(self):
            return self.ncon, self.nvars, self.rowp, self.cols

        def evaluate(self, x, con, grad, jac):
            # Set the design variables into the OpenMDAO model
            for name in self.dvs:
                self.om_problem.set_val(name, x[self.dv_mapping[name]])

            # Run the model
            self.om_problem.run_model()

            # Compute the objective and the objective gradient
            fobj = 0
            if len(self.obj) > 0:
                fobj = self.om_problem.get_val(self.obj[0])
                dfdx = self.om_problem.compute_totals(of=self.obj[0], wrt=self.dvs)

                for name in self.dvs:
                    grad[self.dv_mapping[name]] = dfdx[self.obj[0], name]

            # Extract the constraint values and constraint Jacobian
            if len(self.cons) > 0:
                for name in self.cons:
                    con[self.con_mapping[name]] = self.om_problem.get_val(name)
                dcdx = self.om_problem.compute_totals(of=self.cons, wrt=self.dvs)

                for of, wrt in dcdx:
                    jac[self.jac_mapping[(of, wrt)]] = dcdx[of, wrt]

            return fobj

    return _ExternalOpenMDAOComponent(om_problem, am_model)


def ExplicitOpenMDAOPostOptComponent(**kwargs):
    om = _import_openmdao()

    class _ExplicitOpenMDAOPostOptComponent(om.ExplicitComponent):
        def initialize(self):
            self.options.declare("data", types=list)
            self.options.declare("output", types=list)
            self.options.declare("model", types=Model)
            self.options.declare("x", types=ModelVector)
            self.options.declare("lower", types=ModelVector)
            self.options.declare("upper", types=ModelVector)
            self.options.declare("opt_options", default={}, types=dict)

        def _map_names(self, names):
            mapping = {}
            for name in names:
                sanitized_name = name.replace(".", "_")
                mapping[name] = sanitized_name

            return mapping

        def setup(self):
            self.data = self.options["data"]
            self.output = self.options["output"]
            self.model = self.options["model"]
            self.x = self.options["x"]
            self.lower = self.options["lower"]
            self.upper = self.options["upper"]
            self.opt_options = self.options["opt_options"]

            self.opt = Optimizer(self.model, self.x, lower=self.lower, upper=self.upper)

            self.data_mapping = self._map_names(self.data)
            self.out_mapping = self._map_names(self.output)

            for name in self.data:
                meta = self.model.get_meta(name)
                open_name = self.data_mapping[name]
                self.add_input(open_name, val=meta["value"])

            for name in self.output:
                open_name = self.out_mapping[name]
                self.add_output(open_name)  # , val=meta["value"])

            self.declare_partials(of="*", wrt="*")

            return

        def compute(self, inputs, outputs):
            data = self.model.get_data_vector()
            for name in self.data:
                open_name = self.data_mapping[name]
                data[name] = inputs[open_name]

            self.opt.optimize(self.opt_options)

            out = self.opt.compute_output()
            for name in self.output:
                open_name = self.out_mapping[name]
                outputs[open_name] = out[name]

            self.dfdx, self.of_map, self.wrt_map = (
                self.opt.compute_post_opt_derivatives(of=self.output, wrt=self.data)
            )

        def compute_partials(self, inputs, partials):
            for of in self.of_map:
                open_of = self.out_mapping[of]
                for wrt in self.wrt_map:
                    open_wrt = self.data_mapping[wrt]
                    partials[open_of, open_wrt] = self.dfdx[
                        self.of_map[of], self.wrt_map[wrt]
                    ]

    return _ExplicitOpenMDAOPostOptComponent(**kwargs)
