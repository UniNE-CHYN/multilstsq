import numpy
import re
import ast

from .expreval import ExprEvaluator
from .lstsq import MultiLstSq

class RegrExprEvaluator(ExprEvaluator):
    """Affine expression in b0, b1... bn (bi = linear regression parameters)

    Explanatory can be specified as x0, x1, ...xn.

    (this is only convention, class should work with other naming)

    General form should be: (coeff0)*b0 + (coeff1)*b1 + (coeff2)*b2 + b3

    coefficient should be in parentheses, or can be omitted if == 1.
    """

    def find_coeff_for(self, variable):
        ret = self._find_coeff_for(variable, self._parsed_expr)
        if ret is None:
            return ExprEvaluator(ast.copy_location(ast.Num(n=0), self._parsed_expr))
        else:
            return ret

    def _find_coeff_for(self, variable, expr):
        if isinstance(expr, ast.Name):
            if expr.id == variable:
                return ExprEvaluator(ast.copy_location(ast.Num(n=1), expr))
        elif isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
            #Add: we're still at the top-level poly
            ret1 = self._find_coeff_for(variable, expr = expr.left)
            ret2 = self._find_coeff_for(variable, expr = expr.right)
            if ret1 is not None:
                assert ret2 is None
                return ret1
            elif ret2 is not None:
                assert ret1 is None
                return ret2
            else:
                #Not found!
                return None
        elif isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Mult):
            if isinstance(expr.left, ast.Name) and expr.left.id == variable:
                return ExprEvaluator(expr.right, self._constants)
            elif isinstance(expr.right, ast.Name) and expr.right.id == variable:
                return ExprEvaluator(expr.left, self._constants)
        else:
            #Unknown expr, ignore
            return None

    @property
    def parameter_variables(self):
        return ['b{0}'.format(b) for b in sorted(int(x[1:]) for x in self.variables if re.match('^b[0-9]+$', x))]

    @property
    def explanatory_variables(self):
        return ['x{0}'.format(x) for x in sorted(int(x[1:]) for x in self.variables if re.match('^x[0-9]+$', x))]


class MultiRegression(MultiLstSq):
    def __init__(self, problem_dimensions, model_str, internal_dtype = numpy.float):
        self._build_expressions(problem_dimensions, model_str)
        if len(self._base_model.parameter_variables) == 0:
            raise ValueError("Model should have at least 1 parameter")
        MultiLstSq.__init__(self, problem_dimensions, len(self._base_model.parameter_variables), internal_dtype)

    def _build_expressions(self, problem_dimensions, model_str):
        self._base_model_str = model_str
        self._base_model = RegrExprEvaluator(model_str)
        if len(self._base_model.explanatory_variables) == 0:
            self._n_explanatory_min = 0
        else:
            self._n_explanatory_min = max(int(x[1:]) for x in self._base_model.explanatory_variables) + 1

        if len(self._base_model.parameter_variables) > 0:
            self._conversion_expression = ExprEvaluator(
                'numpy.concatenate([{0}], axis={1})'.format(','.join('(numpy.ones(o_dim)*({0}))[...,numpy.newaxis]'.format(var) for var in self._base_model.parameter_variables), len(problem_dimensions) + 1)
            )
        else:
            self._conversion_expression = ExprEvaluator(
                'numpy.empty({0}+(X.shape[-2],0))'.format(problem_dimensions)
            )
        self._conversion_expression = self._conversion_expression.substitute(
            dict((var, self._base_model.find_coeff_for(var)) for var in self._base_model.parameter_variables)
        )
        self._conversion_expression = self._conversion_expression.substitute(
            dict((var, 'X[..., :, {0}]'.format(var[1:])) for var in self._base_model.explanatory_variables),
        )

        self._apply_expression = self._base_model.substitute(
            dict((var, "beta[...,{0},:]".format(idx)) for idx, var in enumerate(self._base_model.parameter_variables))
        ).substitute(
            dict((var, 'X[..., :, {0}]'.format(var[1:])) for var in self._base_model.explanatory_variables),
        )

        self._apply_expression = ExprEvaluator('x[...,0]').substitute({'x': self._apply_expression,})

    @property
    def base_model_str(self):
        return self._base_model_str

    @property
    def beta_names(self):
        return self._base_model.parameter_variables

    def __validate_dimensions(self, X=None, y=None, w=None):
        assert w is None or y is not None, "Should provide y if providing w"
        assert y is None or X is not None, "Should provide X if providing y"

        if X is not None and (X.ndim != len(self._problem_dimensions) + 2 or X.shape[:-2] != self._problem_dimensions or X.shape[-1] < self._n_explanatory_min):
            X_dim_str = ', '.join([str(d) for d in X.shape])
            dim_str = ', '.join([str(d) for d in self._problem_dimensions]+['<n>', '>=' + str(self._n_explanatory_min)])

            raise ValueError('Wrong dimensions for X ({} instead of {})'.format(X_dim_str, dim_str))

        if y is not None and (y.shape != self._problem_dimensions + (X.shape[-2], 1)):
            y_dim_str = ', '.join([str(d) for d in y.shape])
            dim_str = ', '.join([str(d) for d in self._problem_dimensions + (X.shape[-2], 1)])

            raise ValueError('Wrong dimensions for y ({} instead of {})'.format(y_dim_str, dim_str))

        if w is not None and (w.shape != y.shape):
            w_dim_str = ', '.join([str(d) for d in w.shape])
            dim_str = ', '.join([str(d) for d in y.shape])
            raise ValueError('Wrong dimensions for w ({} instead of {})'.format(w_dim_str, dim_str))


    def add_data(self, X, y, w = None):
        self.__validate_dimensions(X, y, w)

        X_new = self._conversion_expression.substitute(None, {'o_dim': X.shape[:-1], 'X': X,}).eval()

        X_mask = None
        y_mask = None
        if isinstance(X, numpy.ma.MaskedArray) and X.mask.shape == X.shape:
            X_mask = numpy.any(X.mask, axis = len(self._problem_dimensions) + 1)
            X_new = X_new.data.copy()
            X_new[X_mask] = 0
        if isinstance(y, numpy.ma.MaskedArray) and y.mask.shape == y.shape:
            y_mask = numpy.any(y.mask, axis = len(self._problem_dimensions) + 1)
            y = y.data.copy()
            y[y_mask] = 0

        if X_mask is not None or y_mask is not None:
            #generate valid
            if X_mask is not None and y_mask is not None:
                valid = ~numpy.logical_or(X_mask, y_mask)
            elif X_mask is not None:
                valid = ~X_mask
            elif y_mask is not None:
                valid = ~y_mask

            X_valid = numpy.repeat(valid[..., :, numpy.newaxis], X_new.shape[-1], len(self._problem_dimensions) + 1)
            validb = valid[..., :, numpy.newaxis]

            return super().add_data(X_new * X_valid, y * validb, w)
        else:
            return super().add_data(X_new, y, w)

    def get_expr_for_idx(self, pb_idx):
        assert type(pb_idx) == tuple
        expr = self._base_model.substitute(None,
            dict((var, self.beta[pb_idx+(varidx,0)]) for varidx, var in enumerate(self._base_model.parameter_variables))
        ).reduce()
        expr.enable_call(['x{0}'.format(x) for x in range(self._n_explanatory_min + 1)])
        return expr


    @property
    def apply_expr(self):
        return self._apply_expression.substitute(None, {'beta':self.beta})

    def __getstate__(self):
        d = super().__getstate__()
        d['base_model_str'] = self._base_model_str
        return d

    def __setstate__(self, newstate):
        self._build_expressions(newstate['problem_dimensions'], newstate['base_model_str'])
        return super().__setstate__(newstate)

