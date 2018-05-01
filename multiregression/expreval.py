import ast
import copy
import re
import inspect
import types
import itertools

def ast_print(x): # pragma: no cover
    if x is None:
        print(None)
    else:
        print(ast.dump(x))
        
def ast_compare(node1, node2):
    if type(node1) is not type(node2):
        return False
    if isinstance(node1, ast.AST):
        for k, v in vars(node1).items():
            if k in ('lineno', 'col_offset', 'ctx'):
                continue
            if not ast_compare(v, getattr(node2, k)):
                return False
        return True
    elif isinstance(node1, list):
        if len(node1) != len(node2):
            return False
        return all(ast_compare(n1, n2) for n1, n2 in zip(node1, node2))
    else:
        return node1 == node2
        
class _SubstituteTransformer(ast.NodeTransformer):
    """
    Transformer to replace a variable name by an expression.
    """
    def __init__(self, substitution_dict):
        """
        Initialize the tranformer.
        
        :param substitution_dict: Dictionnary whose key are the variables names, and values what the variable should be substituted by.
        
        The values can be strings (in which case they are parse by :func:`ast.parse`), :class:`ExprEvaluator` instances, or :class:`ast.AST`.
        """
        assert type(substitution_dict) == dict
        
        self._substitution_dict = {}
        for variable_name, substitute_by in substitution_dict.items():
            if type(substitute_by) == str:
                self._substitution_dict[variable_name] = ast.parse(substitute_by, mode = 'eval').body
            elif isinstance(substitute_by, ExprEvaluator):
                self._substitution_dict[variable_name] = substitute_by._parsed_expr
            else:
                self._substitution_dict[variable_name] = substitute_by
                
    def visit_Name(self, node):
        """Visitor for the Name type, which should be substituted if node.id is in the substitution dictionnary."""
        if node.id in self._substitution_dict:
            return ast.copy_location(self._substitution_dict[node.id], node)
        else:
            return node
        
class _ReduceTransformer(ast.NodeTransformer):
    """
    Transformer to do one step of the reduction of an :class:`ast.AST` expression.
    
    For example, if the expression is ``x+y*sin(x**2)+(2+2)``, with ``{'x':3}`` as constant,
    the result is ``__ExprEvaluator_0+y*__ExprEvaluator_1+__ExprEvaluator_2``, with constants
    ``{'__ExprEvaluator_0': 3, '__ExprEvaluator_1': 0.4, '__ExprEvaluator_2': 4}``.
    """
    def __init__(self, constants):
        """
        Initialize the tranformer.
    
        :param constants: Dictionnary whose key are the variables names, and values are Python values.
        """        
        self.constants = constants.copy()
        self._did_something = False
        
    @property
    def did_something(self):
        """Return True if the expression has been changed."""
        return self._did_something
    
    @did_something.setter
    def did_something(self, newvalue):
        assert newvalue in (True, False)
        self._did_something = newvalue
        
    def generic_visit(self, node):
        #No change if the node is not an expression or if it's a Name
        if not isinstance(node, ast.expr) or isinstance(node, ast.Name):
            return node
        
        #Get the expression and try to evaluate it
        expr = ast.copy_location(ast.Expression(body = node), node)
        try:
            ret = eval(compile(expr, '<eval>', mode = 'eval'), globals(), self.constants)
        except NameError as e:
            #If we have an undefined name, get a step deeper in the tree
            return super().generic_visit(node)
            
        #find a free constant index
        i = 0
        while '__ExprEvaluator_{0}'.format(i) in self.constants:
            i += 1
            
        const_name = '__ExprEvaluator_{0}'.format(i)
            
        #Do the substitution
        self.constants[const_name] = ret
        
        self._did_something = True
        return ast.copy_location(ast.Name(id = const_name, ctx = ast.Load()), node)
            
class _ListNames(ast.NodeVisitor):
    """Visitor to get all names of an :class:`ast.expr`"""
    def __init__(self):
        self._names = set()
        
    @property
    def names(self):
        return self._names

    def visit_Name(self, node):
        self._names.add(node.id)
        return super().generic_visit(node)
    
        

class ExprEvaluator:
    #__ExprEvaluator_i where is is an int are reserved names in expr
    
    def __init__(self, expr, constants = None, enable_caller_modules=True):
        if constants is None:
            constants = {}
            
        self._caller_modules = {}
        if enable_caller_modules:
            frame = inspect.currentframe()
            while frame.f_globals['__name__'].startswith('.'.join(__name__.split('.')[:-1])):
                frame = frame.f_back
            
            for k, v in itertools.chain(frame.f_locals.items(), frame.f_globals.items()):
                if k not in self._caller_modules and isinstance(v, types.ModuleType):
                    self._caller_modules[k] = v
        
        self._constants = constants.copy()
        if type(expr) == str:
            self._expr = expr
            self._parsed_expr = ast.parse(expr, mode = 'eval').body
        elif isinstance(expr, ast.expr):
            self._expr = '<ast.Expr>'
            self._parsed_expr = expr
        else:
            raise ValueError("Type {0} is not handled for expr".format(type(expr)))
        
        #Remove unneeded constants
        for k in set(self._constants.keys()).difference(self.constants):
            del self._constants[k]
            
        self._call_args = None
        
    def substitute(self, substitution_dict = None, constants = None):
        if substitution_dict is not None:
            new_expr = _SubstituteTransformer(substitution_dict).visit(copy.deepcopy(self._parsed_expr))
        else:
            new_expr = self._parsed_expr
        
        if constants is not None:
            new_constants = self._constants.copy()
            for k, v in constants.items():
                new_constants[k] = v
        else:
            new_constants = self._constants
            
        return ExprEvaluator(new_expr, new_constants)
    
    @property
    def constants(self):
        ln = _ListNames()
        ln.visit(self._parsed_expr)
        return ln.names.intersection(self._constants.keys())
    
    @property
    def variables(self):
        ln = _ListNames()
        ln.visit(self._parsed_expr)
        return ln.names.difference(self._constants.keys())
    
        
    def reduce(self):
        constants_and_modules = self._caller_modules.copy()
        constants_and_modules.update(self._constants)
        rt = _ReduceTransformer(constants_and_modules)
        new_expr = copy.deepcopy(self._parsed_expr)
        reduce_more = True
        while reduce_more:
            new_expr = rt.visit(new_expr)
            reduce_more = rt.did_something
            rt.did_something = False
            
        new_constants = rt.constants
        
        return ExprEvaluator(new_expr, new_constants)
    
    def eval(self):
        reduced_expr = self.reduce()
        if isinstance(reduced_expr._parsed_expr, ast.Name) and reduced_expr._parsed_expr.id in reduced_expr._constants:
            return reduced_expr._constants[reduced_expr._parsed_expr.id]
        else:
            #Not fully reductible, so cannot be evaluated
            remaining_variables = reduced_expr.variables
            raise ValueError("Cannot fully reduce expression, remaining variables: {0}".format(', '.join(remaining_variables)))
        
    def __repr__(self):
        return 'ExprEvaluator({0}, {1!r})'.format(ast.dump(self._parsed_expr), self._constants)
    
    def enable_call(self, variable_list):
        self._call_args = variable_list
        
    def __call__(self, *args):
        if self._call_args is None:
            raise ValueError("enable_call should be called first!")
        return self.substitute(None, dict(zip(self._call_args, args))).eval()
    
    def to_string(self):
        #FIXME: use external library...
        return self._to_string(self._parsed_expr)
    
    def _to_string(self, node):
        if isinstance(node, (ast.Str, ast.Bytes)):
            return repr(node.s)
        elif isinstance(node, ast.Num):
            return repr(node.n)
        elif isinstance(node, ast.Tuple):
            if len(node.elts) == 1:
                return '(' + self._to_string(node.elts[0]) + ', )'
            return '(' + ', '.join(self._to_string(n) for n in node.elts) + ')'
        elif isinstance(node, ast.List):
            return '[' + ', '.join(self._to_string(n) for n in node.elts) + ']'
        elif isinstance(node, ast.Set):
            return '{' + ', '.join(self._to_string(n) for n in node.elts) + '}'
        elif isinstance(node, ast.Dict):
            return '{' + ', '.join(self._to_string(k) + ': ' + self._to_string(v) for k, v in zip(node.keys, node.values)) + '}'
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.UAdd):
                return '+(' + self._to_string(node.operand) + ')'
            elif isinstance(node.op, ast.USub):
                return '-(' + self._to_string(node.operand) + ')'
        elif isinstance(node, ast.BinOp):
            left = self._to_string(node.left)
            right = self._to_string(node.right)
            if isinstance(node.op, ast.Add):
                op = '+'
            elif isinstance(node.op, ast.Sub):
                op = '-'
            elif isinstance(node.op, ast.Mult):
                op = '*'
            elif isinstance(node.op, ast.Div):
                op = '/'
            elif isinstance(node.op, ast.Mod):
                op = '%'
            elif isinstance(node.op, ast.Pow):
                op = '**'
            elif isinstance(node.op, ast.FloorDiv):
                op = '//'
            #TODO: LShift | RShift | BitOr | BitXor | BitAnd | FloorDiv
            return '({0}) {1} ({2})'.format(left, op, right)
        elif isinstance(node, ast.Call):
            args = []
            if node.args is not None:
                args += [self._to_string(n) for n in node.args]

            return '({0})({1})'.format(self._to_string(node.func), ','.join(args))
        elif isinstance(node, ast.Subscript):
            return '({0})[{1}]'.format(self._to_string(node.value), self._to_string(node.slice))
        elif isinstance(node, ast.Attribute):
            return '({0}).{1}'.format(self._to_string(node.value), node.attr)
        elif isinstance(node, ast.Index):
            return self._to_string(node.value)
        elif isinstance(node, ast.Ellipsis):
            return '...'
        elif isinstance(node, ast.ExtSlice):
            return ', '.join(self._to_string(n) for n in node.dims)
        elif isinstance(node, ast.Slice):
            if node.lower is None and node.upper is None and node.step is None:
                return ':'
        
        raise ValueError('malformed node or string: ' + ast.dump(node))
        
        
        
        

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
    
