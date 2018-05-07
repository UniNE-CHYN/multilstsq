import numpy
import sys
import unittest
import ast

from multilstsq import ExprEvaluator, RegrExprEvaluator
from multilstsq.expreval import ast_compare


class TestExprEvaluator(unittest.TestCase):

    def setUp(self):
        pass

    def test_basiceval(self):
        self.assertEqual(ExprEvaluator('a', {'a': 3}).eval(), 3)
        self.assertEqual(ExprEvaluator('a+b', {'a': 3, 'b': 4, }).eval(), 7)

        # Not enough variables to evaluate
        self.assertRaises(ValueError, ExprEvaluator('a').eval)
        self.assertRaises(ValueError, ExprEvaluator('a+b', {'a': 3}).eval)
        self.assertRaises(ValueError, ExprEvaluator('a+b', {'b': 4}).eval)

    def test_substitute_vars(self):
        ee = ExprEvaluator('a+(b*c)')
        self.assertEqual(ee.variables, set(['a', 'b', 'c']))

        ee0 = ee.substitute()
        self.assertEqual(ee0.variables, set(['a', 'b', 'c']))

        ee1 = ee.substitute(None, {'a': 2, })
        self.assertEqual(ee1.variables, set(['b', 'c']))
        self.assertEqual(ee1.constants, set(['a']))

        ee2 = ee.substitute(None, {'b': 3, })
        self.assertEqual(ee2.variables, set(['a', 'c']))
        self.assertEqual(ee2.constants, set(['b']))

        ee3 = ee.substitute(None, {'c': 4, })
        self.assertEqual(ee3.variables, set(['a', 'b']))
        self.assertEqual(ee3.constants, set(['c']))

        ee12 = ee1.substitute(None, {'b': 3, })
        self.assertEqual(ee12.variables, set(['c']))
        self.assertEqual(ee12.constants, set(['a', 'b']))

        ee23 = ee2.substitute(None, {'c': 4, })
        self.assertEqual(ee23.variables, set(['a']))
        self.assertEqual(ee23.constants, set(['b', 'c']))

        ee31 = ee3.substitute(None, {'a': 2, })
        self.assertEqual(ee31.variables, set(['b']))
        self.assertEqual(ee31.constants, set(['a', 'c']))

        ee123a = ee12.substitute(None, {'c': 4, })
        self.assertEqual(ee123a.variables, set([]))
        self.assertEqual(ee123a.constants, set(['a', 'b', 'c']))

        ee123b = ee23.substitute(None, {'a': 2, })
        self.assertEqual(ee123b.variables, set([]))
        self.assertEqual(ee123b.constants, set(['a', 'b', 'c']))

        ee123c = ee31.substitute(None, {'b': 3, })
        self.assertEqual(ee123c.variables, set([]))
        self.assertEqual(ee123c.constants, set(['a', 'b', 'c']))

        self.assertEqual(ee123a.eval(), 14)
        self.assertEqual(ee123b.eval(), 14)
        self.assertEqual(ee123c.eval(), 14)

        self.assertEqual(ee123a.substitute(None, {'a': 10}).eval(), 22)
        self.assertEqual(ee123b.substitute(None, {'b': 4}).eval(), 18)
        self.assertEqual(ee123c.substitute(None, {'c': 10}).eval(), 32)

        eeast = ee.substitute({ast.parse('b*c', mode='eval').body: 'd'})
        self.assertEqual(eeast.variables, set(['a', 'd']))

    def test_reduce(self):
        ee = ExprEvaluator('a+(b*c)')
        self.assertEqual(ee.variables, set(['a', 'b', 'c']))

        ee1r = ee.substitute(None, {'a': 2}).reduce()
        ee2r = ee.substitute(None, {'b': 3}).reduce()
        ee3r = ee.substitute(None, {'c': 4}).reduce()

        ee12r = ee.substitute(None, {'a': 2, 'b': 3}).reduce()
        ee13r = ee.substitute(None, {'a': 2, 'c': 4}).reduce()
        ee23r = ee.substitute(None, {'b': 3, 'c': 4}).reduce()
        ee123r = ee.substitute(None, {'a': 2, 'b': 3, 'c': 4}).reduce()

        self.assertEqual(ee1r.variables, set(['b', 'c']))
        self.assertEqual(ee1r.constants, set(['a']))
        self.assertEqual(ee2r.variables, set(['a', 'c']))
        self.assertEqual(ee2r.constants, set(['b']))
        self.assertEqual(ee3r.variables, set(['a', 'b']))
        self.assertEqual(ee3r.constants, set(['c']))

        self.assertEqual(ee12r.variables, set(['c']))
        self.assertEqual(ee12r.constants, set(['a', 'b']))

        self.assertEqual(ee13r.variables, set(['b']))
        self.assertEqual(ee13r.constants, set(['a', 'c']))

        self.assertEqual(ee23r.variables, set(['a']))
        # Should contain only one constant (b+c -> reduced)
        self.assertEqual(len(ee23r.constants), 1)
        self.assertRegex(list(ee23r.constants)[0], '^\_\_ExprEvaluator\_[0-9]+$')

        self.assertEqual(ee123r.variables, set([]))
        self.assertEqual(len(ee123r.constants), 1)
        self.assertRegex(list(ee123r.constants)[0], '^\_\_ExprEvaluator\_[0-9]+$')

        self.assertEqual(ee123r.eval(), 14)

    def test_substitute_vars_2(self):
        ee = ExprEvaluator('a+(b*c)')
        ee2 = ee.substitute({'a': 'x[0]', 'b': 'x[1]', 'c': 'x[2]'})

        self.assertEqual(ee2.variables, {'x'})
        self.assertEqual(ee2.substitute(None, {'x': [2, 3, 4], }).eval(), 14)

    def test_evaluable(self):
        ee = ExprEvaluator('2+4')
        self.assertEqual(ee.eval(), 6)

    def test_numpy(self):
        ee = ExprEvaluator('numpy.sin(x)')
        self.assertEqual(ee.substitute(None, {'x': [numpy.pi / 2]}).eval(), 1)

    def test_numpy_no_caller_modules(self):
        ee = ExprEvaluator('numpy.sin(x)', enable_caller_modules=False)
        with self.assertRaises(ValueError):
            self.assertEqual(ee.substitute(None, {'x': [numpy.pi / 2]}).eval(), 1)

    def test_ast_compare(self):
        self.assertTrue(ast_compare(ast.parse('a+b', mode='eval'), ast.parse('a+b', mode='eval')))
        self.assertFalse(ast_compare(ast.parse('a+b', mode='eval'), ast.parse('a', mode='eval')))
        self.assertTrue(ast_compare(ast.parse('(1,2,3)', mode='eval'), ast.parse('(1, 2, 3)', mode='eval')))
        self.assertFalse(ast_compare(ast.parse('(1,2,3)', mode='eval'), ast.parse('(1, 3, 2)', mode='eval')))
        self.assertFalse(ast_compare(ast.parse('(1,2,3)', mode='eval'), ast.parse('(1, 2)', mode='eval')))

    def test_substitute_3(self):
        ee = ExprEvaluator('a+(b*c)')
        ee2 = ExprEvaluator('(e*f)')

        eesub1 = ee.substitute({'a': ee2})
        self.assertEqual(eesub1.variables, {'e', 'f', 'b', 'c'})

        eesub2 = ee.substitute({'a': ast.parse('e*f', mode='eval').body})
        self.assertEqual(eesub1.variables, {'e', 'f', 'b', 'c'})

    def test_substitute_4(self):
        ee = ExprEvaluator('a+(b*c)+a*(d*e)')
        ee2 = ee.substitute(None, {'b': 2, 'c': 3})
        ee3 = ee2.substitute(None, {'d': 4, 'e': 5})
        ee3r = ee3.reduce()

        self.assertEqual(ee3r.variables, {'a'})
        with self.assertRaises(ValueError):
            ee3r.eval()

    def test_ExprEvaluator_instantiation(self):
        self.assertEqual(ExprEvaluator(3).eval(), 3)
        self.assertEqual(ExprEvaluator(3.1).eval(), 3.1)
        self.assertEqual(ExprEvaluator('abc').variables, {'abc'})

        with self.assertRaises(TypeError):
            ExprEvaluator({})

    def test_enableCall(self):
        ee = ExprEvaluator('a+(b*c)')
        with self.assertRaises(RuntimeError):
            self.assertEqual(ee(1, 2, 3), 7)

        self.assertEqual(ee(a=1, b=2, c=3), 7)

        ee.enable_call(['a', 'b', 'c'])
        self.assertEqual(ee(1, 2, 3), 7)

    def test_ExprEvaluator_to_string_and_repr(self):
        from ast import BinOp, Name, Load, Add, Mult
        ee = ExprEvaluator('a+(b*c)')
        ee2 = eval(repr(ee))
        self.assertEqual(str(ee), str(ee2))

    def _test_str_parsability(self, str_expr):
        e1 = ExprEvaluator(str_expr)
        e2 = ExprEvaluator(str(e1))

        self.assertEqual(repr(e1), repr(e2))
        self.assertEqual(str(e1), str(e2))

    def test_str_parsability(self):
        self._test_str_parsability('x0+x1+1')
        self._test_str_parsability('("a",["b","c"],({"d"},))')
        self._test_str_parsability('{1: 3}')
        self._test_str_parsability('+2')
        self._test_str_parsability('-2')
        self._test_str_parsability('1+2-3*4/5%6**7//8<<9>>10|11^12&13')
        self._test_str_parsability('a.b(b[2,3:5,...],c[::3],d[:3:])')


class TestRegrExprEvaluator(unittest.TestCase):

    def test_find_coeff_for(self):
        ree = RegrExprEvaluator('b0 + (x0+x1)*b1 + (x0**2)*b2')
        self.assertEqual(ree.find_coeff_for('b0').eval(), 1)
        self.assertEqual(ree.find_coeff_for('b1').substitute(None, {'x0': 4, 'x1': 5}).eval(), 9)
        self.assertEqual(ree.find_coeff_for('b2').substitute(None, {'x0': 4}).eval(), 16)
        self.assertEqual(ree.find_coeff_for('b3').eval(), 0)

        ree = RegrExprEvaluator('numpy.sin(2)')
        self.assertEqual(ree.find_coeff_for('b0').eval(), 0)

    def test_explanatory_variables(self):
        ree = RegrExprEvaluator('b0 + (x0+x1)*b1 + (x0**2)*b2')
        self.assertEqual(ree.parameter_variables, ['b0', 'b1', 'b2'])
        self.assertEqual(ree.explanatory_variables, ['x0', 'x1'])


if __name__ == '__main__':
    unittest.main()
