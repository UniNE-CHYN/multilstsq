import numpy
import sys
import unittest

from multiregression import ExprEvaluator, RegrExprEvaluator

class TestExprEvaluator(unittest.TestCase):

    def setUp(self):
        pass
    
    def test_basiceval(self):
        self.assertEqual(ExprEvaluator('a', {'a': 3}).eval(), 3)
        self.assertEqual(ExprEvaluator('a+b', {'a': 3, 'b': 4,}).eval(), 7)
        
        #Not enough variables to evaluate
        self.assertRaises(ValueError, ExprEvaluator('a').eval)
        self.assertRaises(ValueError, ExprEvaluator('a+b', {'a': 3}).eval)
        self.assertRaises(ValueError, ExprEvaluator('a+b', {'b': 4}).eval)
        
    def test_substitute_vars(self):
        ee = ExprEvaluator('a+(b*c)')
        self.assertEqual(ee.variables, set(['a', 'b', 'c']))
        
        ee1 = ee.substitute(None, {'a': 2,})
        self.assertEqual(ee1.variables, set(['b', 'c']))
        self.assertEqual(ee1.constants, set(['a']))
        
        ee2 = ee.substitute(None, {'b': 3,})
        self.assertEqual(ee2.variables, set(['a', 'c']))
        self.assertEqual(ee2.constants, set(['b']))
        
        ee3 = ee.substitute(None, {'c': 4,})
        self.assertEqual(ee3.variables, set(['a', 'b']))
        self.assertEqual(ee3.constants, set(['c']))
        
        ee12 = ee1.substitute(None, {'b': 3,})
        self.assertEqual(ee12.variables, set(['c']))
        self.assertEqual(ee12.constants, set(['a', 'b']))
        
        ee23 = ee2.substitute(None, {'c': 4,})
        self.assertEqual(ee23.variables, set(['a']))
        self.assertEqual(ee23.constants, set(['b', 'c']))
        
        ee31 = ee3.substitute(None, {'a': 2,})
        self.assertEqual(ee31.variables, set(['b']))
        self.assertEqual(ee31.constants, set(['a', 'c']))
        
        ee123a = ee12.substitute(None, {'c': 4,})
        self.assertEqual(ee123a.variables, set([]))
        self.assertEqual(ee123a.constants, set(['a', 'b', 'c']))
        
        ee123b = ee23.substitute(None, {'a': 2,})
        self.assertEqual(ee123b.variables, set([]))
        self.assertEqual(ee123b.constants, set(['a', 'b', 'c']))
        
        ee123c = ee31.substitute(None, {'b': 3,})
        self.assertEqual(ee123c.variables, set([]))
        self.assertEqual(ee123c.constants, set(['a', 'b', 'c']))
        
        self.assertEqual(ee123a.eval(), 14)
        self.assertEqual(ee123b.eval(), 14)
        self.assertEqual(ee123c.eval(), 14)
        
        self.assertEqual(ee123a.substitute(None, {'a': 10}).eval(), 22)
        self.assertEqual(ee123b.substitute(None, {'b': 4}).eval(), 18)
        self.assertEqual(ee123c.substitute(None, {'c': 10}).eval(), 32)
        
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
        #Should contain only one constant (b+c -> reduced)
        self.assertEqual(len(ee23r.constants), 1)
        self.assertRegex(list(ee23r.constants)[0], '^\_\_ExprEvaluator\_[0-9]+$')
        
        self.assertEqual(ee123r.variables, set([]))
        self.assertEqual(len(ee123r.constants), 1)
        self.assertRegex(list(ee123r.constants)[0], '^\_\_ExprEvaluator\_[0-9]+$')
        
        self.assertEqual(ee123r.eval(), 14)
        
    def test_substitute_vars(self):
        ee = ExprEvaluator('a+(b*c)')
        ee2 = ee.substitute({'a': 'x[0]','b': 'x[1]', 'c': 'x[2]'})
        
        self.assertEqual(ee2.variables, {'x'})
        self.assertEqual(ee2.substitute(None, {'x': [2, 3, 4],}).eval(), 14)
        
    def test_evaluable(self):
        ee = ExprEvaluator('2+4')
        self.assertEqual(ee.eval(), 6)
        
    def test_numpy(self):
        ee = ExprEvaluator('numpy.sin(x)')
        self.assertEqual(ee.substitute(None, {'x': [numpy.pi / 2]}).eval(), 1)
        
        
class TestRegrExprEvaluator(unittest.TestCase):
    def test_find_coeff_for(self):
        ree = RegrExprEvaluator('b0 + (x0+x1)*b1 + (x0**2)*b2')
        self.assertEqual(ree.find_coeff_for('b0').eval(), 1)
        self.assertEqual(ree.find_coeff_for('b1').substitute(None, {'x0': 4, 'x1': 5}).eval(), 9)
        self.assertEqual(ree.find_coeff_for('b2').substitute(None, {'x0': 4}).eval(), 16)
    
    def test_explanatory_variables(self):
        ree = RegrExprEvaluator('b0 + (x0+x1)*b1 + (x0**2)*b2')
        self.assertEqual(ree.parameter_variables, ['b0', 'b1', 'b2'])
        self.assertEqual(ree.explanatory_variables, ['x0', 'x1'])
        

        

    



if __name__ == '__main__':
    unittest.main()


