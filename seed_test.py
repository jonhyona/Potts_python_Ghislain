import numpy.random as rd


def test_internal_naive(seed):
    rd.seed(seed)
    print('Naive internal  = %.2f' % rd.rand())


def test_internal_generator(seed):
    rd.seed(seed)
    internal_generator = rd.RandomState(seed)
    print('Internal generator 1 = %.2f' % internal_generator.rand())


seed = 1

# Naive method

rd.seed(seed)
print('Naive external 1 =  %.2f' % rd.rand())
print('Naive external 2 =  %.2f' % rd.rand())
print('Naive external 3 =  %.2f' % rd.rand())
test_internal_naive(seed)
print('Naive external after seed inside function =  %.2f' % rd.rand())

print()

external_generator = rd.RandomState(seed)
print('External generator 1 = %.2f' % external_generator.rand())
print('External generator 1 = %.2f' % external_generator.rand())
test_internal_generator(seed)
print('External generator after internal = %.2f' % external_generator.rand())
