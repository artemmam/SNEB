import numpy as np
import sympy as sym
from sympy import cos, sin, tan, exp, log, sqrt
from scipy.special import factorial

pi = np.pi
prod = np.prod
asarray = np.asarray
def derive_matrix(f, x):
    """
    Function for calculating partial derivative of matrix g
    :param g : array to be derived
    :param v : variables for derivative
    :return gv: derived matrix
    """
    g_v_all = []
    for i in range(len(x)):
        g_v_all.append(sym.diff(f, x[i]))  # Calculate derivative of G with respect to v
    gv = sym.Matrix()
    for i in range(len(g_v_all)):
        gv = sym.Matrix([gv, g_v_all[i]])
    gv = gv.reshape(f.shape[0], len(x)).T
    return gv

class Benchmark(object):

    """
    Defines a global optimization benchmark problem.

    This abstract class defines the basic structure of a global
    optimization problem. Subclasses should implement the ``fun`` method
    for a particular optimization problem.

    Attributes
    ----------
    N : int
        The dimensionality of the problem.
    bounds : sequence
        The lower/upper bounds to be used for minimizing the problem.
        This a list of (lower, upper) tuples that contain the lower and upper
        bounds for the problem.  The problem should not be asked for evaluation
        outside these bounds. ``len(bounds) == N``.
    xmin : sequence
        The lower bounds for the problem
    xmax : sequence
        The upper bounds for the problem
    fglob : float
        The global minimum of the evaluated function.
    global_optimum : sequence
        A list of vectors that provide the locations of the global minimum.
        Note that some problems have multiple global minima, not all of which
        may be listed.
    nfev : int
        the number of function evaluations that the object has been asked to
        calculate.
    change_dimensionality : bool
        Whether we can change the benchmark function `x` variable length (i.e.,
        the dimensionality of the problem)
    custom_bounds : sequence
        a list of tuples that contain lower/upper bounds for use in plotting.
    """

    def __init__(self, dimensions):
        """
        Initialises the problem
        
        Parameters
        ----------
        
        dimensions : int
            The dimensionality of the problem
        """

        self._dimensions = dimensions
        self.nfev = 0
        self.fglob = np.nan
        self.global_optimum = None
        self.system_roots = []
        self.change_dimensionality = False
        self.custom_bounds = None
        self.f = None
        self.complicated = False
        self.symbols = sym.symbols(["x"+str(i) for i in range(1, dimensions + 1)])

    def __str__(self):
        return '{0} ({1} dimensions)'.format(self.__class__.__name__, self.N)

    def __repr__(self):
        return self.__class__.__name__

    def initial_vector(self):
        """
        Random initialisation for the benchmark problem.

        Returns
        -------
        x : sequence
            a vector of length ``N`` that contains random floating point
            numbers that lie between the lower and upper bounds for a given
            parameter.
        """

        return asarray([np.random.uniform(l, u) for l, u in self.bounds])

    def fun(self):
        """
        Evaluation of the benchmark function.

        Parameters
        ----------
        x : sequence
            The candidate vector for evaluating the benchmark problem. Must
            have ``len(x) == self.N``.

        Returns
        -------
        val : float
              the evaluated benchmark function
        """
        # x = self.symbols
        return None
        
    def system(self):
        return derive_matrix(sym.Matrix([self.fun()]), self.symbols)

    def change_dimensions(self, ndim):
        """
        Changes the dimensionality of the benchmark problem

        The dimensionality will only be changed if the problem is suitable

        Parameters
        ----------
        ndim : int
               The new dimensionality for the problem.
        """

        if self.change_dimensionality:
            self._dimensions = ndim
        else:
            raise ValueError('dimensionality cannot be changed for this'
                             'problem')

    @property
    def bounds(self):
        """
        The lower/upper bounds to be used for minimizing the problem.
        This a list of (lower, upper) tuples that contain the lower and upper
        bounds for the problem.  The problem should not be asked for evaluation
        outside these bounds. ``len(bounds) == N``.
        """
        if self.change_dimensionality:
            return [self._bounds[0]] * self.N
        else:
            return self._bounds

    @property
    def N(self):
        """        
        The dimensionality of the problem.
        
        Returns
        -------
        N : int
            The dimensionality of the problem
        """
        return self._dimensions

    @property
    def xmin(self):
        """
        The lower bounds for the problem

        Returns
        -------
        xmin : sequence
            The lower bounds for the problem
        """
        return asarray([b[0] for b in self.bounds])

    @property
    def xmax(self):
        """
        The upper bounds for the problem

        Returns
        -------
        xmax : sequence
            The upper bounds for the problem
        """
        return asarray([b[1] for b in self.bounds])
    
class Adjiman(Benchmark):
    """
    Adjiman objective function.

    The Adjiman [1]_ global optimization problem is a multimodal minimization
    problem defined as follows:

    .. math::

        f_{\\text{Adjiman}}(x) = \\cos(x_1)\\sin(x_2) - \\frac{x_1}{(x_2^2 + 1)}


    with, :math:`x_1 \\in [-1, 2]` and :math:`x_2 \\in [-1, 1]`.

    *Global optimum*: :math:`f(x) = -2.02181` for :math:`x = [2.0, 0.10578]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = ([-1.0, 2.0], [-1.0, 1.0])
        self.global_optimum = [[2.0, 0.10578]]
        self.fglob = -2.02180678

    def fun(self):
        x = self.symbols
        return cos(x[0]) * sin(x[1]) - x[0] / (x[1] ** 2 + 1)

class Alpine02(Benchmark):
    """
    Alpine02 objective function.

    The Alpine02 [1]_ global optimization problem is a multimodal minimization
    problem defined as follows:

    .. math::

        f_{\\text{Alpine02}(x) = \\prod_{i=1}^{n} \\sqrt{x_i} \\sin(x_i)


    Here, :math:`n` represents the number of dimensions and :math:`x_i \\in [0,
    10]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = -6.1295` for :math:`x =
    [7.91705268, 4.81584232]` for :math:`i = 1, 2`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO: eqn 7 in [1]_ has the wrong global minimum value.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        # self._bounds = list(zip([0.0] * self.N, [10.0] * self.N))
        self._bounds = ((1, 10), (1, 10)) # уменьшил границы
        self.global_optimum = [[7.91705268, 4.81584232]]
        self.fglob = -6.1295
        self.change_dimensionality = True

    def fun(self):
        x = self.symbols
        return prod([sqrt(x_i) * sin(x_i) for x_i in x])

class Brad(Benchmark):
    """ Non-Differentiable System ???
    Brad objective function.

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=3):
        Benchmark.__init__(self, dimensions)
        self._bounds = [(-0.25, 0.25), (0.01, 2.5), (0.01, 2.5)]
        self.global_optimum = [[0.0824, 1.133, 2.3437]]
        self.fglob = 0.00821487

    def fun(self):
        x = self.symbols
        a = 0
        y = [0.14, 0.18, 0.22, 0.25, 0.29, 0.32, 0.35, 0.39, 0.37, 0.58, 0.73, 0.96, 1.34, 2.10, 4.39]
        for i in range(15):
            math_i = i + 1
            u_i = math_i
            v_i = 16 - math_i
            w_i = min(u_i, v_i)
            a += (y[i] - x[0] - u_i/(x[1]*v_i + x[2]*w_i))**2
#             a += ((y[i] - x[0] - u_i)/(v_i*x[1] + w_i*x[2]))**2
        return a

    def system(self):
        return derive_matrix(sym.Matrix([self.fun()]), self.symbols)
    
class Beale(Benchmark):
    """
    Beale objective function.

    The Beale [1]_ global optimization problem is a multimodal
    minimization problem defined as follows:

    .. math::

        f_{\\text{Beale}}(x) = \\left(x_1 x_2 - x_1 + 1.5\\right)^{2} +
        \\left(x_1 x_2^{2} - x_1 + 2.25\\right)^{2} + \\left(x_1 x_2^{3} - x_1 +
        2.625\\right)^{2}


    with :math:`x_i \\in [-4.5, 4.5]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x=[3, 0.5]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-4.5] * self.N, [4.5] * self.N))
        self.global_optimum = [[3.0, 0.5]]
        self.fglob = 0.0

    def fun(self):
        x = self.symbols
        return (1.5 - x[0] + x[0] * x[1]) ** 2 + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2 + (2.625 - x[0] + x[0] * x[1] ** 3) ** 2

class BiggsExp02(Benchmark):

    r"""
    BiggsExp02 objective function.

    The BiggsExp02 [1]_ global optimization problem is a multimodal minimization
    problem defined as follows

    .. math::

        \begin{matrix}
        f_{\text{BiggsExp02}}(x) = \sum_{i=1}^{10} (e^{-t_i x_1}
           - 5 e^{-t_i x_2} - y_i)^2 \\
        t_i = 0.1 i\\
        y_i = e^{-t_i} - 5 e^{-10t_i}\\
        \end{matrix}


    with :math:`x_i \in [0, 20]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [1, 10]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([0] * 2,
                                [20] * 2))
        self.global_optimum = [[1., 10.]]
        self.fglob = 0

    def fun(self):
        x = self.symbols
        a = 0
        for i in range(10):
            math_i = i + 1
            t_i = 0.1*i
            y_i = exp(-t_i) - 5*exp(-10*t_i)
            a += (exp(-t_i*x[0]) - 5*exp(-t_i*x[1]) - y_i)**2
        return a

class BiggsExp03(Benchmark):

    r"""
    BiggsExp03 objective function.

    The BiggsExp03 [1]_ global optimization problem is a multimodal minimization
    problem defined as follows

    .. math::

        \begin{matrix}\ f_{\text{BiggsExp03}}(x) = \sum_{i=1}^{10}
        (e^{-t_i x_1} - x_3e^{-t_i x_2} - y_i)^2\\
        t_i = 0.1i\\
        y_i = e^{-t_i} - 5e^{-10 t_i}\\
        \end{matrix}


    with :math:`x_i \in [0, 20]` for :math:`i = 1, 2, 3`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [1, 10, 5]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    """

    def __init__(self, dimensions=3):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([0] * 3,
                                [20] * 3))
        self.global_optimum = [[1., 10., 5.]]
        self.fglob = 0

    def fun(self):
        x = self.symbols
        a = 0
        for i in range(10):
            math_i = i + 1
            t_i = 0.1*i
            y_i = exp(-t_i) - 5*exp(-10*t_i)
            a += (exp(-t_i*x[0]) - x[2]*exp(-t_i*x[1]) - y_i)**2
        return a

class BiggsExp04(Benchmark):

    r"""
    BiggsExp04 objective function.

    The BiggsExp04 [1]_ global optimization problem is a multimodal
    minimization problem defined as follows

    .. math::

        \begin{matrix}\ f_{\text{BiggsExp04}}(x) = \sum_{i=1}^{10}
        (x_3 e^{-t_i x_1} - x_4 e^{-t_i x_2} - y_i)^2\\
        t_i = 0.1i\\
        y_i = e^{-t_i} - 5 e^{-10 t_i}\\
        \end{matrix}


    with :math:`x_i \in [0, 20]` for :math:`i = 1, ..., 4`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [1, 10, 1, 5]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    """

    def __init__(self, dimensions=4):
        Benchmark.__init__(self, dimensions)
        #self.complicated = True
        self._bounds = [(0, 2), (9, 11), (0, 2), (4, 6)]
        self.global_optimum = [[1., 10., 1., 5.]]
        self.fglob = 0

    def fun(self):
        x = self.symbols
        a = 0
        for i in range(10):
            math_i = i + 1
            t_i = 0.1*i
            y_i = exp(-t_i) - 5*exp(-10*t_i)
            a += (x[2]*exp(-t_i*x[0]) - x[3]*exp(-t_i*x[1]) - y_i)**2
        return a

class BiggsExp05(Benchmark):

    r"""
    BiggsExp05 objective function.

    The BiggsExp05 [1]_ global optimization problem is a multimodal minimization
    problem defined as follows

    .. math::

        \begin{matrix}\ f_{\text{BiggsExp05}}(x) = \sum_{i=1}^{11}
        (x_3 e^{-t_i x_1} - x_4 e^{-t_i x_2} + 3 e^{-t_i x_5} - y_i)^2\\
        t_i = 0.1i\\
        y_i = e^{-t_i} - 5e^{-10 t_i} + 3e^{-4 t_i}\\
        \end{matrix}


    with :math:`x_i \in [0, 20]` for :math:`i=1, ..., 5`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [1, 10, 1, 5, 4]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    """

    def __init__(self, dimensions=5):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([0.] * 5,
                                [20.] * 5))
        self.global_optimum = [[1., 10., 1., 5., 4.]]
        self.fglob = 0
        self.complicated = True
    def fun(self):
        x = self.symbols
        a = 0
        for i in range(10):
            math_i = i + 1
            t_i = 0.1*i
            y_i = exp(-t_i) - 5*exp(-10*t_i) + 3*exp(-4*t_i)
            a += (x[2]*exp(-t_i*x[0]) - x[3]*exp(-t_i*x[1]) + 3*exp(-t_i*x[4]) - y_i)**2
        return a

class Bird(Benchmark):
    """
    Bird objective function.

    The Bird global optimization problem is a multimodal minimization
    problem defined as follows

    .. math::

        f_{\\text{Bird}}(x) = \\left(x_1 - x_2\\right)^{2} + e^{\\left[1 -
         \\sin\\left(x_1\\right) \\right]^{2}} \\cos\\left(x_2\\right) + e^{\\left[1 -
          \\cos\\left(x_2\\right)\\right]^{2}} \\sin\\left(x_1\\right)


    with :math:`x_i \\in [-2\\pi, 2\\pi]`

    *Global optimum*: :math:`f(x) = -106.7645367198034` for :math:`x
    = [4.701055751981055, 3.152946019601391]` or :math:`x =
    [-1.582142172055011, -3.130246799635430]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-2.0 * pi] * self.N, [2.0 * pi] * self.N))
        self.global_optimum = [[4.701055751981055, 3.152946019601391], [-1.582142172055011, -3.13024679963543]]
        self.fglob = -106.7645367198034

    def fun(self):
        x = self.symbols
        return sin(x[0]) * exp((1 - cos(x[1])) ** 2) + cos(x[1]) * exp((1 - sin(x[0])) ** 2) + (x[0] - x[1]) ** 2

class Bohachevsky1(Benchmark):
    """
    Bohachevsky 1 objective function.

    The Bohachevsky 1 [1]_ global optimization problem is a multimodal
    minimization problem defined as follows

        .. math::

        f_{\\text{Bohachevsky}}(x) = \\sum_{i=1}^{n-1}\\left[x_i^2 + 2 x_{i+1}^2 -
        0.3 \\cos(3 \\pi x_i) - 0.4 \\cos(4 \\pi x_{i + 1}) + 0.7 \\right]


    Here, :math:`n` represents the number of dimensions and :math:`x_i \\in
    [-15, 15]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for :math:`i = 1,
    ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO: equation needs to be fixed up in the docstring. see Jamil#17
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-15.0] * self.N, [15.0] * self.N))
        self.global_optimum = [[0 for _ in range(self.N)]]
        self.fglob = 0.0

    def fun(self):
        x = self.symbols
        return x[0] ** 2 + 2 * x[1] ** 2 - 0.3 * cos(3 * pi * x[0]) - 0.4 * cos(4 * pi * x[1]) + 0.7
    
class Bohachevsky2(Benchmark):
    """
    Bohachevsky 2 objective function.

    The Bohachevsky 2 [1]_ global optimization problem is a multimodal
    minimization problem defined as follows

        .. math::

        f_{\\text{Bohachevsky}}(x) = \\sum_{i=1}^{n-1}\\left[x_i^2 + 2 x_{i+1}^2 -
        0.3 \\cos(3 \\pi x_i) - 0.4 \\cos(4 \\pi x_{i + 1}) + 0.7 \\right]


    Here, :math:`n` represents the number of dimensions and :math:`x_i \\in
    [-15, 15]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for :math:`i = 1,
    ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO: equation needs to be fixed up in the docstring. Jamil is also wrong.
    There should be no 0.4 factor in front of the cos term
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-15.0] * self.N, [15.0] * self.N))
        self.global_optimum = [[0 for _ in range(self.N)]]
        self.fglob = 0.0

    def fun(self):
        x = self.symbols
        return x[0] ** 2 + 2 * x[1] ** 2 - 0.3 * cos(3 * pi * x[0]) * cos(4 * pi * x[1]) + 0.3
    
class Bohachevsky3(Benchmark):
    """
    Bohachevsky 3 objective function.

    The Bohachevsky 3 [1]_ global optimization problem is a multimodal
    minimization problem defined as follows

        .. math::

        f_{\\text{Bohachevsky}}(x) = \\sum_{i=1}^{n-1}\\left[x_i^2 + 2 x_{i+1}^2 -
        0.3 \\cos(3 \\pi x_i) - 0.4 \\cos(4 \\pi x_{i + 1}) + 0.7 \\right]


    Here, :math:`n` represents the number of dimensions and :math:`x_i \\in
    [-15, 15]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for :math:`i = 1,
    ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO: equation needs to be fixed up in the docstring. Jamil#19
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-15.0] * self.N, [15.0] * self.N))
        self.global_optimum = [[0 for _ in range(self.N)]]
        self.fglob = 0.0

    def fun(self):
        x = self.symbols
        return x[0] ** 2 + 2 * x[1] ** 2 - 0.3 * cos(3 * pi * x[0] + 4 * pi * x[1]) + 0.3

class Booth(Benchmark):
    """
    Booth objective function.
    
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))
        self.global_optimum = [[1, 3]]
        self.fglob = 0.0

    def fun(self):
        x = self.symbols
        return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2
    
class BoxBetts(Benchmark):

    r"""
    BoxBetts objective function.

    The BoxBetts global optimization problem is a multimodal
    minimization problem defined as follows

    .. math::

        f_{\text{BoxBetts}}(x) = \sum_{i=1}^k g(x_i)^2


    Where, in this exercise:

    .. math::

        g(x) = e^{-0.1i x_1} - e^{-0.1i x_2} - x_3\left[e^{-0.1i}
        - e^{-i}\right]


    And :math:`k = 10`.

    Here, :math:`x_1 \in [0.9, 1.2], x_2 \in [9, 11.2], x_3 \in [0.9, 1.2]`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [1, 10, 1]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=3):
        Benchmark.__init__(self, dimensions)
        self.complicated = True
        self._bounds = ([0.9, 1.2], [9.0, 11.2], [0.9, 1.2])
        self.global_optimum = [[1.0, 10.0, 1.0]]
        self.fglob = 0.0

    def fun(self):
        x = self.symbols
        a = 0
        for i in range(self.N - 1):
            g = exp(-0.1*x[0]*(i + 1)) - exp(-0.1*x[1]*(i + 1)) - (exp(-0.1*(i + 1)) - exp(-(i + 1)))*x[2]
            a += g**2
        return a
    
class Branin01(Benchmark):
    """
    Branin01  objective function.

    The Branin01 global optimization problem is a multimodal minimization
    problem defined as follows

    .. math::

        f_{\\text{Branin01}}(x) = \\left(- 1.275 \\frac{x_1^{2}}{\\pi^{2}} + 5
        \\frac{x_1}{\\pi} + x_2 -6\\right)^{2} + \\left(10 -\\frac{5}{4 \\pi} \\right)
        \\cos\\left(x_1\\right) + 10


    with :math:`x_1 \\in [-5, 10], x_2 \\in [0, 15]`

    *Global optimum*: :math:`f(x) = 0.39788735772973816` for :math:`x =
    [-\\pi, 12.275]` or :math:`x = [\\pi, 2.275]` or :math:`x = [3\\pi, 2.475]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO: Jamil#22, one of the solutions is different
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = [(-5.0, 10.0), (0.0, 15.0)]
        self.global_optimum = [[-pi, 12.275], [pi, 2.275], [3 * pi, 2.475]]
        self.fglob = 0.39788735772973816

    def fun(self):
        x = self.symbols
        return (x[1] - 5.1 / (4 * pi ** 2) * x[0] ** 2 + 5 * x[0] / pi - 6) ** 2 + 10 * (1 - 1 / (8 * pi)) * cos(x[0]) + 10

class Branin02(Benchmark):
    """
    Branin02 objective function.

    The Branin02 global optimization problem is a multimodal minimization
    problem defined as follows

    .. math::

        f_{\\text{Branin02}}(x) = \\left(- 1.275 \\frac{x_1^{2}}{\\pi^{2}}
        + 5 \\frac{x_1}{\\pi} + x_2 - 6 \\right)^{2} + \\left(10 - \\frac{5}{4 \\pi}
        \\right) \\cos\\left(x_1\\right) \\cos\\left(x_2\\right)
        + \\log(x_1^2+x_2^2 + 1) + 10


    with :math:`x_i \\in [-5, 15]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 5.559037` for :math:`x = [-3.2, 12.53]`

    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = [(-5.0, 15.0), (-5.0, 15.0)]
        self.global_optimum = [[-3.1969884, 12.52625787]]
        self.fglob = 5.558914403893825

    def fun(self):
        x = self.symbols
        return (x[1] - 5.1 / (4 * pi ** 2) * x[0] ** 2 + 5 * x[0] / pi - 6) ** 2 + 10 * (1 - 1 / (8 * pi)) * cos(x[0]) * cos(x[1]) + log(x[0] ** 2.0 + x[1] ** 2.0 + 1.0) + 10

class Brent(Benchmark):
    """ 
    Brent objective function.

    The Brent [1]_ global optimization problem is a multimodal minimization
    problem defined as follows:

    .. math::

        f_{\\text{Brent}}(x) = (x_1 + 10)^2 + (x_2 + 10)^2 + e^{(-x_1^2 -x_2^2)}


    with :math:`x_i \\in [-10, 10]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [-10, -10]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO solution is different to Jamil#24
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))
        self.custom_bounds = ([-10, 2], [-10, 2])
        self.global_optimum = [[-10.0, -10.0]]
        self.fglob = 0.0

    def fun(self):
        x = self.symbols
        return (x[0] + 10.0) ** 2.0 + (x[1] + 10.0) ** 2.0 + exp(-x[0] ** 2.0 - x[1] ** 2.0)
    
class Bukin02(Benchmark):
    """ минимум на границе
    Bukin02 objective function.

    The Bukin02 [1]_ global optimization problem is a multimodal minimization
    problem defined as follows:

    .. math::

        f_{\\text{Bukin02}}(x) = 100 (x_2^2 - 0.01x_1^2 + 1)
        + 0.01(x_1 + 10)^2


    with :math:`x_1 \\in [-15, -5], x_2 \\in [-3, 3]`

    *Global optimum*: :math:`f(x) = -124.75` for :math:`x = [-15, 0]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO: I think that Gavana and Jamil are wrong on this function. In both
    sources the x[1] term is not squared. As such there will be a minimum at
    the smallest value of x[1].
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = [(-5.0, 5.0), (-5.0, 5.0)]
        self.global_optimum = [[-15.0, 0.0]]
        self.fglob = -124.75
        # if hasattr(self, '_bounds') and self._bounds and (not isinstance(self._bounds[0], (int, float))):
            # try:
                # lowers = [b[0] for b in self._bounds]
                # uppers = [b[1] for b in self._bounds]
                # self._bounds = [min(lowers), max(uppers)]
            # except:
                # pass

    def fun(self):
        x = self.symbols
        return 100 * (x[1] ** 2 - 0.01 * x[0] ** 2 + 1.0) + 0.01 * (x[0] + 10.0) ** 2.0

class Chichinadze(Benchmark):
    """
    Chichinadze objective function.

    This class defines the Chichinadze [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

        f_{	ext{Chichinadze}}(x) = x_{1}^{2} - 12 x_{1}
        + 8 \\sin\\left(\x0crac{5}{2} \\pi x_{1}\right)
        + 10 \\cos\\left(\x0crac{1}{2} \\pi x_{1}\right) + 11
        - 0.2 \x0crac{\\sqrt{5}}{e^{\x0crac{1}{2} \\left(x_{2} -0.5 \right)^{2}}}


    with :math:`x_i \\in [-30, 30]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = -42.94438701899098` for :math:`x =
    [6.189866586965680, 0.5]`

    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015

    TODO: Jamil#33 has a dividing factor of 2 in the sin term.  However, f(x)
    for the given solution does not give the global minimum. i.e. the equation
    is at odds with the solution.
    Only by removing the dividing factor of 2, i.e. `8 * sin(5 * pi * x[0])`
    does the given solution result in the given global minimum.
    Do we keep the result or equation?
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        # self._bounds = list(zip([-30.0] * self.N, [30.0] * self.N))
        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))
        self.custom_bounds = [(-10, 10), (-10, 10)]
        self.global_optimum = [[6.18986658696568, 0.5]]
        self.fglob = -42.94438701899098

    def fun(self):
        x = self.symbols
        return x[0] ** 2 - 12 * x[0] + 11 + 10 * cos(pi * x[0] / 2) + 8 * sin(5 * pi * x[0] / 2) - 1.0 / sqrt(5) * exp(-(x[1] - 0.5) ** 2 / 2)

class Cigar(Benchmark):
    """
    Cigar objective function.

    This class defines the Cigar [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{	ext{Cigar}}(x) = x_1^2 + 10^6\\sum_{i=2}^{n} x_i^2


    Here, :math:`n` represents the number of dimensions and :math:`x_i \\in
    [-100, 100]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for
    :math:`i = 1, ..., n`

    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-100.0] * self.N, [100.0] * self.N))
        self.custom_bounds = [(-5, 5), (-5, 5)]
        self.global_optimum = [[0 for _ in range(self.N)]]
        self.fglob = 0.0
        self.change_dimensionality = True

    def fun(self):
        x = self.symbols
        sum1 = 0
        for i in range(1, self.N):
            sum1+= x[i]**2
        return x[0] ** 2 + 1000000.0 * sum1

class Colville(Benchmark):
    """
    Colville objective function.

    This class defines the Colville global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{	ext{Colville}}(x) = \\left(x_{1} -1\right)^{2}
        + 100 \\left(x_{1}^{2} - x_{2}\right)^{2}
        + 10.1 \\left(x_{2} -1\right)^{2} + \\left(x_{3} -1\right)^{2}
        + 90 \\left(x_{3}^{2} - x_{4}\right)^{2}
        + 10.1 \\left(x_{4} -1\right)^{2} + 19.8 \x0crac{x_{4} -1}{x_{2}}


    with :math:`x_i \\in [-10, 10]` for :math:`i = 1, ..., 4`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 1` for
    :math:`i = 1, ..., 4`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO docstring equation is wrong use Jamil#36
    """

    def __init__(self, dimensions=4):
        Benchmark.__init__(self, dimensions)
        # self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))
        self._bounds = list(zip([-2.0] * self.N, [2.0] * self.N))
        self.custom_bounds = list(zip([-2.0] * self.N, [2.0] * self.N))
        self.global_optimum = [[1 for _ in range(self.N)]]
        self.fglob = 0.0

    def fun(self):
        x = self.symbols
        a = 100 * (x[0] - (x[1])**2)**2 + (1 - x[0])**2
        b = 90 * (x[3] - (x[2])**2)**2 + (1 - x[2])**2
        c = 10.1 * ((x[1] - 1)**2 + (x[3] - 1)**2)
        d = 19.8 * (x[1] - 1) * (x[3] - 1)
        return a + b + c + d

class CosineMixture(Benchmark):
    """
    Cosine Mixture objective function.

    This class defines the Cosine Mixture global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{	ext{CosineMixture}}(x) = -0.1 \\sum_{i=1}^n \\cos(5 \\pi x_i)
        - \\sum_{i=1}^n x_i^2


    Here, :math:`n` represents the number of dimensions and :math:`x_i \\in
    [-1, 1]` for :math:`i = 1, ..., N`.

    *Global optimum*: :math:`f(x) = -0.1N` for :math:`x_i = 0` for
    :math:`i = 1, ..., N`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO, Jamil #38 has wrong minimum and wrong fglob. I plotted it.
    -(x**2) term is always negative if x is negative.
     cos(5 * pi * x) is equal to -1 for x=-1.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self.change_dimensionality = True
        self._bounds = list(zip([-1.0] * self.N, [1.0] * self.N))
        self.global_optimum = [[-1.0 for _ in range(self.N)]]
        self.fglob = -0.9 * self.N

    def fun(self):
        x = self.symbols
        sum1 = 0
        sum2 = 0
        for i in range(self.N):
            sum1 += cos(5*pi*x[i])
            sum2 += x[i]**2
        return -0.1*sum1 - sum2

class Cube(Benchmark):
    """
    Cube objective function.

    This class defines the Cube global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{	ext{Cube}}(x) = 100(x_2 - x_1^3)^2 + (1 - x1)^2


    Here, :math:`n` represents the number of dimensions and :math:`x_i \\in [-10, 10]` for :math:`i=1,...,N`.

    *Global optimum*: :math:`f(x_i) = 0.0` for :math:`x = [1, 1]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO: jamil#41 has the wrong solution.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))
        self.custom_bounds = ([0, 2], [0, 2])
        self.global_optimum = [[1.0, 1.0]]
        self.fglob = 0.0

    def fun(self):
        x = self.symbols
        return 100.0 * (x[1] - x[0] ** 3.0) ** 2.0 + (1.0 - x[0]) ** 2.0

    
class Deb01(Benchmark):
    """
    Deb 1 objective function.

    This class defines the Deb 1 [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

        f_{	ext{Deb01}}(x) = - \x0crac{1}{N} \\sum_{i=1}^n \\sin^6(5 \\pi x_i)


    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \\in [-1, 1]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x_i) = 0.0`. The number of global minima is
    :math:`5^n` that are evenly spaced in the function landscape, where
    :math:`n` represents the dimension of the problem.

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self.change_dimensionality = True
        # self._bounds = list(zip([-1.0] * self.N, [1.0] * self.N))
        self._bounds = list(zip([-0.5] * self.N, [0.5] * self.N))
        self.global_optimum = [[0.3, -0.3]]
        self.fglob = -1.0

    def fun(self):
        x = self.symbols
        return -(1.0 / self.N) * sum([sin(5 * pi * x_i) ** 6.0 for x_i in x])
    
class DeckkersAarts(Benchmark):
    """
    Deckkers-Aarts objective function.

    This class defines the Deckkers-Aarts [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{	ext{DeckkersAarts}}(x) = 10^5x_1^2 + x_2^2 - (x_1^2 + x_2^2)^2
        + 10^{-5}(x_1^2 + x_2^2)^4


    with :math:`x_i \\in [-20, 20]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = -24776.518242168` for
    :math:`x = [0, \\pm 14.9451209]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO: jamil solution and global minimum are slightly wrong.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-20.0] * self.N, [20.0] * self.N))
        self.custom_bounds = ([-1, 1], [14, 16])
        self.global_optimum = [[0.0, 14.9451209]]
        self.fglob = -24776.518342168

    def fun(self):
        x = self.symbols
        return 100000.0 * x[0] ** 2 + x[1] ** 2 - (x[0] ** 2 + x[1] ** 2) ** 2 + 1e-05 * (x[0] ** 2 + x[1] ** 2) ** 4

class DeVilliersGlasser01(Benchmark):

    """ too hard
    DeVilliers-Glasser 1 objective function.

    This class defines the DeVilliers-Glasser 1 [1]_ function global optimization
    problem. This is a multimodal minimization problem defined as follows:

    .. math::

       f_{\text{DeVilliersGlasser01}}(x) = \sum_{i=1}^{24} \left[ x_1x_2^{t_i}
       \sin(x_3t_i + x_4) - y_i \right ]^2


    Where, in this exercise, :math:`t_i = 0.1(i - 1)` and
    :math:`y_i = 60.137(1.371^{t_i}) \sin(3.112t_i + 1.761)`.

    Here, :math:`n` represents the number of dimensions and :math:`x_i \in
    [1, 100]` for :math:`i = 1, ..., 4`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for
    :math:`x = [60.137, 1.371, 3.112, 1.761]`.

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    """

    def __init__(self, dimensions=4):
        Benchmark.__init__(self, dimensions)

        self._bounds = [(59, 61), (1, 2), (3, 4), (1, 2)]#list(zip([1.0] * self.N, [100.0] * self.N))
        # self.complicated = True
        self.global_optimum = [[60.137, 1.371, 3.112, 1.761]]
        self.fglob = 0.0

    def fun(self):
        x = self.symbols
        a = 0
        for i in range(24):
            math_i = i + 1
            t_i = 0.1*(i - 1)
            y_i = 60.137 * 1.371**(t_i)*sin(3.112*t_i + 1.761)
            a += (x[0] * x[1]**(t_i)*sin(x[2]*t_i + x[3]) - y_i)**2
        return a

class DixonPrice(Benchmark):
    """
    Dixon and Price objective function.

    This class defines the Dixon and Price global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{	ext{DixonPrice}}(x) = (x_i - 1)^2
        + \\sum_{i=2}^n i(2x_i^2 - x_{i-1})^2


    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \\in [-10, 10]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x_i) = 0` for
    :math:`x_i = 2^{- \x0crac{(2^i - 2)}{2^i}}` for :math:`i = 1, ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO: Gavana code not correct.  i array should start from 2.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))
        self.custom_bounds = [(-2, 3), (-2, 3)]
        self.global_optimum = [[2.0 ** (-(2.0 ** i - 2.0) / 2.0 ** i) for i in range(1, self.N + 1)]]
        self.fglob = 0.0
        self.change_dimensionality = True

    def fun(self):
        x = self.symbols
        sum1 = 0
        for i in range(1, self.N):
            sum1 += (i + 1)*((2*(x[i])**2 - x[i - 1])**2) 
        return (x[0] - 1)**2 + sum1

class EggCrate(Benchmark):
    """
    Egg Crate objective function.

    This class defines the Egg Crate [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\\text{EggCrate}}(x) = x_1^2 + x_2^2 + 25 \\left[ \\sin^2(x_1)
        + \\sin^2(x_2) \\right]


    with :math:`x_i \\in [-5, 5]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [0, 0]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-5.0] * self.N, [5.0] * self.N))
        self.global_optimum = [[0.0, 0.0]]
        self.fglob = 0.0

    def fun(self):
        x = self.symbols
        return x[0] ** 2 + x[1] ** 2 + 25 * (sin(x[0]) ** 2 + sin(x[1]) ** 2)

class Easom(Benchmark):

    r"""
    Easom objective function.

    This class defines the Easom [1]_ global optimization problem. This is a
    a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Easom}}({x}) = a - \frac{a}{e^{b \sqrt{\frac{\sum_{i=1}^{n}
        x_i^{2}}{n}}}} + e - e^{\frac{\sum_{i=1}^{n} \cos\left(c x_i\right)}
        {n}}


    Where, in this exercise, :math:`a = 20, b = 0.2` and :math:`c = 2 \pi`.

    Here, :math:`x_i \in [-100, 100]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [0, 0]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO Gavana website disagrees with Jamil, etc. Gavana equation in docstring is totally wrong.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-10.0] * self.N,
                           [10.0] * self.N))

        self.global_optimum = [[pi for _ in range(self.N)]]
        self.fglob = -1.0
        # self.complicated = True
    def fun(self):
        x = self.symbols
        a = (x[0] - pi)**2 + (x[1] - pi)**2
        return -cos(x[0]) * cos(x[1]) * exp(-a)
    
class ElAttarVidyasagarDutta(Benchmark):
    """
    El-Attar-Vidyasagar-Dutta [1]_ objective function.

    This class defines the El-Attar-Vidyasagar-Dutta function global
    optimization problem. This is a multimodal minimization problem defined as
    follows:

    .. math::

       f_{\\text{ElAttarVidyasagarDutta}}(x) = (x_1^2 + x_2 - 10)^2
       + (x_1 + x_2^2 - 7)^2 + (x_1^2 + x_2^3 - 1)^2


    with :math:`x_i \\in [-100, 100]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 1.712780354` for
    :math:`x= [3.40918683, -2.17143304]`

    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-100.0] * self.N, [100.0] * self.N))
        self.custom_bounds = [(-4, 4), (-4, 4)]
        self.global_optimum = [[3.40918683, -2.17143304]]
        self.fglob = 1.712780354

    def fun(self):
        x = self.symbols
        return (x[0] ** 2 + x[1] - 10) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2 + (x[0] ** 2 + x[1] ** 3 - 1) ** 2

class Engvall(Benchmark):
    """
    El-Attar-Vidyasagar-Dutta [1]_ objective function.

    This class defines the El-Attar-Vidyasagar-Dutta function global
    optimization problem. This is a multimodal minimization problem defined as
    follows:

    .. math::

       f_{\\text{ElAttarVidyasagarDutta}}(x) = (x_1^2 + x_2 - 10)^2
       + (x_1 + x_2^2 - 7)^2 + (x_1^2 + x_2^3 - 1)^2


    with :math:`x_i \\in [-100, 100]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 1.712780354` for
    :math:`x= [3.40918683, -2.17143304]`

    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-5.0] * self.N, [5.0] * self.N))
        self.global_optimum = [[1, 0]]
        self.fglob = 0

    def fun(self):
        x = self.symbols
        return x[0]**4 + x[1]**4 + 2.0 * (x[0]**2) * (x[1]**2) - 4*x[0] + 3.0

class Exp2(Benchmark):
    """
    Exp2 objective function.

    This class defines the Exp2 global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

        f_{\\text{Exp2}}(x) = \\sum_{i=0}^9 \\left ( e^{-ix_1/10} - 5e^{-ix_2/10}
        - e^{-i/10} + 5e^{-i} \\right )^2


    with :math:`x_i \\in [0, 20]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [1, 10.]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([0.0] * self.N, [20.0] * self.N))
        self.custom_bounds = [(0, 2), (0, 20)]
        self.global_optimum = [[1.0, 10.0]]
        self.fglob = 0.0

    def fun(self):
        x = self.symbols
        sum1 = 0
        for i in range(10):
            sum1 += (exp(-i*x[0]/10) - 5*exp(-i*x[1]/10) - exp(-i/10) + 5*exp(-i))**2
        return sum1

class Exponential(Benchmark):
    """
    Exponential [1] objective function.

    This class defines the Exponential global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\\text{Exponential}}(x) = -e^{-0.5 \\sum_{i=1}^n x_i^2}


    Here, :math:`n` represents the number of dimensions and :math:`x_i \\in
    [-1, 1]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x_i) = -1` for :math:`x_i = 0` for
    :math:`i = 1, ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO Jamil are missing a minus sign on fglob
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-1.0] * self.N, [1.0] * self.N))
        self.global_optimum = [[0.0 for _ in range(self.N)]]
        self.fglob = -1.0
        self.change_dimensionality = True

    def fun(self):
        x = self.symbols
        sum1 = 0
        for i in range(self.N):
            sum1 += x[i]**2
        return -exp(-0.5*sum1)

class FreudensteinRoth(Benchmark):
    """
    FreudensteinRoth objective function.

    This class defines the Freudenstein & Roth [1]_ global optimization problem.
    This is a multimodal minimization problem defined as follows:

    .. math::

        f_{\\text{FreudensteinRoth}}(x) =  \\left\\{x_1 - 13 + \\left[(5 - x_2) x_2
        - 2 \\right] x_2 \\right\\}^2 + \\left \\{x_1 - 29 
        + \\left[(x_2 + 1) x_2 - 14 \\right] x_2 \\right\\}^2


    with :math:`x_i \\in [-10, 10]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [5, 4]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))
        self.custom_bounds = [(-3, 3), (-5, 5)]
        self.global_optimum = [[5.0, 4.0]]
        self.fglob = 0.0

    def fun(self):
        x = self.symbols
        f1 = (-13.0 + x[0] + ((5.0 - x[1]) * x[1] - 2.0) * x[1]) ** 2
        f2 = (-29.0 + x[0] + ((x[1] + 1.0) * x[1] - 14.0) * x[1]) ** 2
        return f1 + f2

class Giunta(Benchmark):
    """
    Giunta objective function.

    This class defines the Giunta [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\\text{Giunta}}({x}) = 0.6 + \\sum_{i=1}^{n} \\left[\\sin^{2}\\left(1
        - \\frac{16}{15} x_i\\right) - \\frac{1}{50} \\sin\\left(4
        - \\frac{64}{15} x_i\\right) - \\sin\\left(1
        - \\frac{16}{15} x_i\\right)\\right]


    with :math:`x_i \\in [-1, 1]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0.06447042053690566` for
    :math:`x = [0.4673200277395354, 0.4673200169591304]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO Jamil has the wrong fglob.  I think there is a lower value.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-1.0] * self.N, [1.0] * self.N))
        self.global_optimum = [[0.4673200277395354, 0.4673200169591304]]
        self.fglob = 0.06447042053690566

    def fun(self):
        x = self.symbols
        sum1 = 0
        for i in range(self.N):
            arg = (16/15) * x[i] - 1
            sum1 += sin(arg) + (sin(arg))**2 + (1/50) * sin(4 * arg)
        return 0.6 + sum1
    
class GoldsteinPrice(Benchmark):
    """
    Goldstein-Price objective function.

    This class defines the Goldstein-Price [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\\text{GoldsteinPrice}}(x) = \\left[ 1 + (x_1 + x_2 + 1)^2 
        (19 - 14 x_1 + 3 x_1^2 - 14 x_2 + 6 x_1 x_2 + 3 x_2^2) \\right]
        \\left[ 30 + ( 2x_1 - 3 x_2)^2 (18 - 32 x_1 + 12 x_1^2
        + 48 x_2 - 36 x_1 x_2 + 27 x_2^2) \\right]


    with :math:`x_i \\in [-2, 2]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 3` for :math:`x = [0, -1]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-2.0] * self.N, [2.0] * self.N))
        self.global_optimum = [[0.0, -1.0]]
        self.fglob = 3.0

    def fun(self):
        x = self.symbols
        a = 1 + (x[0] + x[1] + 1) ** 2 * (19 - 14 * x[0] + 3 * x[0] ** 2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1] ** 2)
        b = 30 + (2 * x[0] - 3 * x[1]) ** 2 * (18 - 32 * x[0] + 12 * x[0] ** 2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1] ** 2)
        return a * b

class GramacyLee2(Benchmark):
    """
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-1.0] * self.N, [1.0] * self.N))
        self.global_optimum = [[-0.707106776321847, 0.0]]
        self.fglob = -0.428881942480353
    def fun(self):
        x = self.symbols
        a = -(x[0]**2)-(x[1]**2)
        return x[0]*sym.exp(a)

class Griewank(Benchmark):
    """
    Griewank objective function.

    This class defines the Griewank global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\\text{Griewank}}(x) = \\frac{1}{4000}\\sum_{i=1}^n x_i^2
        - \\prod_{i=1}^n\\cos\\left(\\frac{x_i}{\\sqrt{i}}\\right) + 1


    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \\in [-600, 600]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for
    :math:`i = 1, ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-50.0] * self.N, [50.0] * self.N))
        self.custom_bounds = [(-10, 10), (-10, 10)]
        self.global_optimum = [[0 for _ in range(self.N)]]
        self.fglob = 0.0
        self.change_dimensionality = True

    def fun(self):
        x = self.symbols
        sum1 = 0
        pr1 = 1
        for i in range(self.N):
            sum1 += (x[i]**2)/4000
            pr1 *= cos(x[i]/(sym.sqrt(i + 1)))
        return sum1 - pr1 + 1

class Hansen(Benchmark):
    """
    Hansen objective function.

    This class defines the Hansen [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

        f_{\\text{Hansen}}(x) = \\left[ \\sum_{i=0}^4(i+1)\\cos(ix_1+i+1)\\right ]
        \\left[\\sum_{j=0}^4(j+1)\\cos[(j+2)x_2+j+1])\\right ]


    with :math:`x_i \\in [-10, 10]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = -176.54179` for
    :math:`x = [-7.58989583, -7.70831466]`.

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO Jamil #61 is missing the starting value of i.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))
        self.global_optimum = [[-7.58989583, -7.70831466]]
        self.fglob = -176.54179

    def fun(self):
        x = self.symbols
        sum1 = 0
        sum2 = 0
        for i in range(5):
            sum1 += (i + 1)*cos(i*x[0] + i + 1)
            sum2 += (i + 1)*cos((i + 2)*x[1] + i + 1)
        return sum1*sum2

class Hartmann3(Benchmark):

    r"""
    Hartmann3 objective function.

    This class defines the Hartmann3 [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Hartmann3}}(x) = -\sum\limits_{i=1}^{4} c_i 
        e^{-\sum\limits_{j=1}^{n}a_{ij}(x_j - p_{ij})^2}


    Where, in this exercise:

    .. math::

        \begin{array}{l|ccc|c|ccr}
        \hline
        i & & a_{ij}&  & c_i & & p_{ij} &  \\
        \hline
        1 & 3.0 & 10.0 & 30.0 & 1.0 & 0.3689  & 0.1170 & 0.2673 \\
        2 & 0.1 & 10.0 & 35.0 & 1.2 & 0.4699 & 0.4387 & 0.7470 \\
        3 & 3.0 & 10.0 & 30.0 & 3.0 & 0.1091 & 0.8732 & 0.5547 \\
        4 & 0.1 & 10.0 & 35.0 & 3.2 & 0.03815 & 0.5743 & 0.8828 \\
        \hline
        \end{array}


    with :math:`x_i \in [0, 1]` for :math:`i = 1, 2, 3`.

    *Global optimum*: :math:`f(x) = -3.8627821478` 
    for :math:`x = [0.11461292,  0.55564907,  0.85254697]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO Jamil #62 has an incorrect coefficient. p[1, 1] should be 0.4387
    """

    def __init__(self, dimensions=3):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([0.0] * self.N, [1.0] * self.N))

        self.global_optimum = [[0.11461292, 0.55564907, 0.85254697]]
        self.fglob = -3.8627821478

        self.a = np.asarray([[3.0, 10., 30.],
                          [0.1, 10., 35.],
                          [3.0, 10., 30.],
                          [0.1, 10., 35.]])

        self.p = np.asarray([[0.3689, 0.1170, 0.2673],
                          [0.4699, 0.4387, 0.7470],
                          [0.1091, 0.8732, 0.5547],
                          [0.03815, 0.5743, 0.8828]])

        self.c = np.asarray([1., 1.2, 3., 3.2])

    def fun(self):
        x = self.symbols
        a = 0
        
        for i in range(4):
            b = 0
            for j in range(3):
                b += self.a[i, j]*((x[j] - self.p[i, j])**2)
            a += self.c[i]*exp(-b)
        return -a

class Hartmann6(Benchmark):

    r"""
    Hartmann6 objective function.

    This class defines the Hartmann6 [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Hartmann6}}(x) = -\sum\limits_{i=1}^{4} c_i
        e^{-\sum\limits_{j=1}^{n}a_{ij}(x_j - p_{ij})^2}


    Where, in this exercise:

    .. math::

        \begin{array}{l|cccccc|r}
        \hline
        i & &   &   a_{ij} &  &  & & c_i  \\
        \hline
        1 & 10.0  & 3.0  & 17.0 & 3.50  & 1.70  & 8.00  & 1.0 \\
        2 & 0.05  & 10.0 & 17.0 & 0.10  & 8.00  & 14.00 & 1.2 \\
        3 & 3.00  & 3.50 & 1.70 & 10.0  & 17.00 & 8.00  & 3.0 \\
        4 & 17.00 & 8.00 & 0.05 & 10.00 & 0.10  & 14.00 & 3.2 \\
        \hline
        \end{array}

        \newline
        \
        \newline

        \begin{array}{l|cccccr}
        \hline
        i &  &   & p_{ij} &  & & \\
        \hline
        1 & 0.1312 & 0.1696 & 0.5569 & 0.0124 & 0.8283 & 0.5886 \\
        2 & 0.2329 & 0.4135 & 0.8307 & 0.3736 & 0.1004 & 0.9991 \\
        3 & 0.2348 & 0.1451 & 0.3522 & 0.2883 & 0.3047 & 0.6650 \\
        4 & 0.4047 & 0.8828 & 0.8732 & 0.5743 & 0.1091 & 0.0381 \\
        \hline
        \end{array}


    with :math:`x_i \in [0, 1]` for :math:`i = 1, ..., 6`.

    *Global optimum*: :math:`f(x_i) = -3.32236801141551` for
    :math:`{x} = [0.20168952, 0.15001069, 0.47687398, 0.27533243, 0.31165162,
    0.65730054]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=6):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([0.0] * self.N, [1.0] * self.N))

        self.global_optimum = [[0.20168952, 0.15001069, 0.47687398, 0.27533243,
                                0.31165162, 0.65730054]]

        self.fglob = -3.32236801141551
        # self.complicated = True
        self.a = asarray([[10., 3., 17., 3.5, 1.7, 8.],
                          [0.05, 10., 17., 0.1, 8., 14.],
                          [3., 3.5, 1.7, 10., 17., 8.],
                          [17., 8., 0.05, 10., 0.1, 14.]])

        self.p = asarray([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                          [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                          [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.665],
                          [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])

        self.c = asarray([1.0, 1.2, 3.0, 3.2])

    def fun(self):
        x = self.symbols
        a = 0
        
        for i in range(4):
            b = 0
            for j in range(6):
                b += self.a[i, j]*((x[j] - self.p[i, j])**2)
            a += self.c[i]*exp(-b)
        return -a

class HimmelBlau(Benchmark):
    """
    HimmelBlau objective function.

    This class defines the HimmelBlau [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\\text{HimmelBlau}}({x}) = (x_1^2 + x_2 - 11)^2 + (x_1 + x_2^2 - 7)^2


    with :math:`x_i \\in [-6, 6]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [3, 2]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-6.0] * self.N, [6.0] * self.N))
        self.global_optimum = [[3.0, 2.0]]
        self.fglob = 0.0

    def fun(self):
        x = self.symbols
        return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

class Hosaki(Benchmark):

    r"""
    Hosaki objective function.

    This class defines the Hosaki [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Hosaki}}(x) = \left ( 1 - 8 x_1 + 7 x_1^2 - \frac{7}{3} x_1^3
        + \frac{1}{4} x_1^4 \right ) x_2^2 e^{-x_1}


    with :math:`x_i \in [0, 10]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = -2.3458115` for :math:`x = [4, 2]`.

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self._bounds = ([1, 5], [1, 3])
        self.custom_bounds = [(0, 5), (0, 5)]

        self.global_optimum = [[4, 2]]
        self.fglob = -2.3458115

    def fun(self):
        x = self.symbols
        val = (1 - 8 * x[0] + 7 * x[0] ** 2 - 7 / 3. * x[0] ** 3
               + 0.25 * x[0] ** 4)
        return val * x[1] ** 2 * exp(-x[1])
    
class JennrichSampson(Benchmark):
    """
    Jennrich-Sampson objective function.

    This class defines the Jennrich-Sampson [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\\text{JennrichSampson}}(x) = \\sum_{i=1}^{10} \\left [2 + 2i
        - (e^{ix_1} + e^{ix_2}) \\right ]^2


    with :math:`x_i \\in [-1, 1]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 124.3621824` for
    :math:`x = [0.257825, 0.257825]`.

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-1.0] * self.N, [1.0] * self.N))
        self.global_optimum = [[0.257825, 0.257825]]
        self.custom_bounds = [(-1, 0.34), (-1, 0.34)]
        self.fglob = 124.3621824

    def fun(self):
        x = self.symbols
        sum1 = 0
        for i in range(1, 11):
            sum1 += (2 + 2*i - (exp(i*x[0]) + exp(i*x[1])))**2
        return sum1

class Judge(Benchmark):

    """
    Judge objective function.

    This class defines the Judge [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Judge}}(x) = \sum_{i=1}^{20} 
        \left [ \left (x_1 + A_i x_2 + B x_2^2 \right ) - C_i \right ]^2


    Where, in this exercise:

    .. math::

        \begin{cases}
        C = [4.284, 4.149, 3.877, 0.533, 2.211, 2.389, 2.145,
        3.231, 1.998, 1.379, 2.106, 1.428, 1.011, 2.179, 2.858, 1.388, 1.651,
        1.593, 1.046, 2.152] \\
        A = [0.286, 0.973, 0.384, 0.276, 0.973, 0.543, 0.957, 0.948, 0.543,
             0.797, 0.936, 0.889, 0.006, 0.828, 0.399, 0.617, 0.939, 0.784,
             0.072, 0.889] \\
        B = [0.645, 0.585, 0.310, 0.058, 0.455, 0.779, 0.259, 0.202, 0.028,
             0.099, 0.142, 0.296, 0.175, 0.180, 0.842, 0.039, 0.103, 0.620,
             0.158, 0.704]
        \end{cases}


    with :math:`x_i \in [-10, 10]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x_i) = 16.0817307` for
    :math:`\mathbf{x} = [0.86479, 1.2357]`.

    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))

        self.global_optimum = [[0.86479, 1.2357]]
        self.custom_bounds = [(-2.0, 2.0), (-2.0, 2.0)]
        self.fglob = 16.0817307
        self.c = asarray([4.284, 4.149, 3.877, 0.533, 2.211, 2.389, 2.145,
                          3.231, 1.998, 1.379, 2.106, 1.428, 1.011, 2.179,
                          2.858, 1.388, 1.651, 1.593, 1.046, 2.152])

        self.a = asarray([0.286, 0.973, 0.384, 0.276, 0.973, 0.543, 0.957,
                          0.948, 0.543, 0.797, 0.936, 0.889, 0.006, 0.828,
                          0.399, 0.617, 0.939, 0.784, 0.072, 0.889])

        self.b = asarray([0.645, 0.585, 0.310, 0.058, 0.455, 0.779, 0.259,
                          0.202, 0.028, 0.099, 0.142, 0.296, 0.175, 0.180,
                          0.842, 0.039, 0.103, 0.620, 0.158, 0.704])

    def fun(self):
        x = self.symbols
        a = 0
        for i in range(20):
            a += (x[0] + x[1]*self.a[i] + (x[1]**2)*self.b[i] - self.c[i])**2
        return a

class Keane(Benchmark):

    r"""
    Keane objective function.

    This class defines the Keane [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Keane}}(x) = \frac{\sin^2(x_1 - x_2)\sin^2(x_1 + x_2)}
        {\sqrt{x_1^2 + x_2^2}}


    with :math:`x_i \in [0, 10]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0.0` for 
    :math:`x = [7.85396153, 7.85396135]`.

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO: Jamil #69, there is no way that the function can have a negative
    value.  Everything is squared.  I think that they have the wrong solution.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([1.0] * self.N, [10.0] * self.N))

        self.global_optimum = [[7.85396153, 7.85396135]]
        self.custom_bounds = [(-1, 0.34), (-1, 0.34)]
        self.fglob = 0.

    def fun(self):
        x = self.symbols
        val = sin(x[0] - x[1]) ** 2 * sin(x[0] + x[1]) ** 2
        return val / sqrt(x[0] ** 2 + x[1] ** 2)
    
class Langermann(Benchmark):
    """
    Langermann objective function.

    This class defines the Langermann [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\\text{Langermann}}(x) = - \\sum_{i=1}^{5} 
        \\frac{c_i \\cos\\left\\{\\pi \\left[\\left(x_{1}- a_i\\right)^{2}
        + \\left(x_{2} - b_i \\right)^{2}\\right]\\right\\}}{e^{\\frac{\\left( x_{1}
        - a_i\\right)^{2} + \\left( x_{2} - b_i\\right)^{2}}{\\pi}}}


    Where:

    .. math::

        \\begin{matrix}
        a = [3, 5, 2, 1, 7]\\\\
        b = [5, 2, 1, 4, 9]\\\\
        c = [1, 2, 5, 2, 3] \\\\
        \\end{matrix}


    Here :math:`x_i \\in [0, 10]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = -5.1621259`
    for :math:`x = [2.00299219, 1.006096]`

    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015

    TODO: Langermann from Gavana is _not the same_ as Jamil #68.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        # self._bounds = list(zip([0.0] * self.N, [10.0] * self.N))
        self._bounds = list(zip([0.0] * self.N, [3.0] * self.N))
        self.global_optimum = [[2.00299219, 1.006096]]
        self.fglob = -5.1621259

    def fun(self):
        x = self.symbols
        a = [3, 5, 2, 1, 7]
        b = [5, 2, 1, 4, 9]
        c = [1, 2, 5, 2, 3]
        sum1 = 0
        for i in range(5):
            arg = (x[0] - a[i])**2 + (x[1] - b[i])**2
            up = c[i] * cos(pi * (arg))
            down = exp(arg/pi)
            sum1 += up/down
        return -sum1

class Leon(Benchmark):
    """
    Leon objective function.

    This class defines the Leon [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\\text{Leon}}(\\mathbf{x}) = \\left(1 - x_{1}\\right)^{2} 
        + 100 \\left(x_{2} - x_{1}^{2} \\right)^{2}


    with :math:`x_i \\in [-1.2, 1.2]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [1, 1]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-1.2] * self.N, [1.2] * self.N))
        self.global_optimum = [[1 for _ in range(self.N)]]
        self.fglob = 0.0

    def fun(self):
        x = self.symbols
        return 100.0 * (x[1] - x[0] ** 2.0) ** 2.0 + (1 - x[0]) ** 2.0


class Levy03(Benchmark):
    """
    Levy 3 objective function.

    This class defines the Levy 3 [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\\text{Levy03}}(\\mathbf{x}) = \\sin^2(\\pi y_1)+\\sum_{i=1}^{n-1}(y_i-1)^2[1+10\\sin^2(\\pi y_{i+1})]+(y_n-1)^2

    Where, in this exercise:

    .. math::

        y_i=1+\\frac{x_i-1}{4}


    Here, :math:`n` represents the number of dimensions and :math:`x_i \\in [-10, 10]` for :math:`i=1,...,n`.

    *Global optimum*: :math:`f(x_i) = 0` for :math:`x_i = 1` for :math:`i=1,...,n`

    .. [1] Mishra, S. Global Optimization by Differential Evolution and
    Particle Swarm Methods: Evaluation on Some Benchmark Functions.
    Munich Personal RePEc Archive, 2006, 1005

    TODO: not clear what the Levy function definition is.  Gavana, Mishra,
    Adorio have different forms. Indeed Levy 3 docstring from Gavana
    disagrees with the Gavana code!  The following code is from the Mishra
    listing of Levy08.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))
        self.custom_bounds = [(-5, 5), (-5, 5)]
        self.global_optimum = [[1 for _ in range(self.N)]]
        self.fglob = 0.0

    def fun(self):
        x = self.symbols
        sum1 = 0
        y_1 = 1 + (x[0] - 1) / 4
        y_n = 1 + (x[-1] - 1) / 4
        for i in range(self.N - 1):
            y_i = 1 + (x[i] - 1) / 4
            sum1 += (y_i -1)**2 * (1 + 10*( sin(pi* y_i + 1) )**2) + (y_n - 1)**2 * (1 + (sin(2*pi*y_n))**2)
        return sin(pi * y_1) ** 2 + sum1

class Levy05(Benchmark):
    """
    Levy 5 objective function.

    This class defines the Levy 5 [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\\text{Levy05}}(\\mathbf{x}) = \\sum_{i=1}^{5} i \\cos \\left[(i-1)x_1 + i \\right] \\times \\sum_{j=1}^{5} j \\cos \\left[(j+1)x_2 + j \\right] + (x_1 + 1.42513)^2 + (x_2 + 0.80032)^2

    Here, :math:`n` represents the number of dimensions and :math:`x_i \\in [-10, 10]` for :math:`i=1,...,n`.

    *Global optimum*: :math:`f(x_i) = -176.1375779` for :math:`\\mathbf{x} = [-1.30685, -1.42485]`.

    .. [1] Mishra, S. Global Optimization by Differential Evolution and
    Particle Swarm Methods: Evaluation on Some Benchmark Functions.
    Munich Personal RePEc Archive, 2006, 1005
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))
        self.custom_bounds = ([-2.0, 2.0], [-2.0, 2.0])
        self.global_optimum = [[-1.30685, -1.42485]]
        self.fglob = -176.1375779

    def fun(self):
        x = self.symbols
        sum1 = 0
        sum2 = 0
        for i in range(1, 6):
            sum1 += i*cos((i - 1) * x[0] + i)
            sum2 += i*cos((i + 1) * x[1] + i)
        return sum1 * sum2 + (x[0] + 1.42513)**2 + (x[1] + 0.80032)**2

class Levy13(Benchmark):
    """
    Levy13 objective function.

    This class defines the Levy13 [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

        f_{\\text{Levy13}}(x) = \\left(x_{1} -1\\right)^{2} \\left[\\sin^{2}
        \\left(3 \\pi x_{2}\\right) + 1\\right] + \\left(x_{2} 
        - 1\\right)^{2} \\left[\\sin^{2}\\left(2 \\pi x_{2}\\right)
        + 1\\right] + \\sin^{2}\\left(3 \\pi x_{1}\\right)


    with :math:`x_i \\in [-10, 10]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [1, 1]`

    .. [1] Mishra, S. Some new test functions for global optimization and
    performance of repulsive particle swarm method.
    Munich Personal RePEc Archive, 2006, 2718
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-2.0] * self.N, [2.0] * self.N))
        self.custom_bounds = [(-2, 2), (-2, 2)]
        self.global_optimum = [[1 for _ in range(self.N)]]
        self.fglob = 0.0
        # if hasattr(self, '_bounds') and self._bounds and (not isinstance(self._bounds[0], (int, float))):
            # try:
                # lowers = [b[0] for b in self._bounds]
                # uppers = [b[1] for b in self._bounds]
                # self._bounds = [min(lowers), max(uppers)]
            # except:
                # pass

    def fun(self):
        x = self.symbols
        u = sin(3 * pi * x[0]) ** 2
        v = (x[0] - 1) ** 2 * (1 + sin(3 * pi * x[1]) ** 2)
        w = (x[1] - 1) ** 2 * (1 + sin(2 * pi * x[1]) ** 2)
        return u + v + w


class Matyas(Benchmark):
    """
    Matyas objective function.

    This class defines the Matyas [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

        f_{\\text{Matyas}}(x) = 0.26(x_1^2 + x_2^2) - 0.48 x_1 x_2


    with :math:`x_i \\in [-10, 10]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [0, 0]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))
        self.global_optimum = [[0 for _ in range(self.N)]]
        self.fglob = 0.0

    def fun(self):
        x = self.symbols
        return 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]

class Michalewicz(Benchmark):
    """
    Michalewicz objective function.

    This class defines the Michalewicz [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

       f_{\\text{Michalewicz}}(x) = - \\sum_{i=1}^{2} \\sin\\left(x_i\\right)
       \\sin^{2 m}\\left(\\frac{i x_i^{2}}{\\pi}\\right)


    Where, in this exercise, :math:`m = 10`.

    with :math:`x_i \\in [0, \\pi]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x_i) = -1.8013` for :math:`x = [0, 0]`

    .. [1] Adorio, E. MVF - "Multivariate Test Functions Library in C for
    Unconstrained Global Optimization", 2005

    TODO: could change dimensionality, but global minimum might change.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([0.0] * self.N, [pi] * self.N))
        self.global_optimum = [[2.20290555, 1.570796]]
        self.fglob = -1.8013

    def fun(self):
        x = self.symbols
        sum1 = 0
        m = 10
        for i in range(self.N):
            i_math = i + 1
            sum1 += sin(x[i]) * ( sin( (i_math*(x[i]**2)) / np.pi) )**(2*m)
        return -sum1

class McCormick(Benchmark):
    """
    McCormick objective function.

    This class defines the McCormick [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

       f_{\\text{McCormick}}(x) = - x_{1} + 2 x_{2} + \\left(x_{1}
       - x_{2}\\right)^{2} + \\sin\\left(x_{1} + x_{2}\\right) + 1

    with :math:`x_1 \\in [-1.5, 4]`, :math:`x_2 \\in [-3, 4]`.

    *Global optimum*: :math:`f(x) = -1.913222954981037` for
    :math:`x = [-0.5471975602214493, -1.547197559268372]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = [(-1.5, 4.0), (-3.0, 4.0)]
        self.global_optimum = [[-0.5471975602214493, -1.547197559268372]]
        self.fglob = -1.913222954981037

    def fun(self):
        x = self.symbols
        return sin(x[0] + x[1]) + (x[0] - x[1]) ** 2 - 1.5 * x[0] + 2.5 * x[1] + 1

class Mishra05(Benchmark):

    r"""
    Mishra 5 objective function.

    This class defines the Mishra 5 [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

       f_{\text{Mishra05}}(x) = \left [ \sin^2 ((\cos(x_1) + \cos(x_2))^2)
       + \cos^2 ((\sin(x_1) + \sin(x_2))^2) + x_1 \right ]^2 + 0.01(x_1 + x_2)


    with :math:`x_i \in [-10, 10]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = -0.119829` for :math:`x = [-1.98682, -10]`

    .. [1] Mishra, S. Global Optimization by Differential Evolution and
    Particle Swarm Methods: Evaluation on Some Benchmark Functions.
    Munich Personal RePEc Archive, 2006, 1005

    TODO Line 381 in paper
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))

        self.global_optimum = [[-1.98682, -10.0]]
        self.fglob = -1.019829519930646

    def fun(self):
        x = self.symbols

        return (0.01 * x[0] + 0.1 * x[1]
                + (sin((cos(x[0]) + cos(x[1])) ** 2) ** 2
                   + cos((sin(x[0]) + sin(x[1])) ** 2) ** 2 + x[0]) ** 2)

class Mishra07(Benchmark):

    r"""
    Mishra 7 objective function.

    This class defines the Mishra 7 [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

       f_{\text{Mishra07}}(x) = \left [\prod_{i=1}^{n} x_i - n! \right]^2


    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-10, 10]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = \sqrt{n}` 
    for :math:`i = 1, ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))
        self.custom_bounds = [(-2, 2), (-2, 2)]
        self.global_optimum = [[np.sqrt(self.N)
                               for i in range(self.N)]]
        self.fglob = 0.0
        self.change_dimensionality = True

    def fun(self):
        x = self.symbols
        a = 1
        for i in range(self.N):
            a *= x[i]
        return (a - factorial(self.N)) ** 2

class Mishra09(Benchmark):
    """
    Mishra 9 objective function.

    This class defines the Mishra 9 [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

       f_{\\text{Mishra09}}({x}) = \\left[ ab^2c + abc^2 + b^2
       + (x_1 + x_2 - x_3)^2 \\right]^2


    Where, in this exercise:

    .. math::

        \\begin{cases} a = 2x_1^3 + 5x_1x_2 + 4x_3 - 2x_1^2x_3 - 18 \\\\
        b = x_1 + x_2^3 + x_1x_2^2 + x_1x_3^2 - 22 \\\\
        c = 8x_1^2 + 2x_2x_3 + 2x_2^2 + 3x_2^3 - 52 \\end{cases}


    with :math:`x_i \\in [-10, 10]` for :math:`i = 1, 2, 3`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [1, 2, 3]`

    .. [1] Mishra, S. Global Optimization by Differential Evolution and
    Particle Swarm Methods: Evaluation on Some Benchmark Functions.
    Munich Personal RePEc Archive, 2006, 1005

    TODO Line 1103
    """

    def __init__(self, dimensions=3):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))
        self.global_optimum = [[1.0, 2.0, 3.0]]
        self.fglob = 0.0
        self.complicated=True

    def fun(self):
        x = self.symbols
        a = 2 * x[0] ** 3 + 5 * x[0] * x[1] + 4 * x[2] - 2 * x[0] ** 2 * x[2] - 18
        b = x[0] + x[1] ** 3 + x[0] * x[1] ** 2 + x[0] * x[2] ** 2 - 22.0
        c = 8 * x[0] ** 2 + 2 * x[1] * x[2] + 2 * x[1] ** 2 + 3 * x[1] ** 3 - 52
        return (a * c * b ** 2 + a * b * c ** 2 + b ** 2 + (x[0] + x[1] - x[2]) ** 2) ** 2
    
class Parsopoulos(Benchmark):
    """
    Parsopoulos objective function.

    This class defines the Parsopoulos [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

        f_{\\text{Parsopoulos}}(x) = \\cos(x_1)^2 + \\sin(x_2)^2


    with :math:`x_i \\in [-5, 5]` for :math:`i = 1, 2`.

    *Global optimum*: This function has infinite number of global minima in R2,
    at points :math:`\\left(k\\frac{\\pi}{2}, \\lambda \\pi \\right)`,
    where :math:`k = \\pm1, \\pm3, ...` and :math:`\\lambda = 0, \\pm1, \\pm2, ...`

    In the given domain problem, function has 12 global minima all equal to
    zero.

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-5.0] * self.N, [5.0] * self.N))
        self.global_optimum = [[pi / 2.0, pi]]
        self.fglob = 0

    def fun(self):
        x = self.symbols
        return cos(x[0]) ** 2.0 + sin(x[1]) ** 2.0

class Paviani(Benchmark):

    r"""
    Paviani objective function.

    This class defines the Paviani [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Paviani}}(x) = \sum_{i=1}^{10} \left[\log^{2}\left(10
        - x_i\right) + \log^{2}\left(x_i -2\right)\right]
        - \left(\prod_{i=1}^{10} x_i^{10} \right)^{0.2}


    with :math:`x_i \in [2.001, 9.999]` for :math:`i = 1, ... , 10`.

    *Global optimum*: :math:`f(x_i) = -45.7784684040686` for
    :math:`x_i = 9.350266` for :math:`i = 1, ..., 10`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO: think Gavana web/code definition is wrong because final product term
    shouldn't raise x to power 10.
    """

    def __init__(self, dimensions=10):
        Benchmark.__init__(self, dimensions)
        # self.complicated = True
        self._bounds = list(zip([4.5] * self.N, [9.5] * self.N))

        self.global_optimum = [[9.350266 for _ in range(self.N)]]
        self.fglob = -45.7784684040686

    def fun(self):
        x = self.symbols
        a = 0
        b = 1
        for i in range(self.N):
            a += log(x[i] - 2)**2 + log(10 - x[i])**2
            b *= x[i]
        return a - b ** 0.2

class PermFunction02(Benchmark):

    r"""
    PermFunction 2 objective function.

    This class defines the Perm Function 2 [1]_ global optimization problem. This is
    a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{PermFunction02}}(x) = \sum_{k=1}^n \left\{ \sum_{j=1}^n (j
        + \beta) \left[ \left(x_j^k - {\frac{1}{j}}^{k} \right )
        \right] \right\}^2


    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-n, n+1]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = \frac{1}{i}`
    for :math:`i = 1, ..., n`

    .. [1] Mishra, S. Global Optimization by Differential Evolution and
    Particle Swarm Methods: Evaluation on Some Benchmark Functions.
    Munich Personal RePEc Archive, 2006, 1005

    TODO: line 582
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-self.N] * self.N,
                           [self.N + 1] * self.N))
        self.custom_bounds = ([0, 1.5], [0, 1.0])

        self.global_optimum = [1. / np.arange(1, self.N + 1)]
        self.fglob = 0.0
        self.change_dimensionality = True

    def fun(self):
        x = self.symbols
        a = 0
        beta = 10
        for k in range(self.N):
            k_math = k + 1
            b = 0
            for j in range(self.N):
                j_math = j + 1
                b += (j_math + beta)*(x[j]**k_math - (1/j_math)**k_math)
            a += b**2
        return a

class Pinter(Benchmark):

    r"""
    Pinter objective function.

    This class defines the Pinter [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

       f_{\text{Pinter}}(x) = \sum_{i=1}^n ix_i^2 + \sum_{i=1}^n 20i
       \sin^2 A + \sum_{i=1}^n i \log_{10} (1 + iB^2)


    Where, in this exercise:

    .. math::

        \begin{cases}
        A = x_{i-1} \sin x_i + \sin x_{i+1} \\
        B = x_{i-1}^2 - 2x_i + 3x_{i + 1} - \cos x_i + 1\\
        \end{cases}

    Where :math:`x_0 = x_n` and :math:`x_{n + 1} = x_1`.

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-10, 10]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for
    :math:`i = 1, ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-5.0] * self.N, [5.0] * self.N))

        self.global_optimum = [[0.0 for _ in range(self.N)]]
        self.fglob = 0.0
        self.change_dimensionality = True


    def fun(self):
        x = self.symbols
        xx = list(sym.zeros(1, self.N + 2))
        xx[1: - 1] = x
        xx[0] = x[-1]
        xx[-1] = x[0]
        a = 0
        b = 0
        c = 0
        for i in range(1, self.N):
            i_math = i
            A = (xx[i-1]*sin(xx[i]) + sin(xx[i + 1]))
            B = (xx[i - 1]**2 - 2*xx[i] + 3*xx[i + 1] - cos(xx[i]) + 1)
            a += i_math*xx[i]**2
            b += 20*i_math*(sin(A))**2
            c += i_math*log(1 + i_math*B**2, 10)
        return a + b + c

class Powell(Benchmark):
    """
    Powell objective function.

    This class defines the Powell [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\\text{Powell}}(x) = (x_3+10x_1)^2 + 5(x_2-x_4)^2 + (x_1-2x_2)^4
        + 10(x_3-x_4)^4


    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \\in [-4, 5]` for :math:`i = 1, ..., 4`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for
    :math:`i = 1, ..., 4`

    ..[1] Powell, M. An iterative method for finding stationary values of a
    function of several variables Computer Journal, 1962, 5, 147-151
    """

    def __init__(self, dimensions=4):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-4.0] * self.N, [5.0] * self.N))
        self.global_optimum = [[0, 0, 0, 0]]
        self.fglob = 0
        self.complicated = True

    def fun(self):
        x = self.symbols
        return (x[0] + 10 * x[1]) ** 2 + 5 * (x[2] - x[3]) ** 2 + (x[1] - 2 * x[2]) ** 4 + 10 * (x[0] - x[3]) ** 4

class PowerSum(Benchmark):

    r"""
    Power sum objective function.

    This class defines the Power Sum global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{PowerSum}}(x) = \sum_{k=1}^n\left[\left(\sum_{i=1}^n x_i^k
        \right) - b_k \right]^2

    Where, in this exercise, :math:`b = [8, 18, 44, 114]`

    Here, :math:`x_i \in [0, 4]` for :math:`i = 1, ..., 4`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [1, 2, 2, 3]`

    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015

    """

    def __init__(self, dimensions=4):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([0.0] * self.N,
                           [4.0] * self.N))

        self.global_optimum = [[1.0, 2.0, 2.0, 3.0]]
        self.fglob = 0.0
        self.complicated = True

    def fun(self):
        x = self.symbols
        b = [8.0, 18.0, 44.0, 114.0]
        a = 0
        for k in range(self.N):
            k_m = k + 1
            c = 0
            for i in range(self.N):
                c += x[i]**k_m
            a += (c - b[k])**2
        
        return a
    
class Price02(Benchmark):
    """
    Price 2 objective function.

    This class defines the Price 2 [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

       f_{\\text{Price02}}(x) = 1 + \\sin^2(x_1) + \\sin^2(x_2)
       - 0.1e^{(-x_1^2 - x_2^2)}


    with :math:`x_i \\in [-10, 10]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0.9` for :math:`x_i = [0, 0]`

    .. [1] Price, W. A controlled random search procedure for global
    optimisation Computer Journal, 1977, 20, 367-370
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))
        self.global_optimum = [[0.0, 0.0]]
        self.fglob = 0.9

    def fun(self):
        x = self.symbols
        return 1 + (sin(x[0]))**2 + (sin(x[1]))**2 - 0.1 * exp(-(x[0]**2 + x[1]**2))

class Price03(Benchmark):
    """
    Price 3 objective function.

    This class defines the Price 3 [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

       f_{\\text{Price03}}(x) = 100(x_2 - x_1^2)^2 + \\left[6.4(x_2 - 0.5)^2
       - x_1 - 0.6 \\right]^2

    with :math:`x_i \\in [-50, 50]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [-5, -5]`,
    :math:`x = [-5, 5]`, :math:`x = [5, -5]`, :math:`x = [5, 5]`.

    .. [1] Price, W. A controlled random search procedure for global
    optimisation Computer Journal, 1977, 20, 367-370

    TODO Jamil #96 has an erroneous factor of 6 in front of the square brackets
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-50.0] * self.N, [50.0] * self.N))
        self.custom_bounds = ([-5, 5], [-5, 5])
        self.global_optimum = [[1.0, 1.0]]
        self.fglob = 0.0

    def fun(self):
        x = self.symbols
        return 100 * (x[1] - x[0] ** 2) ** 2 + (6.4 * (x[1] - 0.5) ** 2 - x[0] - 0.6) ** 2
    
class Price04(Benchmark):
    """
    Price 4 objective function.

    This class defines the Price 4 [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\\text{Price04}}(x) = (2 x_1^3 x_2 - x_2^3)^2
        + (6 x_1 - x_2^2 + x_2)^2

    with :math:`x_i \\in [-50, 50]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [0, 0]`,
    :math:`x = [2, 4]` and :math:`x = [1.464, -2.506]`

    .. [1] Price, W. A controlled random search procedure for global
    optimisation Computer Journal, 1977, 20, 367-370
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([0.0] * self.N, [5.0] * self.N))
        self.custom_bounds = ([-5, 5], [-5, 5])
        self.global_optimum = [[2.0, 4.0]]
        self.fglob = 0.0

    def fun(self):
        x = self.symbols
        return (2.0 * x[1] * x[0] ** 3.0 - x[1] ** 3.0) ** 2.0 + (6.0 * x[0] - x[1] ** 2.0 + x[1]) ** 2.0

class Qing(Benchmark):
    """
    Qing objective function.

    This class defines the Qing [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

        f_{	ext{Qing}}(x) = \\sum_{i=1}^{n} (x_i^2 - i)^2


    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \\in [-500, 500]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = \\pm \\sqrt(i)` for
    :math:`i = 1, ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-500.0] * self.N, [500.0] * self.N))
        self.custom_bounds = [(-2, 2), (-2, 2)]
        self.global_optimum = [[np.sqrt(_) for _ in range(1, self.N + 1)]]
        self.fglob = 0
        self.change_dimensionality = True

    def fun(self):
        x = self.symbols
        sum1 = 0
        for i in range(self.N):
            sum1 += (x[i]**2 - (i + 1))**2
        return sum1
    
class Quadratic(Benchmark):
    """
    Quadratic objective function.

    This class defines the Quadratic [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{	ext{Quadratic}}(x) = -3803.84 - 138.08x_1 - 232.92x_2 + 128.08x_1^2
        + 203.64x_2^2 + 182.25x_1x_2


    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \\in [-10, 10]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = -3873.72418` for
    :math:`x = [0.19388, 0.48513]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))
        self.custom_bounds = [(0, 1), (0, 1)]
        self.global_optimum = [[0.19388, 0.48513]]
        self.fglob = -3873.72418
        self.change_dimensionality = False

    def fun(self):
        x = self.symbols
        return -3803.84 - 138.08 * x[0] - 232.92 * x[1] + 128.08 * x[0] ** 2.0 + 203.64 * x[1] ** 2.0 + 182.25 * x[0] * x[1]

    
class Rastrigin(Benchmark):
    """
    Rastrigin objective function.

    This class defines the Rastrigin [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

        f_{\\text{Rastrigin}}(x) = 10n \\sum_{i=1}^n \\left[ x_i^2
        - 10 \\cos(2\\pi x_i) \\right]

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \\in [-5.12, 5.12]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for
    :math:`i = 1, ..., n`

    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-5.12] * self.N, [5.12] * self.N))
        self.global_optimum = [[0 for _ in range(self.N)]]
        self.fglob = 0.0
        self.change_dimensionality = True

    def fun(self):
        x = self.symbols
        return 10.0 * self.N + sum([x_i ** 2.0 - 10.0 * cos(2.0 * pi * x_i) for x_i in x])

class Ratkowsky02(Benchmark):

    r"""
    Ratkowsky02 objective function.

    This class defines the Ratkowsky 2 [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::
        f_{\text{Ratkowsky02}}(x) = \sum_{m=1}^{9}(a_m - x[0] / (1 + exp(x[1]
        - b_m x[2]))^2

    where
    
    .. math::
        
        \begin{cases}
        a=[8.93, 10.8, 18.59, 22.33, 39.35, 56.11, 61.73, 64.62, 67.08]\\
        b=[9., 14., 21., 28., 42., 57., 63., 70., 79.]\\
        \end{cases}       
        
        
    Here :math:`x_1 \in [1, 100]`, :math:`x_2 \in [0.1, 5]` and
    :math:`x_3 \in [0.01, 0.5]`

    *Global optimum*: :math:`f(x) = 8.0565229338` for
    :math:`x = [7.2462237576e1, 2.6180768402, 6.7359200066e-2]`

    .. [1] http://www.itl.nist.gov/div898/strd/nls/data/ratkowsky2.shtml
    """

    def __init__(self, dimensions=3):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([10, 0.5, 0.01],
                           [200, 5., 0.5]))
        self.global_optimum = [[7.2462237576e1, 2.6180768402, 6.7359200066e-2]]
        self.fglob = 8.0565229338
        self.a = asarray([8.93, 10.8, 18.59, 22.33, 39.35, 56.11, 61.73, 64.62,
                          67.08])
        self.b = asarray([9., 14., 21., 28., 42., 57., 63., 70., 79.])
        self.compilcated = True

    def fun(self):
        x = self.symbols
        a = 0
        for i in range(len(self.a)):
            a += (self.a[i] - (x[0])/(1 + exp(x[1] - self.b[i]*x[2])))**2
        return a

class Ripple01(Benchmark):

    r"""
    Ripple 1 objective function.

    This class defines the Ripple 1 [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Ripple01}}(x) = \sum_{i=1}^2 -e^{-2 \log 2 
        (\frac{x_i-0.1}{0.8})^2} \left[\sin^6(5 \pi x_i)
        + 0.1\cos^2(500 \pi x_i) \right]


    with :math:`x_i \in [0, 1]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = -2.2` for :math:`x_i = 0.1` for
    :math:`i = 1, 2`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([0.0] * self.N, [1.0] * self.N))

        self.global_optimum = [[0.1 for _ in range(self.N)]]
        self.fglob = -2.2
        self.complicated = True
    def fun(self):
        x = self.symbols
        a = 0
        for i in range(self.N):
            u = -2.0 * log(2.0) * ((x[i] - 0.1) / 0.8) ** 2.0
            v = sin(5.0 * pi * x[i]) ** 6.0 + 0.1 * cos(500.0 * pi * x[i]) ** 2.0
            a += -exp(u)*v
        return a

class Ripple25(Benchmark):

    r"""
    Ripple 25 objective function.

    This class defines the Ripple 25 [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Ripple25}}(x) = \sum_{i=1}^2 -e^{-2 
        \log 2 (\frac{x_i-0.1}{0.8})^2}
        \left[\sin^6(5 \pi x_i) \right]


    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [0, 1]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = -2` for :math:`x_i = 0.1` for
    :math:`i = 1, ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([0.0] * self.N, [1.0] * self.N))

        self.global_optimum = [[0.1 for _ in range(self.N)]]
        self.fglob = -2.0
        
    def fun(self):
        x = self.symbols
        a = 0
        for i in range(self.N):
            u = -2.0 * log(2.0) * ((x[i] - 0.1) / 0.8) ** 2.0
            v = sin(5.0 * pi * x[i]) ** 6.0
            a += -exp(u)*v
        return a
    
class Rosenbrock(Benchmark):
    """
    Rosenbrock objective function.

    This class defines the Rosenbrock [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

       f_{\\text{Rosenbrock}}(x) = \\sum_{i=1}^{n-1} [100(x_i^2
       - x_{i+1})^2 + (x_i - 1)^2]


    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \\in [-5, 10]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 1` for
    :math:`i = 1, ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-5.0] * self.N, [10.0] * self.N))
        self.custom_bounds = [(-2, 2), (-2, 2)]
        self.global_optimum = [[1 for _ in range(self.N)]]
        self.fglob = 0.0
        self.change_dimensionality = True

    def fun(self):
        x = self.symbols
        sum1 = 0
        for i in range(self.N - 1):
            sum1 += 100*(x[i]**2 - x[i + 1])**2 + (x[i] - 1)**2
        return sum1

class RosenbrockModified(Benchmark):
    """
    Modified Rosenbrock objective function.

    This class defines the Modified Rosenbrock [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

       f_{\\text{RosenbrockModified}}(x) = 74 + 100(x_2 - x_1^2)^2
       + (1 - x_1)^2 - 400 e^{-\\frac{(x_1+1)^2 + (x_2 + 1)^2}{0.1}}

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \\in [-2, 2]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 34.04024310` for
    :math:`x = [-0.90955374, -0.95057172]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO: We have different global minimum compared to Jamil #106. This is
    possibly because of the (1-x) term is using the wrong parameter.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-2.0] * self.N, [2.0] * self.N))
        self.custom_bounds = ([-1.0, 0.5], [-1.0, 1.0])
        self.global_optimum = [[-0.90955374, -0.95057172]]
        self.fglob = 34.040243106640844

    def fun(self):
        x = self.symbols
        a = 74 + 100.0 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
        a -= 400 * exp(-((x[0] + 1.0) ** 2 + (x[1] + 1.0) ** 2) / 0.1)
        return a
    
class RotatedEllipse01(Benchmark):
    """
    Rotated Ellipse 1 objective function.

    This class defines the Rotated Ellipse 1 [1]_ global optimization problem. This
    is a unimodal minimization problem defined as follows:

    .. math::

       f_{\\text{RotatedEllipse01}}(x) = 7x_1^2 - 6 \\sqrt{3} x_1x_2 + 13x_2^2

    with :math:`x_i \\in [-500, 500]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [0, 0]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-500.0] * self.N, [500.0] * self.N))
        self.custom_bounds = ([-2.0, 2.0], [-2.0, 2.0])
        self.global_optimum = [[0.0, 0.0]]
        self.fglob = 0.0

    def fun(self):
        x = self.symbols
        return 7.0 * x[0] ** 2.0 - 6.0 * sqrt(3) * x[0] * x[1] + 13 * x[1] ** 2.0
    
class RotatedEllipse02(Benchmark):
    """
    Rotated Ellipse 2 objective function.

    This class defines the Rotated Ellipse 2 [1]_ global optimization problem. This
    is a unimodal minimization problem defined as follows:

    .. math::

       f_{\\text{RotatedEllipse02}}(x) = x_1^2 - x_1 x_2 + x_2^2

    with :math:`x_i \\in [-500, 500]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [0, 0]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-500.0] * self.N, [500.0] * self.N))
        self.custom_bounds = ([-2.0, 2.0], [-2.0, 2.0])
        self.global_optimum = [[0.0, 0.0]]
        self.fglob = 0.0

    def fun(self):
        x = self.symbols
        return x[0] ** 2.0 - x[0] * x[1] + x[1] ** 2.0

class Sargan(Benchmark):
    """
    Sargan objective function.

    This class defines the Sargan [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\\text{Sargan}}(x) = \\sum_{i=1}^{n} n \\left (x_i^2
        + 0.4 \\sum_{i \\neq j}^{n} x_ix_j \\right)

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \\in [-100, 100]` for
    :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for
    :math:`i = 1, ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-100.0] * self.N, [100.0] * self.N))
        self.custom_bounds = [(-5, 5), (-5, 5)]
        self.global_optimum = [[0.0 for _ in range(self.N)]]
        self.fglob = 0.0
        self.change_dimensionality = True

    def fun(self):
        x = self.symbols
        sum1 = 0
        for i in range(self.N):
            sum2 = 0
            for j in range(self.N):
                if i != j:
                    sum2 += x[i]*x[j]
            sum1 += self.N * (x[i]**2 + 0.4 * sum2)
        return sum1

    def system(self):
        return derive_matrix(sym.Matrix([self.fun()]), self.symbols)

class Schaffer01(Benchmark):
    """
    Schaffer 1 objective function.

    This class defines the Schaffer 1 [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\\text{Schaffer01}}(x) = 0.5 + \\frac{\\sin^2 (x_1^2 + x_2^2)^2 - 0.5}
        {1 + 0.001(x_1^2 + x_2^2)^2}

    with :math:`x_i \\in [-100, 100]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [0, 0]` for
    :math:`i = 1, 2`

    .. [1] Mishra, S. Some new test functions for global optimization and
    performance of repulsive particle swarm method.
    Munich Personal RePEc Archive, 2006, 2718
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        # self._bounds = list(zip([-100.0] * self.N, [100.0] * self.N))
        self._bounds = list(zip([-2.0] * self.N, [2.0] * self.N))
        self.custom_bounds = [(-2, 2), (-2, 2)]
        self.global_optimum = [[0.0 for _ in range(self.N)]]
        self.fglob = 0.0

    def fun(self):
        x = self.symbols
        return 0.5 + ((sin(x[0]**2 - x[1]**2))**2 - 0.5)/ ((1 + 0.001*(x[0]**2 + x[1]**2))**2)

class Schwefel02(Benchmark):
    """
    Schwefel 2 objective function.

    This class defines the Schwefel 2 [1]_ global optimization problem. This
    is a unimodal minimization problem defined as follows:

    .. math::

        f_{\\text{Schwefel02}}(x) = \\sum_{i=1}^n \\left(\\sum_{j=1}^i 
        x_i \\right)^2


    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \\in [-100, 100]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for
    :math:`i = 1, ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-100.0] * self.N, [100.0] * self.N))
        self.custom_bounds = ([-4.0, 4.0], [-4.0, 4.0])
        self.global_optimum = [[0.0 for _ in range(self.N)]]
        self.fglob = 0.0
        self.change_dimensionality = True

    def fun(self):
        x = self.symbols
        mat = np.repeat(np.atleast_2d(x), self.N, axis=0)
        inner = np.sum(np.tril(mat), axis=1)
        return sum(inner ** 2)

class Schwefel04(Benchmark):
    """
    Schwefel 4 objective function.

    This class defines the Schwefel 4 [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\\text{Schwefel04}}(x) = \\sum_{i=1}^n \\left[(x_i - 1)^2
        + (x_1 - x_i^2)^2 \\right]


    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \\in [0, 10]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for:math:`x_i = 1` for
    :math:`i = 1, ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([0.0] * self.N, [10.0] * self.N))
        self.custom_bounds = ([0.0, 2.0], [0.0, 2.0])
        self.global_optimum = [[1.0 for _ in range(self.N)]]
        self.fglob = 0.0
        self.change_dimensionality = True
        # if hasattr(self, '_bounds') and self._bounds and (not isinstance(self._bounds[0], (int, float))):
            # try:
                # lowers = [b[0] for b in self._bounds]
                # uppers = [b[1] for b in self._bounds]
                # self._bounds = [min(lowers), max(uppers)]
            # except:
                # pass

    def fun(self):
        x = self.symbols
        return sum([(x_i - 1)**2 + (x[0] - x_i**2)**2 for x_i in x])

class Schwefel36(Benchmark):
    """
    Schwefel 36 objective function.

    This class defines the Schwefel 36 [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\\text{Schwefel36}}(x) = -x_1x_2(72 - 2x_1 - 2x_2)


    with :math:`x_i \\in [0, 500]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = -3456` for :math:`x = [12, 12]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([0.0] * self.N, [500.0] * self.N))
        self.custom_bounds = ([0.0, 20.0], [0.0, 20.0])
        self.global_optimum = [[12.0, 12.0]]
        self.fglob = -3456.0

    def fun(self):
        x = self.symbols
        return -x[0] * x[1] * (72 - 2 * x[0] - 2 * x[1])

class Shekel05(Benchmark):

    r"""
    Shekel 5 objective function.

    This class defines the Shekel 5 [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Shekel05}}(x) = -\sum_{i=1}^{m} \frac{1}{c_{i}
        + \sum_{j=1}^{n} (x_{j} - a_{ij})^2 }`

    Where, in this exercise:

    .. math::

        a = 
        \begin{bmatrix}
        4.0 & 4.0 & 4.0 & 4.0 \\ 1.0 & 1.0 & 1.0 & 1.0 \\
        8.0 & 8.0 & 8.0 & 8.0 \\ 6.0 & 6.0 & 6.0 & 6.0 \\
        3.0 & 7.0 & 3.0 & 7.0 
        \end{bmatrix}
    .. math::

        c = \begin{bmatrix} 0.1 \\ 0.2 \\ 0.2 \\ 0.4 \\ 0.4 \end{bmatrix}

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [0, 10]` for :math:`i = 1, ..., 4`.

    *Global optimum*: :math:`f(x) = -10.15319585` for :math:`x_i = 4` for
    :math:`i = 1, ..., 4`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO: this is a different global minimum compared to Jamil#130.  The
    minimum is found by doing lots of optimisations. The solution is supposed
    to be at [4] * N, is there any numerical overflow?
    """

    def __init__(self, dimensions=4):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([0.0] * self.N, [10.0] * self.N))

        self.global_optimum = [[4.00003715092,
                                4.00013327435,
                                4.00003714871,
                                4.0001332742]]
        self.fglob = -10.1531996791
        self.A = asarray([[4.0, 4.0, 4.0, 4.0],
                          [1.0, 1.0, 1.0, 1.0],
                          [8.0, 8.0, 8.0, 8.0],
                          [6.0, 6.0, 6.0, 6.0],
                          [3.0, 7.0, 3.0, 7.0]])

        self.C = asarray([0.1, 0.2, 0.2, 0.4, 0.4])

    def fun(self):
        x = self.symbols
        a = 0
        for i in range(5):
            den = self.C[i]
            for j in range(4):
                den += (x[j] - self.A[i, j])**2
            a += 1/den
        return -a

class Shekel07(Benchmark):

    r"""
    Shekel 7 objective function.

    This class defines the Shekel 7 [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Shekel07}}(x) = -\sum_{i=1}^{m} \frac{1}{c_{i}
                                 + \sum_{j=1}^{n} (x_{j} - a_{ij})^2 }`

    Where, in this exercise:

    .. math::

        a =
        \begin{bmatrix}
        4.0 & 4.0 & 4.0 & 4.0 \\ 1.0 & 1.0 & 1.0 & 1.0 \\
        8.0 & 8.0 & 8.0 & 8.0 \\ 6.0 & 6.0 & 6.0 & 6.0 \\
        3.0 & 7.0 & 3.0 & 7.0 \\ 2.0 & 9.0 & 2.0 & 9.0 \\
        5.0 & 5.0 & 3.0 & 3.0
        \end{bmatrix}


    .. math::

        c =
        \begin{bmatrix}
        0.1 \\ 0.2 \\ 0.2 \\ 0.4 \\ 0.4 \\ 0.6 \\ 0.3 
        \end{bmatrix}


    with :math:`x_i \in [0, 10]` for :math:`i = 1, ..., 4`.

    *Global optimum*: :math:`f(x) = -10.4028188` for :math:`x_i = 4` for
    :math:`i = 1, ..., 4`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO: this is a different global minimum compared to Jamil#131. This
    minimum is obtained after running lots of minimisations!  Is there any
    numerical overflow that causes the minimum solution to not be [4] * N?
    """

    def __init__(self, dimensions=4):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([0.0] * self.N, [10.0] * self.N))

        self.global_optimum = [[4.00057291078,
                                4.0006893679,
                                3.99948971076,
                                3.99960615785]]
        self.fglob = -10.4029405668
        self.A = asarray([[4.0, 4.0, 4.0, 4.0],
                          [1.0, 1.0, 1.0, 1.0],
                          [8.0, 8.0, 8.0, 8.0],
                          [6.0, 6.0, 6.0, 6.0],
                          [3.0, 7.0, 3.0, 7.0],
                          [2.0, 9.0, 2.0, 9.0],
                          [5.0, 5.0, 3.0, 3.0]])

        self.C = asarray([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3])

    def fun(self):
        x = self.symbols
        a = 0
        for i in range(7):
            den = self.C[i]
            for j in range(4):
                den += (x[j] - self.A[i, j])**2 
            a += 1/den
        return -a

class Shekel10(Benchmark):

    r"""
    Shekel 10 objective function.

    This class defines the Shekel 10 [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

       f_{\text{Shekel10}}(x) = -\sum_{i=1}^{m} \frac{1}{c_{i} 
                                + \sum_{j=1}^{n} (x_{j} - a_{ij})^2 }`

    Where, in this exercise:

    .. math::

        a =
        \begin{bmatrix}
        4.0 & 4.0 & 4.0 & 4.0 \\ 1.0 & 1.0 & 1.0 & 1.0 \\
        8.0 & 8.0 & 8.0 & 8.0 \\ 6.0 & 6.0 & 6.0 & 6.0 \\
        3.0 & 7.0 & 3.0 & 7.0 \\ 2.0 & 9.0 & 2.0 & 9.0 \\
        5.0 & 5.0 & 3.0 & 3.0 \\ 8.0 & 1.0 & 8.0 & 1.0 \\
        6.0 & 2.0 & 6.0 & 2.0 \\ 7.0 & 3.6 & 7.0 & 3.6
        \end{bmatrix}


    .. math::

        c =
        \begin{bmatrix}
        0.1 \\ 0.2 \\ 0.2 \\ 0.4 \\ 0.4 \\ 0.6 \\ 0.3 \\ 0.7 \\ 0.5 \\ 0.5
        \end{bmatrix}


    with :math:`x_i \in [0, 10]` for :math:`i = 1, ..., 4`.

    *Global optimum*: :math:`f(x) = -10.5362837` for :math:`x_i = 4` for
    :math:`i = 1, ..., 4`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO Found a lower global minimum than Jamil#132... Is this numerical overflow?
    """

    def __init__(self, dimensions=4):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([0.0] * self.N, [10.0] * self.N))

        self.global_optimum = [[4.0007465377266271,
                                4.0005929234621407,
                                3.9996633941680968,
                                3.9995098017834123]]
        self.fglob = -10.536409816692023
        self.A = asarray([[4.0, 4.0, 4.0, 4.0],
                          [1.0, 1.0, 1.0, 1.0],
                          [8.0, 8.0, 8.0, 8.0],
                          [6.0, 6.0, 6.0, 6.0],
                          [3.0, 7.0, 3.0, 7.0],
                          [2.0, 9.0, 2.0, 9.0],
                          [5.0, 5.0, 3.0, 3.0],
                          [8.0, 1.0, 8.0, 1.0],
                          [6.0, 2.0, 6.0, 2.0],
                          [7.0, 3.6, 7.0, 3.6]])

        self.C = asarray([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])

    def fun(self):
        x = self.symbols
        a = 0
        for i in range(10):
            den = self.C[i]
            for j in range(4):
                den += (x[j] - self.A[i, j])**2
            a += 1/den
        return -a

class Shubert01(Benchmark):
    """
    Shubert 1 objective function.

    This class defines the Shubert 1 [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\\text{Shubert01}}(x) = \\prod_{i=1}^{n}\\left(\\sum_{j=1}^{5}
                                  cos(j+1)x_i+j \\right )

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \\in [-10, 10]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = -186.7309` for
    :math:`x = [-7.0835, 4.8580]` (and many others).

    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015

    TODO: Jamil#133 is missing a prefactor of j before the cos function.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))
        self.custom_bounds = ([-8.0, -4.0], [2.0, 6.0])
        self.global_optimum = [[-7.0835, 4.858]]
        self.fglob = -186.7309
        self.change_dimensionality = True

    def fun(self):
        x = self.symbols
        sum1 = 0
        sum2 = 0
        for i in range(1, 6):
            sum1 += i * cos((i + 1)*x[0] + i)
            sum2 += i * cos((i + 1)*x[1] + i)
        return sum1 * sum2

class Shubert03(Benchmark):

    r"""
    Shubert 3 objective function.

    This class defines the Shubert 3 [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Shubert03}}(x) = \sum_{i=1}^n \sum_{j=1}^5 -j 
                                  \sin((j+1)x_i + j)

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-10, 10]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = -24.062499` for
    :math:`x = [5.791794, 5.791794]` (and many others).

     .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015

    TODO: Jamil#134 has wrong global minimum value, and is missing a minus sign
    before the whole thing.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))

        self.global_optimum = [[5.791794, 5.791794]]
        self.fglob = -24.062499

        self.change_dimensionality = True

    def fun(self):
        x = self.symbols
        sum1 = 0
        for i in range(self.N):
            for j in range(1, 6):
                sum1 += -j*sin((j + 1)*x[i] + j)
        return sum1

class Shubert04(Benchmark):
    """
    Shubert 4 objective function.

    This class defines the Shubert 4 [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\\text{Shubert04}}(x) = \\left(\\sum_{i=1}^n \\sum_{j=1}^5 -j
                                  \\cos ((j+1)x_i + j)\\right)

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \\in [-10, 10]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = -29.016015` for
    :math:`x = [-0.80032121, -7.08350592]` (and many others).

    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015

    TODO: Jamil#135 has wrong global minimum value, and is missing a minus sign
    before the whole thing.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))
        self.global_optimum = [[-0.80032121, -7.08350592]]
        self.fglob = -29.016015
        self.change_dimensionality = True

    def fun(self):
        x = self.symbols
        sum1 = 0
        for i in range(self.N):
            for j in range(1, 6):
                sum1 += -j*cos((j + 1)*x[i] + j)
        return sum1

class SineEnvelope(Benchmark):

    r"""
    SineEnvelope objective function.

    This class defines the SineEnvelope [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{SineEnvelope}}(x) = -\sum_{i=1}^{n-1}\left[\frac{\sin^2(
                                       \sqrt{x_{i+1}^2+x_{i}^2}-0.5)}
                                       {(0.001(x_{i+1}^2+x_{i}^2)+1)^2}
                                       + 0.5\right]

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-100, 100]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for
    :math:`i = 1, ..., n`

    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015

    TODO: Jamil #136
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-10.0] * self.N,
                           [10.0] * self.N))
        self.custom_bounds = [(-20, 20), (-20, 20)]
        self.global_optimum = [[0 for _ in range(self.N)]]
        self.fglob = 0.0
        self.change_dimensionality = True

    def fun(self):
        x = self.symbols
        a = 0
        for i in range(len(x) - 1):
            term1 = sin(x[i] + x[i+1])
            term2 = exp(-((x[i] - x[i+1])**2) / np.pi)
            a += term1 * term2
        return -a
    
class SixHumpCamel(Benchmark):
    """
    Six Hump Camel objective function.

    This class defines the Six Hump Camel [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\\text{SixHumpCamel}}(x) = 4x_1^2+x_1x_2-4x_2^2-2.1x_1^4+
                                    4x_2^4+\\frac{1}{3}x_1^6

    with :math:`x_i \\in [-5, 5]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = -1.031628453489877` for
    :math:`x = [0.08984201368301331 , -0.7126564032704135]` or 
    :math:`x = [-0.08984201368301331, 0.7126564032704135]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-5.0] * self.N, [5.0] * self.N))
        self.custom_bounds = [(-2, 2), (-1.5, 1.5)]
        self.global_optimum = [(0.08984201368301331, -0.7126564032704135), (-0.08984201368301331, 0.7126564032704135)]
        self.fglob = -1.031628

    def fun(self):
        x = self.symbols
        return (4 - 2.1 * x[0] ** 2 + x[0] ** 4 / 3) * x[0] ** 2 + x[0] * x[1] + (4 * x[1] ** 2 - 4) * x[1] ** 2
    
class Sphere(Benchmark):
    """
    Sphere objective function.

    This class defines the Sphere [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\\text{Sphere}}(x) = \\sum_{i=1}^{n} x_i^2

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \\in [-1, 1]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for
    :math:`i = 1, ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO Jamil has stupid limits
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-5.12] * self.N, [5.12] * self.N))
        self.global_optimum = [[0 for _ in range(self.N)]]
        self.fglob = 0.0
        self.change_dimensionality = True

    def fun(self):
        x = self.symbols
        return sum([x_i ** 2 for x_i in x])
    
class StyblinskiTang(Benchmark):
    """
    StyblinskiTang objective function.

    This class defines the Styblinski-Tang [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

       f_{\\text{StyblinskiTang}}(x) = \\sum_{i=1}^{n} \\left(x_i^4
                                       - 16x_i^2 + 5x_i \\right)

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \\in [-5, 5]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = -39.16616570377142n` for
    :math:`x_i = -2.903534018185960` for :math:`i = 1, ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-5.0] * self.N, [5.0] * self.N))
        self.global_optimum = [[-2.90353401818596 for _ in range(self.N)]]
        self.fglob = -39.16616570377142 * self.N
        self.change_dimensionality = True

    def fun(self):
        x = self.symbols
        sum1 = 0
        for i in range(self.N):
            sum1 += x[i]**4 - 16*x[i]**2 + 5*x[i]
        return sum1/2

class ThreeHumpCamel(Benchmark):
    """
    Three Hump Camel objective function.

    This class defines the Three Hump Camel [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\\text{ThreeHumpCamel}}(x) = 2x_1^2 - 1.05x_1^4 + \\frac{x_1^6}{6}
                                       + x_1x_2 + x_2^2


    with :math:`x_i \\in [-5, 5]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [0, 0]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-5.0] * self.N, [5.0] * self.N))
        self.custom_bounds = [(-2, 2), (-1.5, 1.5)]
        self.global_optimum = [[0.0, 0.0]]
        self.fglob = 0.0

    def fun(self):
        x = self.symbols
        return 2.0 * x[0] ** 2.0 - 1.05 * x[0] ** 4.0 + x[0] ** 6 / 6.0 + x[0] * x[1] + x[1] ** 2.0
    
class Treccani(Benchmark):
    """
    Treccani objective function.

    This class defines the Treccani [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

        f_{\\text{Treccani}}(x) = x_1^4 + 4x_1^3 + 4x_1^2 + x_2^2


    with :math:`x_i \\in
    [-5, 5]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [-2, 0]` or
    :math:`x = [0, 0]`.

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-5.0] * self.N, [5.0] * self.N))
        self.custom_bounds = [(-2, 2), (-2, 2)]
        self.global_optimum = [[-2.0, 0.0]]
        self.fglob = 0

    def fun(self):
        x = self.symbols
        return x[0] ** 4 + 4.0 * x[0] ** 3 + 4.0 * x[0] ** 2 + x[1] ** 2

class Trefethen(Benchmark):
    """
    Trefethen objective function.

    This class defines the Trefethen [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

        f_{\\text{Trefethen}}(x) = 0.25 x_{1}^{2} + 0.25 x_{2}^{2}
                                  + e^{\\sin\\left(50 x_{1}\\right)}
                                  - \\sin\\left(10 x_{1} + 10 x_{2}\\right)
                                  + \\sin\\left(60 e^{x_{2}}\\right)
                                  + \\sin\\left[70 \\sin\\left(x_{1}\\right)\\right]
                                  + \\sin\\left[\\sin\\left(80 x_{2}\\right)\\right]


    with :math:`x_i \\in [-10, 10]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = -3.3068686474` for
    :math:`x = [-0.02440307923, 0.2106124261]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-1.0] * self.N, [1.0] * self.N))
        self.custom_bounds = list(zip([-0.5] * self.N, [0.5] * self.N))
        self.global_optimum = [[-0.02440307923, 0.2106124261]]
        self.fglob = -3.3068686474

    def fun(self):
        x = self.symbols
        val = 0.25 * x[0] ** 2 + 0.25 * x[1] ** 2
        val += exp(sin(50.0 * x[0])) - sin(10 * x[0] + 10 * x[1])
        val += sin(60 * exp(x[1]))
        val += sin(70 * sin(x[0]))
        val += sin(sin(80 * x[1]))
        return val

class Trid(Benchmark):

    r"""
    Trid objective function.

    This class defines the Trid [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Trid}}(x) = \sum_{i=1}^{n} (x_i - 1)^2
                            - \sum_{i=2}^{n} x_i x_{i-1}


    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-20, 20]` for :math:`i = 1, ..., 6`.

    *Global optimum*: :math:`f(x) = -50` for :math:`x = [6, 10, 12, 12, 10, 6]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO Jamil#150, starting index of second summation term should be 2.
    """

    def __init__(self, dimensions=6):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-20.0] * self.N, [20.0] * self.N))

        self.global_optimum = [[6, 10, 12, 12, 10, 6]]
        self.fglob = -50.0
        self.change_dimensionality = True

    def fun(self):
        x = self.symbols
        a = 0
        for i in range(self.N):
            a += (x[i] - 1)**2
        b = 0
        for i in range(1, self.N):
            b += x[i]*x[i-1]
        return a - b
    
class Trigonometric01(Benchmark):
    """
    Trigonometric 1 objective function.

    This class defines the Trigonometric 1 [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\\text{Trigonometric01}}(x) = \\sum_{i=1}^{n} \\left [n -
                                        \\sum_{j=1}^{n} \\cos(x_j)
                                        + i \\left(1 - cos(x_i)
                                        - sin(x_i) \\right ) \\right]^2

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \\in [0, \\pi]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for
    :math:`i = 1, ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO: equaiton uncertain here.  Is it just supposed to be the cos term
    in the inner sum, or the whole of the second line in Jamil #153.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([0] * self.N, [np.pi] * self.N))
        self.global_optimum = [[0.0 for _ in range(self.N)]]
        self.fglob = 0.0
        self.change_dimensionality = True

    def fun(self):
        x = self.symbols
        N = self.N
        sum_cos_x = sum(cos(x[j]) for j in range(N))
        sum1 = 0
        for i in range(N):
            math_i = i + 1  
            inner_expr = N + math_i - (sin(x[i]) + math_i * cos(x[i]) + sum_cos_x)
            sum1 += inner_expr**2
        return sum1

class Trigonometric02(Benchmark):
    """
    Trigonometric 2 objective function.

    This class defines the Trigonometric 2 [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\\text{Trigonometric2}}(x) = 1 + \\sum_{i=1}^{n} 8 \\sin^2
                                       \\left[7(x_i - 0.9)^2 \\right]
                                       + 6 \\sin^2 \\left[14(x_i - 0.9)^2 \\right]
                                       + (x_i - 0.9)^2

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \\in [-500, 500]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 1` for :math:`x_i = 0.9` for
    :math:`i = 1, ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-500.0] * self.N, [500.0] * self.N))
        self._bounds = list(zip([-1.0] * self.N, [1.0] * self.N))
        self.custom_bounds = list(zip([-1.0] * self.N, [1.0] * self.N))
        self.global_optimum = [[0.9 for _ in range(self.N)]]
        self.fglob = 1.0
        self.change_dimensionality = True

    def fun(self):
        x = self.symbols
        sum1 = 0
        for i in range(self.N):
            sum1 += 8 * (sin(7*(x[i] - 0.9)**2))**2 + 6*(sin(14*(x[0] - 0.9)**2))**2 + (x[i] - 0.9)**2
        return sum1
    
class Ursem01(Benchmark):
    """
    Ursem 1 objective function.

    This class defines the Ursem 1 [1]_ global optimization problem. This is a
    unimodal minimization problem defined as follows:

    .. math::

        f_{\\text{Ursem01}}(x) = - \\sin(2x_1 - 0.5 \\pi) - 3 \\cos(x_2) - 0.5 x_1

    with :math:`x_1 \\in [-2.5, 3]` and :math:`x_2 \\in [-2, 2]`.

    *Global optimum*: :math:`f(x) = -4.81681406371` for
    :math:`x = [1.69714, 0.0]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = [(-2.5, 3.0), (-2.0, 2.0)]
        self.global_optimum = [[1.69714, 0.0]]
        self.fglob = -4.81681406371

    def fun(self):
        x = self.symbols
        return -sin(2 * x[0] - 0.5 * pi) - 3.0 * cos(x[1]) - 0.5 * x[0]
    
class UrsemWaves(Benchmark):
    """
    Ursem Waves objective function.

    This class defines the Ursem Waves [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\\text{UrsemWaves}}(x) = -0.9x_1^2 + (x_2^2 - 4.5x_2^2)x_1x_2
                                   + 4.7 \\cos \\left[ 2x_1 - x_2^2(2 + x_1)
                                   \\right ] \\sin(2.5 \\pi x_1)

    with :math:`x_1 \\in [-0.9, 1.2]`, :math:`x_2 \\in [-1.2, 1.2]`.

    *Global optimum*: :math:`f(x) = -8.5536` for :math:`x = [1.2, 1.2]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO Jamil #159, has an x_2^2 - 4.5 x_2^2 in the brackets. Why wasn't this
    rationalised to -5.5 x_2^2? This makes me wonder if the equation  is listed
    correctly?
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = [(-0.9, 1.2), (-1.2, 1.2)]
        self.global_optimum = [[1.2 for _ in range(self.N)]]
        self.fglob = -8.5536

    def fun(self):
        x = self.symbols
        u = -0.9 * x[0] ** 2
        v = (x[1] ** 2 - 4.5 * x[1] ** 2) * x[0] * x[1]
        w = 4.7 * cos(3 * x[0] - x[1] ** 2 * (2 + x[0])) * sin(2.5 * pi * x[0])
        return u + v + w

class VenterSobiezcczanskiSobieski(Benchmark):
    """
    Venter Sobiezcczanski-Sobieski objective function.

    This class defines the Venter Sobiezcczanski-Sobieski [1]_ global optimization
    problem. This is a multimodal minimization problem defined as follows:

    .. math::

        f_{\\text{VenterSobiezcczanskiSobieski}}(x) = x_1^2 - 100 \\cos^2(x_1)
                                                      - 100 \\cos(x_1^2/30)
                                                      + x_2^2 - 100 \\cos^2(x_2)
                                                      - 100 \\cos(x_2^2/30)


    with :math:`x_i \\in [-50, 50]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = -400` for :math:`x = [0, 0]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO Jamil #160 hasn't written the equation very well. Normally a cos
    squared term is written as cos^2(x) rather than cos(x)^2
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-20.0] * self.N, [20.0] * self.N))
        self.custom_bounds = list(zip([-20.0] * self.N, [20.0] * self.N))
        self.global_optimum = [[0.0 for _ in range(self.N)]]
        self.fglob = -400

    def fun(self):
        x = self.symbols
        u = x[0] ** 2.0 - 100.0 * cos(x[0]) ** 2.0
        v = -100.0 * cos(x[0] ** 2.0 / 30.0) + x[1] ** 2.0
        w = -100.0 * cos(x[1]) ** 2.0 - 100.0 * cos(x[1] ** 2.0 / 30.0)
        return u + v + w

class Vincent(Benchmark):
    """
    Vincent objective function.

    This class defines the Vincent [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\\text{Vincent}}(x) = - \\sum_{i=1}^{n} \\sin(10 \\log(x))

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \\in [0.25, 10]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = -n` for :math:`x_i = 7.70628098`
    for :math:`i = 1, ..., n`

    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([0.25] * self.N, [10.0] * self.N))
        self.global_optimum = [[7.70628098 for _ in range(self.N)]]
        self.fglob = -float(self.N)
        self.change_dimensionality = True

    def fun(self):
        x = self.symbols
        return -sum([sin(10.0 * log(x_i)) for x_i in x])

class Wavy(Benchmark):
    """
    Wavy objective function.

    This class defines the W / Wavy [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

        f_{\\text{Wavy}}(x) = 1 - \\frac{1}{n} \\sum_{i=1}^{n}
                             \\cos(kx_i)e^{-\\frac{x_i^2}{2}}


    Where, in this exercise, :math:`k = 10`. The number of local minima is
    :math:`kn` and :math:`(k + 1)n` for odd and even :math:`k` respectively.

    Here, :math:`x_i \\in [-\\pi, \\pi]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [0, 0]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-pi] * self.N, [pi] * self.N))
        self.global_optimum = [[0.0 for _ in range(self.N)]]
        self.fglob = 0.0
        self.change_dimensionality = True

    def fun(self):
        x = self.symbols
        return 1.0 - 1.0 / self.N * sum([cos(10 * x_i) * exp(-x_i ** 2.0 / 2.0) for x_i in x])

    
class WayburnSeader01(Benchmark):
    """
    Wayburn and Seader 1 objective function.

    This class defines the Wayburn and Seader 1 [1]_ global optimization
    problem. This is a unimodal minimization problem defined as follows:

    .. math::

        f_{\\text{WayburnSeader01}}(x) = (x_1^6 + x_2^4 - 17)^2
                                        + (2x_1 + x_2 - 4)^2


    with :math:`x_i \\in [-5, 5]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [1, 2]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-5.0] * self.N, [5.0] * self.N))
        self.custom_bounds = ([-2, 2], [-2, 2])
        self.global_optimum = [[1.0, 2.0]]
        self.fglob = 0.0

    def fun(self):
        x = self.symbols
        return (x[0] ** 6 + x[1] ** 4 - 17) ** 2 + (2 * x[0] + x[1] - 4) ** 2
    
class WayburnSeader02(Benchmark):
    """
    Wayburn and Seader 2 objective function.

    This class defines the Wayburn and Seader 2 [1]_ global optimization
    problem. This is a unimodal minimization problem defined as follows:

    .. math::

        f_{\\text{WayburnSeader02}}(x) = \\left[ 1.613 - 4(x_1 - 0.3125)^2
                                        - 4(x_2 - 1.625)^2 \\right]^2
                                        + (x_2 - 1)^2


    with :math:`x_i \\in [-500, 500]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [0.2, 1]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-500.0] * self.N, [500.0] * self.N))
        self.custom_bounds = ([-1, 2], [-1, 2])
        self.global_optimum = [[0.2, 1.0]]
        self.fglob = 0.0

    def fun(self):
        x = self.symbols
        u = (1.613 - 4 * (x[0] - 0.3125) ** 2 - 4 * (x[1] - 1.625) ** 2) ** 2
        v = (x[1] - 1) ** 2
        return u + v

class Weierstrass(Benchmark):

    r"""
    Weierstrass objective function.

    This class defines the Weierstrass [1]_ global optimization problem.
    This is a multimodal minimization problem defined as follows:

    .. math::

       f_{\text{Weierstrass}}(x) = \sum_{i=1}^{n} \left [
                                   \sum_{k=0}^{kmax} a^k \cos 
                                   \left( 2 \pi b^k (x_i + 0.5) \right) - n
                                   \sum_{k=0}^{kmax} a^k \cos(\pi b^k) \right ]


    Where, in this exercise, :math:`kmax = 20`, :math:`a = 0.5` and
    :math:`b = 3`.

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-0.5, 0.5]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 4` for :math:`x_i = 0` for
    :math:`i = 1, ..., n`

    .. [1] Mishra, S. Global Optimization by Differential Evolution and
    Particle Swarm Methods: Evaluation on Some Benchmark Functions.
    Munich Personal RePEc Archive, 2006, 1005

    TODO line 1591.
    TODO Jamil, Gavana have got it wrong.  The second term is not supposed to
    be included in the outer sum. Mishra code has it right as does the
    reference referred to in Jamil#166.
    """
    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self.complicated = True
        self._bounds = list(zip([-0.5] * self.N, [0.5] * self.N))

        self.global_optimum = [[0.0 for _ in range(self.N)]]
        self.fglob = 0
        self.change_dimensionality = True

    def fun(self):
        x = self.symbols

        kmax = 20
        a, b = 0.5, 3.0
        sum1 = 0
        for i in range(self.N):
            sum2 = 0
            sum3 = 0
            for k in range(kmax+1):
                sum2 += a**k * cos(2*pi*b**k*(x[i] + 0.5))
            for k in range(kmax+1):
                sum3 += a**k *cos(pi*b**k)
            sum1 += sum2
        return sum1 - self.N*sum3

class Whitley(Benchmark):

    r"""
    Whitley objective function.

    This class defines the Whitley [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Whitley}}(x) = \sum_{i=1}^n \sum_{j=1}^n
                                \left[\frac{(100(x_i^2-x_j)^2
                                + (1-x_j)^2)^2}{4000} - \cos(100(x_i^2-x_j)^2
                                + (1-x_j)^2)+1 \right]


    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-10.24, 10.24]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 1` for
    :math:`i = 1, ..., n`

    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015

    TODO Jamil#167 has '+ 1' inside the cos term, when it should be outside it.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-10.24] * self.N,
                           [10.24] * self.N))
        self.custom_bounds = ([-1, 2], [-1, 2])

        self.global_optimum = [[1.0 for _ in range(self.N)]]
        self.fglob = 0.0
        self.change_dimensionality = True
        self.complicated = True

    def fun(self):
        x = self.symbols
        a = 0
        for i in range(self.N):
            b = 0
            for j in range(self.N):
                left = ((100 * (x[i]**2 - x[j])**2 + (1 - x[j])**2)**2)/4000
                right = cos(100*(x[i]**2 - x[j])**2 + (1 - x[j])**2)
                b += left - right + 1
            a += b
        return a

    
class XinSheYang03(Benchmark):
    """
    Xin-She Yang 3 objective function.

    This class defines the Xin-She Yang 3 [1]_ global optimization problem.
    This is a multimodal minimization problem defined as follows:

    .. math::

        f_{\\text{XinSheYang03}}(x) = e^{-\\sum_{i=1}^{n} (x_i/\\beta)^{2m}}
                                     - 2e^{-\\sum_{i=1}^{n} x_i^2}
                                     \\prod_{i=1}^{n} \\cos^2(x_i)


    Where, in this exercise, :math:`\\beta = 15` and :math:`m = 3`.

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \\in [-20, 20]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = -1` for :math:`x_i = 0` for
    :math:`i = 1, ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-1.0] * self.N, [1.0] * self.N))
        self.custom_bounds = list(zip([-1.0] * self.N, [1.0] * self.N))
        self.global_optimum = [[0 for _ in range(self.N)]]
        self.fglob = -1.0
        self.change_dimensionality = True

    def fun(self):
        x = self.symbols
        beta, m = (15.0, 5.0)
        u = sum([(x_i / beta) ** (2 * m) for x_i in x])
        v = sum([x_i ** 2 for x_i in x])
        w = prod([cos(x_i) ** 2 for x_i in x])
        return exp(-u) - 2 * exp(-v) * w

class Zacharov(Benchmark):
    """
    Zacharov objective function.

    This class defines the Zacharov [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\\text{Zacharov}}(x) = \\sum_{i=1}^{n} x_i^2 + \\left ( \\frac{1}{2}
                                 \\sum_{i=1}^{n} i x_i \\right )^2
                                 + \\left ( \\frac{1}{2} \\sum_{i=1}^{n} i x_i 
                                 \\right )^4

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \\in [-5, 10]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for
    :math:`i = 1, ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-5.0] * self.N, [10.0] * self.N))
        self.custom_bounds = ([-1, 1], [-1, 1])
        self.global_optimum = [[0 for _ in range(self.N)]]
        self.fglob = 0.0
        self.change_dimensionality = True

    def fun(self):
        x = self.symbols
        u = 0
        v = 0
        for i in range(self.N):
            u += x[i]**2
            v += (i + 1)*x[i]
        return u + (0.5 * v) ** 2 + (0.5 * v) ** 4
    
class Zettl(Benchmark):
    """
    Zettl objective function.

    This class defines the Zettl [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

       f_{\\text{Zettl}}(x) = \\frac{1}{4} x_{1} + \\left(x_{1}^{2} - 2 x_{1}
                             + x_{2}^{2}\\right)^{2}


    with :math:`x_i \\in [-1, 5]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = -0.0037912` for :math:`x = [-0.029896, 0.0]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-5.0] * self.N, [10.0] * self.N))
        self.global_optimum = [[-0.02989597760285287, 0.0]]
        self.fglob = -0.003791237220468656

    def fun(self):
        x = self.symbols
        return (x[0] ** 2 + x[1] ** 2 - 2 * x[0]) ** 2 + 0.25 * x[0]
    
class Zirilli(Benchmark):
    """
    Zettl objective function.

    This class defines the Zirilli [1]_ global optimization problem. This is a
    unimodal minimization problem defined as follows:

    .. math::

        f_{\\text{Zirilli}}(x) = 0.25x_1^4 - 0.5x_1^2 + 0.1x_1 + 0.5x_2^2

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \\in [-10, 10]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = -0.3523` for :math:`x = [-1.0465, 0]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))
        self.custom_bounds = ([-2.0, 2.0], [-2.0, 2.0])
        self.global_optimum = [[-1.0465, 0.0]]
        self.fglob = -0.35238603

    def fun(self):
        x = self.symbols
        return 0.25 * x[0] ** 4 - 0.5 * x[0] ** 2 + 0.1 * x[0] + 0.5 * x[1] ** 2
    
class F01(Benchmark):
    """
    Locating multiple optimal solutions of nonlinear equation systems based on multiobjective optimization
    Liao Z., Gong W., Wang L. Memetic niching-based evolutionary algorithms for solving nonlinear equation system
    It has 11 roots
    roots = −0.924840 −0.924840
            −0.866760 −0.866760
            −0.562010 −0.562010
            −0.428168 −0.428168
            −0.187960 −0.187960
            0.000000 0.000000
            0.187960 0.187960
            0.428168 0.428168
            0.562010 0.562010
            0.866760 0.866760
            0.924840 0.924840
    """
    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-1.0] * self.N, [1.0] * self.N))
        self.system_roots = [[-0.924840, -0.924840],
                             [-0.866760, -0.866760],
                             [-0.562010, -0.562010],
                             [-0.428168, -0.428168],
                             [-0.187960, -0.187960],
                             [0.000000, 0.000000],
                             [0.187960, 0.187960],
                             [0.428168, 0.428168],
                             [0.562010, 0.562010],
                             [0.866760, 0.866760],
                             [0.924840, 0.924840]]

    def system(self):
        x = self.symbols
        F = sym.Matrix([[x[0] - sin(5*pi*x[1])],
                       [x[0] - x[1]]])
        return F


class F02(Benchmark):
    """
    Locating multiple optimal solutions of nonlinear equation systems based on multiobjective optimization
    Liao Z., Gong W., Wang L. Memetic niching-based evolutionary algorithms for solving nonlinear equation system
    It has 15 roots
    0.416408 −0.909178
    −0.561364 −0.827569
    −0.724322 −0.689462
    0.837812 −0.545959
    0.886984 −0.461799
    −0.962322 −0.271914
    −0.972855 −0.231415
    1.000000 0.000000
    −0.972855 0.231416
    −0.962322 0.271914
    0.886984 0.461799
    0.837812 0.545959
    −0.724322 0.689462
    −0.561364 0.827569
    0.416408 0.909178
    """
    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))
        self.system_roots = [[0.416408, -0.909178],
                             [-0.561364, -0.827569],
                             [-0.724322, -0.689462],
                             [0.837812, -0.545959],
                             [0.886984, -0.461799],
                             [-0.962322, -0.271914],
                             [-0.972855, -0.231415],
                             [1.000000, 0.000000],
                             [-0.972855, 0.231416],
                             [-0.962322, 0.271914],
                             [0.886984, 0.461799],
                             [0.837812, 0.545959],
                             [-0.724322, 0.689462],
                             [-0.561364, 0.827569],
                             [0.416408, 0.909178]]

    def system(self):
        x = self.symbols
        F = sym.Matrix([[x[0] - cos(4*pi*x[1])],
                       [x[0]**2 + x[1]**2 - 1]])
        return F


class F03(Benchmark):
    """
    Grosan, C. , & Abraham, A. (2008). A new approach for solving nonlinear equations systems. IEEE Transactions on Systems, Man and Cybernetics
    Liao Z., Gong W., Wang L. Memetic niching-based evolutionary algorithms for solving nonlinear equation system
    It has 1 root
    (0.257833, 0.381097, 0.278745, 0.200669, 0.445251, 0.149184, 0.432010, 0.073403, 0.345967, 0.427326)
    """
    def __init__(self, dimensions=10):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-2.0] * self.N, [2.0] * self.N))
        self.system_roots = [[0.257833, 0.381097, 0.278745, 0.200669, 0.445251, 0.149184, 0.432010, 0.073403, 0.345967, 0.427326]]

    def system(self):
        x = self.symbols
        F = sym.Matrix([
                        [x[0] - 0.25428722 - 0.18324757*x[3]*x[2]*x[8]],
                       [x[1] - 0.37842197 - 0.16275449*x[0]*x[9]*x[5]],
                       [x[2] - 0.27162577 - 0.16955071*x[0]*x[1]*x[9]],
                       [x[3] - 0.19807914 - 0.15585316*x[0]*x[5]*x[6]],
                       [x[4] - 0.44166728 - 0.19950920*x[6]*x[5]*x[2]],
                       [x[5] - 0.14654113 - 0.18922793*x[7]*x[4]*x[9]],
                       [x[6] - 0.42937161 - 0.21180486*x[1]*x[4]*x[7]],
                       [x[7] - 0.07056438 - 0.17081208*x[0]*x[6]*x[5]],
                       [x[8] - 0.34504906 - 0.19612740*x[9]*x[5]*x[7]],
                       [x[9] - 0.42651102 - 0.21466544*x[3]*x[7]*x[0]],
                       ])
        return F


class F05(Benchmark):
    """
    Finding all solutions of nonlinear systems using a hybrid metaheuristic with Fuzzy Clustering Means
    Liao Z., Gong W., Wang L. Memetic niching-based evolutionary algorithms for solving nonlinear equation system
    It has 9 roots
    −0.127961 −1.953715
    −0.270845 −0.923039
    0.086678 2.884255
    3.385154 0.073852
    3.584428 −1.848127
    3.000000 2.000000
    −3.779310 −3.283186
    −3.073026 −0.081353
    −2.805118 3.131313
    """
    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-20.0] * self.N, [20.0] * self.N))
        self.system_roots = [[-0.127961, -1.953715],
                             [-0.270845, -0.923039],
                             [0.086678, 2.884255],
                             [3.385154, 0.073852],
                             [3.584428, -1.848127],
                             [3.000000, 2.000000],
                             [-3.779310, -3.283186],
                             [-3.073026, -0.081353],
                             [-2.805118, 3.131313]]

    def system(self):
        x = self.symbols
        F = sym.Matrix([[4 * x[0]**3 + 4*x[0]*x[1] + 2*x[1]**2 - 42*x[0] - 14],
                       [4 * x[1]**3 + 2*x[0]**2 + 4*x[0]*x[1] - 26*x[1] - 22]])
        return F


class F06(Benchmark):
    """Liao Z., Gong W., Wang L. Memetic niching-based evolutionary algorithms for solving nonlinear equation system
    It has 13 roots
    0.000000 0.000000
    3.141593 0.000000
    1.570796 1.570796
    6.283185 0.000000
    0.000000 3.141593
    4.712389 1.570796
    3.141593 3.141593
    1.570796 4.712389
    6.283185 3.141593
    0.000000 6.283185
    4.712389 4.712389
    3.141593 6.283185
    6.283185 6.283185
    """
    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([0.0] * self.N, [2.0*pi] * self.N))
        self.system_roots = [[0.000000, 0.000000],
                             [3.141593, 0.000000],
                             [1.570796, 1.570796],
                             [6.283185, 0.000000],
                             [0.000000, 3.141593],
                             [4.712389, 1.570796],
                             [3.141593, 3.141593],
                             [1.570796, 4.712389],
                             [6.283185, 3.141593],
                             [0.000000, 6.283185],
                             [4.712389, 4.712389],
                             [3.141593, 6.283185],
                             [6.283185, 6.283185]]

    def system(self):
        x = self.symbols
        F = sym.Matrix([[-sin(x[0])*cos(x[1]) - 2 * cos(x[0])*sin(x[1])],
                       [-cos(x[0])*sin(x[1]) - 2 * sin(x[0])*cos(x[1])]])
        return F


class F07(Benchmark):
    """Liao Z., Gong W., Wang L. Memetic niching-based evolutionary algorithms for solving nonlinear equation system
    It has 16 roots
    0.1644 −0.9864 −0.9471 −0.3210 −0.9982 −0.0594 0.4110 0.9116
    0.1644 −0.9864 −0.9471 −0.3210 −0.9982 0.0594 0.4110 −0.9116
    0.1644 −0.9864 −0.9471 −0.3210 0.9982 −0.0594 0.4110 0.9116
    0.1644 −0.9864 −0.9471 −0.3210 0.9982 0.0594 0.4110 −0.9116
    0.1644 −0.9864 0.7185 −0.6956 −0.9980 −0.0638 −0.5278 0.8494
    0.1644 −0.9864 0.7185 −0.6956 −0.9980 0.0638 −0.5278 −0.8494
    0.1644 −0.9864 0.7185 −0.6956 0.9980 −0.0638 −0.5278 0.8494
    0.1644 −0.9864 0.7185 −0.6956 0.9980 0.0638 −0.5278 −0.8494
    0.6716 0.7410 −0.6516 −0.7586 −0.9625 −0.2711 −0.4376 0.8992
    0.6716 0.7410 −0.6516 −0.7586 −0.9625 0.2711 −0.4376 −0.8992
    0.6716 0.7410 −0.6516 −0.7586 0.9625 −0.2711 −0.4376 0.8992
    0.6716 0.7410 −0.6516 −0.7586 0.9625 0.2711 −0.4376 −0.8992
    0.6716 0.7410 0.9519 −0.3064 −0.9638 −0.2666 0.4046 0.9145
    0.6716 0.7410 0.9519 −0.3064 −0.9638 0.2666 0.4046 −0.9145
    0.6716 0.7410 0.9519 −0.3064 0.9638 0.2666 0.4046 −0.9145
    0.6716 0.7410 0.9519 −0.3064 0.9638 −0.2666 0.4046 0.9145
    """
    def __init__(self, dimensions=8):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-1.0] * self.N, [1.0] * self.N))
        self.system_roots = [
            [0.1644, -0.9864, -0.9471, -0.3210, -0.9982, -0.0594, 0.4110, 0.9116],
            [0.1644, -0.9864, -0.9471, -0.3210, -0.9982, 0.0594, 0.4110, -0.9116],
            [0.1644, -0.9864, -0.9471, -0.3210, 0.9982, -0.0594, 0.4110, 0.9116],
            [0.1644, -0.9864, -0.9471, -0.3210, 0.9982, 0.0594, 0.4110, -0.9116],
            [0.1644, -0.9864, 0.7185, -0.6956, -0.9980, -0.0638, -0.5278, 0.8494],
            [0.1644, -0.9864, 0.7185, -0.6956, -0.9980, 0.0638, -0.5278, -0.8494],
            [0.1644, -0.9864, 0.7185, -0.6956, 0.9980, -0.0638, -0.5278, 0.8494],
            [0.1644, -0.9864, 0.7185, -0.6956, 0.9980, 0.0638, -0.5278, -0.8494],
            [0.6716, 0.7410, -0.6516, -0.7586, -0.9625, -0.2711, -0.4376, 0.8992],
            [0.6716, 0.7410, -0.6516, -0.7586, -0.9625, 0.2711, -0.4376, -0.8992],
            [0.6716, 0.7410, -0.6516, -0.7586, 0.9625, -0.2711, -0.4376, 0.8992],
            [0.6716, 0.7410, -0.6516, -0.7586, 0.9625, 0.2711, -0.4376, -0.8992],
            [0.6716, 0.7410, 0.9519, -0.3064, -0.9638, -0.2666, 0.4046, 0.9145],
            [0.6716, 0.7410, 0.9519, -0.3064, -0.9638, 0.2666, 0.4046, -0.9145],
            [0.6716, 0.7410, 0.9519, -0.3064, 0.9638, 0.2666, 0.4046, -0.9145],
            [0.6716, 0.7410, 0.9519, -0.3064, 0.9638, -0.2666, 0.4046, 0.9145]
        ]

    def system(self):
        x = self.symbols
        F = sym.Matrix([
                        [x[0]**2 + x[1]**2 - 1],
                       [x[2]**2 + x[3]**2 - 1],
                       [x[4]**2 + x[5]**2 - 1],
                       [x[6]**2 + x[7]**2 - 1],
                       [4.731*10e-3 * x[0]*x[2] - 0.3578*x[1]*x[2] - 0.1238*x[0] + x[6] - 1.637*10e-3 * x[1] - 0.9338*x[3] - 0.3571],
                       [0.2238 * x[0]*x[2] + 0.7623*x[1]*x[2] + 0.2638*x[0] - x[6] - 0.07745 * x[1] - 0.6734*x[3] - 0.6022],
                       [x[5]*x[7] + 0.3578*x[0] + 4.731*10e-3*x[1]],
                       [-0.7623*x[0] + 0.2238*x[1] + 0.3461],
                       ])
        return F


class F08(Benchmark):
    """Liao Z., Gong W., Wang L. Memetic niching-based evolutionary algorithms for solving nonlinear equation system
    It has 7 roots
    0.810561 0.810561 −0.625687
    0.810561 −0.625687 0.810561
    −0.625687 0.810561 0.810561
    0.543850 0.995778 0.543850
    0.543850 0.543850 0.995778
    0.995778 0.543850 0.543850
    0.739086 0.739086 0.739086
    """
    def __init__(self, dimensions=3):
        Benchmark.__init__(self, dimensions)
        self.change_dimensionality = True
        self._bounds = list(zip([-20.0] * self.N, [20.0] * self.N))
        self.system_roots = [
            [0.810561, 0.810561, -0.625687],
            [0.810561, -0.625687, 0.810561],
            [-0.625687, 0.810561, 0.810561],
            [0.543850, 0.995778, 0.543850],
            [0.543850, 0.543850, 0.995778],
            [0.995778, 0.543850, 0.543850],
            [0.739086, 0.739086, 0.739086]
        ]

    def system(self):
        x = self.symbols
        F_loc = []
        sum1 = 0
        for j in range(self.N):
            sum1 += x[j]
        for i in range(self.N):
            F_loc.append(x[i] - cos(2*x[i] - sum1))
        F = sym.Matrix(F_loc)
        return F


class F09(Benchmark):
    """Liao Z., Gong W., Wang L. Memetic niching-based evolutionary algorithms for solving nonlinear equation system
    It has 2 roots
    (0, −2) and (0.707660, −1.5)
    """
    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = [[0, 1], [-10, 0]]
        self.system_roots = [[0, -2],
                             [0.707660, -1.5]]

    def system(self):
        x = self.symbols
        F = sym.Matrix([[x[0]**2 - x[1] - 2],
                       [x[0] + sin(pi/2 * x[1])]
                       ])
        return F


class F13(Benchmark):
    """Liao Z., Gong W., Wang L. Memetic niching-based evolutionary algorithms for solving nonlinear equation system
    It has two roots: (1, 1, 1, 1, 1), (0.916355, 0.916355, 0.916355, 0.916355, 1.418227)
    """
    def __init__(self, dimensions=5):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-2.0] * self.N, [2.0] * self.N))
        self.system_roots = [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [0.916355, 0.916355, 0.916355, 0.916355, 1.418227]
        ]

    def system(self):
        x = self.symbols
        F = sym.Matrix([[2 * x[0] + x[1] + x[2] + x[3] + x[4] - 6],
                       [x[0] + 2 * x[1] + x[2] + x[3] + x[4] - 6],
                       [x[0] + x[1] + 2 * x[2] + x[3] + x[4] - 6],
                       [x[0] + x[1] + x[2] + 2 * x[3] + x[4] - 6],
                       [x[0] * x[1] * x[2] * x[3] * x[4] - 1],
                       ])
        return F


class F16(Benchmark):
    """Liao Z., Gong W., Wang L. Memetic niching-based evolutionary algorithms for solving nonlinear equation system
    Third-order methods from quadrature formulae for solving systems of nonlinear equations
    It has 2 roots
    (3.75683, 2.779849) and (1.37347, −1.52496)
    """
    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = [[0.1, 4], [-3, 4]]
        self.system_roots = [[3.75683, 2.779849],
                             [1.37347, -1.52496]]

    def system(self):
        x = self.symbols
        F = sym.Matrix([[x[0] - x[1]**2 + 3*log(x[0])],
                       [1 - 5*x[0] + 2*x[1]**2 - x[0]*x[1]]
                       ])
        return F


class F18(Benchmark):
    """Liao Z., Gong W., Wang L. Memetic niching-based evolutionary algorithms for solving nonlinear equation system
    It has 2 roots
    (0.99742 1.00259 −4.02826) and (1, 2, 4)
    Modified Newton’s method for systems of nonlinear equations with singular Jacobian
    Original paper only 1 solution: (1, 2 -4)
    
    """
    def __init__(self, dimensions=3):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-5.0] * self.N, [5.0] * self.N))
        self.system_roots = [[0.99742, 1.00259, -4.02826],
                             [1.0, 2.0, 4.0]]

    def system(self):
        x = self.symbols
        F = sym.Matrix([[(x[0] - 1)**4 * exp(x[1])],
                       [(x[1] - 2)**5 * (x[0]*x[1] - 1)],
                       [(x[2] + 4)**6]
                       ])
        return F


class F19(Benchmark):
    """
    Modified Newton’s method for systems of nonlinear equations with singular Jacobian
    Liao Z., Gong W., Wang L. Memetic niching-based evolutionary algorithms for solving nonlinear equation system
    It has two roots: (0.1756, 0.8244, 1) and (0.7042, 0.2958, 0.9999)
    """
    def __init__(self, dimensions=3):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-5.0] * self.N, [5.0] * self.N))
        self.system_roots = [[0.1756, 0.8244, 1.0],
                             [0.7042, 0.2958, 0.9999]]

    def system(self):
        x = self.symbols
        F = sym.Matrix([[exp(x[0]**2) - 8*x[0]*sin(x[1])],
                       [x[0] + x[1] - 1],
                       [(x[2] - 1)**3]
                       ])
        return F


class F20(Benchmark):
    """
    An efficient solver for systems of nonlinear equations with singular Jacobian via diagonal updating
    Liao Z., Gong W., Wang L. Memetic niching-based evolutionary algorithms for solving nonlinear equation system
    It has three roots: ( −0.1153, −0.1153, −0.1153),(0.3577, 0.3577, 0.3577) and ( −0.0998, 0, 0)
    """
    def __init__(self, dimensions=3):
        Benchmark.__init__(self, dimensions)
        self._bounds = [[-2, 2], [-2, 2], [-10, 10]]
        self.system_roots = [[-0.1153, -0.1153, -0.1153],
                             [0.3577, 0.3577, 0.3577],
                             [-0.0998, 0.0, 0.0]]

    def system(self):
        x = self.symbols
        F = sym.Matrix([[x[0]**3 - x[0]*x[1]*x[2]],
                       [x[1]**2 - x[0]*x[2]],
                       [10*x[0]*x[1] * x[2] - x[0] - 0.1]
                       ])
        return F


class F24(Benchmark):
    """Liao Z., Gong W., Wang L. Memetic niching-based evolutionary algorithms for solving nonlinear equation system
    It has 8 roots
    0 0 0
    0.488122 0.959435 0.149452
    0.540304 0.953754 0.169399
    0.959447 0.149373 0.487917
    0.14944 0.488092 0.95944
    0.953781 0.169343 0.540157
    0.169254 0.539937 0.953788
    0.739584 0.739584 0.739574
    """
    def __init__(self, dimensions=3):
        Benchmark.__init__(self, dimensions)
        self._bounds = [[0, 10], [0, 10], [0, 1]]
        self.system_roots = [
            [0.0, 0.0, 0.0],
            [0.488122, 0.959435, 0.149452],
            [0.540304, 0.953754, 0.169399],
            [0.959447, 0.149373, 0.487917],
            [0.14944, 0.488092, 0.95944],
            [0.953781, 0.169343, 0.540157],
            [0.169254, 0.539937, 0.953788],
            [0.739584, 0.739584, 0.739574]
        ]

    def system(self):
        x = self.symbols
        F = sym.Matrix([[-3.84*x[0]**2 + 3.84*x[0] - x[1]],
                       [-3.84*x[1]**2 + 3.84*x[1] - x[2]],
                       [-3.84*x[2]**2 + 3.84*x[2] - x[0]]
                       ])
        return F


class F27(Benchmark):
    """Liao Z., Gong W., Wang L. Memetic niching-based evolutionary algorithms for solving nonlinear equation system
    It has 18 roots
    1.82668 3.41518
    −2.97476 2.48055
    2.97436 −2.48055
    0.46246 3.84527
    2.12002 3.24121
    −3.27714 −2.06406
    −1.2827 3.65437
    −1.82661 −3.41518
    1.08376 −3.71827
    1.28276 −3.65437
    −1.08379 3.71827
    −0.46246 −3.84527
    −2.12002 −3.24121
    2.55955 −2.90665
    −0.326176 −3.85922
    0.326176 3.85922
    −2.55955 2.90665
    3.27714 2.06406
    """
    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-20.0] * self.N, [20.0] * self.N))
        self.system_roots = [
            [1.82668, 3.41518],
            [-2.97476, 2.48055],
            [2.97436, -2.48055],
            [0.46246, 3.84527],
            [2.12002, 3.24121],
            [-3.27714, -2.06406],
            [-1.2827, 3.65437],
            [-1.82661, -3.41518],
            [1.08376, -3.71827],
            [1.28276, -3.65437],
            [-1.08379, 3.71827],
            [-0.46246, -3.84527],
            [-2.12002, -3.24121],
            [2.55955, -2.90665],
            [-0.326176, -3.85922],
            [0.326176, 3.85922],
            [-2.55955, 2.90665],
            [3.27714, 2.06406]
        ]

    def system(self):
        x = self.symbols
        F = sym.Matrix([[4*sin(4*x[0]) - x[1]],
                       [x[0]**2 + x[1]**2 - 15]
                       ])
        return F


class F28(Benchmark):
    """Liao Z., Gong W., Wang L. Memetic niching-based evolutionary algorithms for solving nonlinear equation system
    It has 18 roots
    −12.409851 −12.072994
    −11.886135 −10.306380
    −9.268258 −8.931402
    −8.744542 −7.164787
    −6.126665 −5.789809
    −5.602950 −4.023195
    −2.985073 −2.648216
    −2.461357 −0.881602
    0.156520 0.493376
    0.680236 2.259991
    3.298113 3.634969
    3.821828 5.401583
    6.439705 6.776562
    6.963421 8.543176
    9.581298 9.918154
    10.105014 11.684769
    12.722891 13.059747
    13.246606 14.826361
    """
    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-15.0] * self.N, [15.0] * self.N))
        self.system_roots = [
            [-12.409851, -12.072994],
            [-11.886135, -10.306380],
            [-9.268258, -8.931402],
            [-8.744542, -7.164787],
            [-6.126665, -5.789809],
            [-5.602950, -4.023195],
            [-2.985073, -2.648216],
            [-2.461357, -0.881602],
            [0.156520, 0.493376],
            [0.680236, 2.259991],
            [3.298113, 3.634969],
            [3.821828, 5.401583],
            [6.439705, 6.776562],
            [6.963421, 8.543176],
            [9.581298, 9.918154],
            [10.105014, 11.684769],
            [12.722891, 13.059747],
            [13.246606, 14.826361]
        ]

    def system(self):
        x = self.symbols
        F = sym.Matrix([[cos(2*x[0]) - cos(2*x[1]) - 0.4],
                       [2*(x[1] - x[0]) + sin(2*x[1]) - sin(2*x[0]) - 1.2]
                       ])
        return F


class F29(Benchmark):
    """Liao Z., Gong W., Wang L. Memetic niching-based evolutionary algorithms for solving nonlinear equation system
    It has 4 roots
    2.33387 2.30916
    −1.99146 −3.73926
    4.00916 −1.40771
    4.69774 −0.77751
    """
    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-5.0] * self.N, [5.0] * self.N))
        self.system_roots = [[2.33387, 2.30916],
                             [-1.99146, -3.73926],
                             [4.00916, -1.40771],
                             [4.69774, -0.77751]]

    def system(self):
        x = self.symbols
        F = sym.Matrix([[x[0] + 0.5*x[1]**2 - 5],
                       [x[0] + 5*sin(pi*x[1]/2)]
                       ])
        return F


class F30(Benchmark):
    """Liao Z., Gong W., Wang L. Memetic niching-based evolutionary algorithms for solving nonlinear equation system
    It has 6 roots
    0.35969 −0.93306
    −0.99874 −0.05012
    −0.20230 0.979322
    −0.35969 −0.93306
    0.20230 0.97932
    0.99874 −0.05016
    """
    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-2.0] * self.N, [2.0] * self.N))
        self.system_roots = [[0.35969, -0.93306],
                             [-0.99874, -0.05012],
                             [-0.20230, 0.979322],
                             [-0.35969, -0.93306],
                             [0.20230, 0.97932],
                             [0.99874, -0.05016]]

    def system(self):
        x = self.symbols
        F = sym.Matrix([[x[0]**2 + x[1]**2 - 1],
                       [20*x[0]**2 * x[1] + 2*x[1]**5 + 1]
                       ])
        return F