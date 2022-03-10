import numpy as np
import sympy as sym
from sympy import lambdify
import plotly.graph_objects as go

import random
import sys
import time

from typing import Tuple, List

# adjust parameters for plotting module
# setting these parameters automatically goes beyond the requirements of this matter
MAX_VALUE_IN_DOMAIN = 1200  # maximum value of function in domain
MIN_VALUE_IN_DOMAIN = 0
PLOT_DOMAIN_MAX = 6
PLOT_DOMAIN_MIN = -6

# ====================================== #
#           Plotting functions           #
# ====================================== #


def plot_function_descent(self, type_method) -> None:
    """
    This ploter method illustrates gradient descent of points in 3D.
    type_method: Type of method to plot. One of the following: 'Newton', 'Gradient.
    Return: None
    """

    # Number of samples to generate | resolution of the 3d surface
    N_SAMPLES = 200

    # arrays with evenly spaced numbers over a specified interval
    X_lin = np.linspace(PLOT_DOMAIN_MIN, PLOT_DOMAIN_MAX, N_SAMPLES)
    Y_lin = np.linspace(PLOT_DOMAIN_MIN, PLOT_DOMAIN_MAX, N_SAMPLES)

    X, Y = np.meshgrid(X_lin, Y_lin)

    x, y = sym.symbols('x y')
    calculate_value = lambdify([x, y], self.function, 'numpy')  # returns a numpy-ready function

    # calculate function value for points
    F = calculate_value(X, Y)

    x_line = np.array([x1 for x1, _ in self._step_points], dtype=float)
    y_line = np.array([y1 for _, y1 in self._step_points], dtype=float)

    function_value_increased = calculate_value(x_line, y_line)

    fig = go.Figure(data=[
                    go.Scatter3d(x=x_line,
                                 y=y_line,
                                 z=function_value_increased,
                                 mode='markers',
                                 marker=dict(size=8,
                                             color=function_value_increased,
                                             colorscale=['#00A160', '#AA0A2C']),  # https://plotly.com/python/builtin-colorscales/
                                 hoverinfo='none',
                                 showlegend=False),  # disable additional scale
                    go.Scatter3d(x=x_line,
                                 y=y_line,
                                 z=function_value_increased,
                                 mode='lines',
                                 line=dict(color='rgba(50, 50, 50, 0.5)',
                                           width=2),
                                 hoverinfo='none',
                                 showlegend=False),  # disable additional scale
                    go.Surface(z=F,
                               x=X,
                               y=Y,
                               opacity=0.65,
                               showlegend=False,  # disable additional scale
                               contours={"z": {
                                         "show": True,
                                         "start": MIN_VALUE_IN_DOMAIN,
                                         "end": MAX_VALUE_IN_DOMAIN,
                                         "size": 15,
                                         "color":
                                         "rgba(100, 100, 100, 0.35)"}},
                               colorbar=dict(
                                        title='Function value',  # title here
                                        titleside='right',
                                        titlefont=dict(size=16,
                                                       family='Arial, sans-serif')))
                    ])

    point_str = f'({self._point[0]},{self._point[1]})'

    fig.update_layout(
        title=f'{type_method} descent method of {self.function} from point {point_str}'
              f'<br />Step size: {self._step_size}<br />Epsilon: {self._epsilon}'
              f'<br />Max iterations: {self._max_iterations}',
        autosize=True,
        scene=dict(xaxis_title='X',
                   yaxis_title='Y',
                   zaxis_title='Function Value'),
        scene_camera=dict(
            eye=dict(x=1, y=-1.75, z=1)
        ),
        margin=dict(l=50, r=50, b=10, t=150))

    # save plot as html
    fig.write_html(f"{point_str}_{self._step_size}_{type_method}_descent.html")

    data_4_plots = (type_method,
                    self.function,
                    self._step_size,
                    self._epsilon,
                    self._max_iterations)

    # also execute 2D plot of contours
    _plot_contour(F, X_lin, Y_lin, x_line, y_line,
                  point_str, function_value_increased,
                  data_4_plots)


def _plot_contour(F, X_lin, Y_lin, x_line, y_line, point_str, function_value, data_4_plots):
    fig = go.Figure(data=[go.Contour(z=F,
                                     x=X_lin,
                                     y=Y_lin,
                                     contours_coloring='heatmap',
                                     contours=dict(
                                         start=MIN_VALUE_IN_DOMAIN,
                                         end=MAX_VALUE_IN_DOMAIN,
                                         size=15,
                                     ),
                                     showlegend=False,  # disable additional scale
                                     colorbar=dict(
                                         title='Function value',  # title here
                                         titleside='right',
                                         titlefont=dict(size=16,
                                                        family='Arial, sans-serif')),
                                     line_smoothing=0.85,
                                     line=dict(color="rgba(200, 200, 200, 0.65)", width=0.5)),
                          go.Scatter(x=x_line,
                                     y=y_line,
                                     mode='markers',
                                     marker=dict(size=11,
                                                 color=function_value,
                                                 colorscale=['#00A160', '#AA0A2C']),  # https://plotly.com/python/builtin-colorscales/
                                     hoverinfo='skip',
                                     showlegend=False),  # disable additional scale
                          go.Scatter(x=x_line,
                                     y=y_line,
                                     mode='lines',
                                     showlegend=False,  # disable additional scale
                                     line=dict(color='rgba(50, 200, 100, 0.7)',
                                               width=2)
                                     )
                          ]
                    )

    fig.update_layout(title=f'{data_4_plots[0]} descent method of {data_4_plots[1]}'
                            f' from point {point_str}<br />'
                            f'Step size: {data_4_plots[2]}<br />Epsilon: {data_4_plots[3]}'
                            f'<br />Max iterations: {data_4_plots[4]}',
                            autosize=False,
                            scene_camera_eye=dict(x=1, y=1, z=1),
                            width=960, height=960,
                            margin=dict(l=50, r=100, t=150),
                            xaxis_title="X",
                            yaxis_title="Y")

    # save plot to html
    fig.write_html(f"{point_str}_{data_4_plots[2]}_{data_4_plots[0]}_descent_contour.html")


# ====================================== #
#               Algorithms               #
# ====================================== #


class NewtonMethod:
    """
    Implementation of Newton's method algorithm.
    Read more in the :ref:`https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization`.

    Parameters
    ----------
    function : str
        Two-variable function that is defined and differentiable in a neighborhood of
        a considered point.
    domain : List[int], default=[-5, 5]
        Maximum interval for which the function is defined on in one dimension. For function with
        two variables as a domain will be considered a Cartesian product of two sets.
        For default [-5, 5]. The domain is [-5, 5] x [-5, 5].
    start_point : Tuple[int, int], default=None
        Coordinates of a point from which to start the algorithm run. If left blank, point will
        be chosen randomly from the domain.
    step_size : float, default=1e-2
        Step size influence the magnitude of impact of hessian on the next point.
    max_iterations : int, default=5000
        Maximum number of iterations in which the algorithm can find the function's minimum.
    epsilon : float, default=1e-12
        Represents the maximum acceptable and absolute difference between previous and next point
        calculated by gradient descent algorithm.
    """
    def __init__(self,
                 function,
                 domain: List[int] = [-5, 5],
                 start_point: Tuple[int] = None,
                 step_size: float = 1e-2,
                 max_iterations: int = 5000,
                 epsilon: float = 1e-12
                 ):
        self.function = self._convert_function(function)  # convert str function to sympy object
        self._domain = domain
        self._step_size = step_size
        self._epsilon = epsilon
        self._max_iterations = max_iterations

        if start_point:  # if points have not been passed
            self._point = np.array(start_point)  # assign to numpy array
        else:
            self._point = self._generate_point()

        self._step_points = None
        self._p_minimum = None

    def _convert_function(self, function):
        """
        Converts str function to sympify object.
        function: String containing function definition
        Return: Sympify object
        """
        return sym.sympify(function)

    def _generate_point(self) -> np.array:
        """
        Randomly picks a point from the defined domain. Second element of domain must not
        be not reater than the first one.
        Return: numpy array with generated point.
        """
        # define minimum and maximum range of each  coordinate
        min_c = self._domain[0]  # first element of domain
        max_c = self._domain[1]  # second element of domain

        random_point = [
            random.randint(min_c, max_c),
            random.randint(min_c, max_c)
        ]

        return np.array(random_point)

    def __str__(self) -> None:
        """
        Describes the object's parameters.
        Return: None
        """
        return f"\nNewton's method for finding local optimum of a given function.\n"\
            "Parameters: \n"\
            f"> function: {self.function}\n"\
            f"> domain: {self._domain} x {self._domain}\n"\
            f"> starting point: {tuple(self._point)}\n"\
            f"> step size: {self._step_size}\n"\
            f"> sigma: {self._epsilon}\n"\
            f"> max iterations: {self._max_iterations}"

    @staticmethod
    def _calculate_gradient(functions: List[any], symbols: List[any], point: List[float]) -> np.array:
        """
        Calculates the vector of gradient for functions of two variables.
        functions: sympy objects. For this algorithm, they are derivatives for each variable
        symbols: sympy symbols. Basically names of variables used in sympy function object
        point: Point for which to calculate function's value
        Return: Vector saved as numpy array
        """
        # define sympy symbols
        x, y = sym.symbols('x y')

        x_c = point[0]
        y_c = point[1]
        x_symbol = symbols[0]
        y_symbol = symbols[1]

        result_vector = []
        v_component = None
        for function in functions:  # iterate through functions
            # substitude symbol for coordinates
            v_component = function.subs(x_symbol, x_c).subs(y_symbol, y_c)
            result_vector.append(v_component)

        return np.array(result_vector)

    def _partial(self, function, element):
        """
        Performs partial derivative of a function of several variables is its derivative
        with respect to one of those variables, with the others held constant.
        function: function to take partial derivative from.
        element: element to consider when taking derivative: sympy.core.symbol.Symbol.
        Return: partial_diff as sym object.
        """
        partial_diff = function.diff(element)

        return partial_diff

    @staticmethod
    def _substitude_value(function, x1, y1):
        """
        Performs substitution of point for fucntion:
        function: sympy object of function
        x1: x corodinate of considered point
        y1: y coordinate of considered point
        Return: value of the function in inserted coordinates
        """
        # define sympy symbols
        x, y = sym.symbols('x y')
        return function.subs(x, x1).subs(y, y1)

    def find_minimum(self) -> Tuple[List[float], int]:
        """
        This function implements Newton's method algorithm for finding minimum of a given function.
        Return: Tuple with approximate point with potential function's minimum and number of iterations.
        """

        # define sympy symbols
        x, y = sym.symbols('x y')
        symbols_list = [x, y]

        step_difference, n_of_iterations = 1, 0  # define starting values

        curr_point = self._point  # set current point

        self._step_points = []  # list containing each step point used for plotting

        # execute loop if the conditions are met
        while n_of_iterations < self._max_iterations and step_difference > self._epsilon:

            self._step_points.append(list(curr_point))  # append current point

            prev_point = curr_point.copy()  # set previous point as copy of the current one

            # Algorithm #

            partials, partials_second = [], []  # create lists for partial derivatives

            # calculate both partial derivatives
            for symbol in symbols_list:
                first_partial_derivative = self._partial(self.function, symbol)
                partials.append(first_partial_derivative)  # append calculated partial derivative

            # compute all four second partial derivatives
            for first_partial_drv in partials:
                col_drv = []
                for symbol in symbols_list:
                    second_partial_drv = self._partial(first_partial_drv, symbol)
                    col_drv.append(second_partial_drv)
                partials_second.append(col_drv)

            # subsitude each partial second for point
            partial_sec_val = []
            for partials_sec in partials_second:
                col_sec = []
                for partial_sec_drv in partials_sec:
                    val = self._substitude_value(partial_sec_drv, curr_point[0], curr_point[1])
                    col_sec.append(val)
                partial_sec_val.append(col_sec)

            partials_second = np.array(partial_sec_val, dtype=float)  # create matrix of sec partials

            partials_second_inv = np.linalg.inv(partials_second)  # inverse matrix

            vals = []
            for partial in partials:
                val = self._substitude_value(partial, curr_point[0], curr_point[1])
                vals.append(val)

            hessian_gradient = np.dot(partials_second_inv, np.array(vals)) * self._step_size

            curr_point = np.subtract(curr_point, hessian_gradient)  # calculate next point

            vec = abs(prev_point - curr_point)  # calculate vector created from points difference

            step_difference = np.linalg.norm(vec, ord=1)  # calculate magnitude of difference vector

            n_of_iterations += 1  # increment number of iterations

        self._p_minimum = curr_point  # set minimum point to attribute

        return curr_point, n_of_iterations


class SteepestGradientDescent:
    """
    Implementation of simple gradient descent algorithm.
    Read more in the :ref:`https://en.wikipedia.org/wiki/Gradient_descent`.

    Parameters
    ----------
    function : str
        Two-variable function that is defined and differentiable in a neighborhood of
        a considered point.
    domain : List[int], default=[-5, 5]
        Maximum interval for which the function is defined on in one dimension. For function with
        two variables as a domain will be considered a Cartesian product of two sets.
        For default [-5, 5]. The domain is [-5, 5] x [-5, 5].
    start_point : Tuple[int, int], default=None
        Coordinates of a point from which to start the algorithm run. If left blank, point will
        be chosen randomly from the domain.
    step_size : float, default=1e-2
        Step size influence the magnitude of impact of gradient on the next point.
    max_iterations : int, default=5000
        Maximum number of iterations in which the algorithm can find the function's minimum.
    epsilon : float, default=1e-12
        Represents the maximum acceptable and absolute difference between previous and next point
        calculated by gradient descent algorithm.
    """
    def __init__(self,
                 function,
                 domain: List[int] = [-5, 5],
                 start_point: Tuple[int] = None,
                 step_size: float = 1e-2,
                 max_iterations: int = 5000,
                 epsilon: float = 1e-12
                 ):
        self.function = self._convert_function(function)  # convert str function to sympy object
        self._domain = domain
        self._step_size = step_size
        self._epsilon = epsilon
        self._max_iterations = max_iterations

        if start_point:  # if points have not been passed
            self._point = np.array(start_point)  # assign to numpy array
        else:
            self._point = self._generate_point()

        self._step_points = None
        self._p_minimum = None

    def _convert_function(self, function):
        """
        Converts str function to sympify object.
        function: String containing function definition
        Return: Sympify object
        """
        return sym.sympify(function)

    def _generate_point(self) -> np.array:
        """
        Randomly picks a point from the defined domain. Second element of domain must not
        be not reater than the first one.
        Return: numpy array with generated point.
        """
        # define minimum and maximum range of each  coordinate
        min_c = self._domain[0]  # first element of domain
        max_c = self._domain[1]  # second element of domain

        random_point = [
            random.randint(min_c, max_c),
            random.randint(min_c, max_c)
        ]

        return np.array(random_point)

    def __str__(self) -> None:
        """
        Describes the object's parameters.
        Return: None
        """
        return f"\nGradient descent for finding local optimum of a given function.\n"\
            "Parameters: \n"\
            f"> function: {self.function}\n"\
            f"> domain: {self._domain} x {self._domain}\n"\
            f"> starting point: {tuple(self._point)}\n"\
            f"> step size: {self._step_size}\n"\
            f"> sigma: {self._epsilon}\n"\
            f"> max iterations: {self._max_iterations}"

    @staticmethod
    def _calculate_gradient(functions: List[any], symbols: List[any], point: List[float]) -> np.array:
        """
        Calculates the vector of gradient for functions of two variables.
        functions: sympy objects. For this algorithm, they are derivatives for each variable
        symbols: sympy symbols. Basically names of variables used in sympy function object
        point: Point for which to calculate function's value
        Return: Vector saved as numpy array
        """
        # define sympy symbols
        x, y = sym.symbols('x y')

        x_c = point[0]
        y_c = point[1]
        x_symbol = symbols[0]
        y_symbol = symbols[1]

        result_vector = []
        v_component = None
        for function in functions:  # iterate through functions
            # substitude symbol for coordinates
            v_component = function.subs(x_symbol, x_c).subs(y_symbol, y_c)
            result_vector.append(v_component)

        return np.array(result_vector)

    def find_minimum(self) -> Tuple[List[float], int]:
        """
        This function implements gradient descent algorithm for finding minimum of a given function.
        Return: Tuple with approximate point with potential function's minimum and number of iterations.
        """

        # define sympy symbols
        x, y = sym.symbols('x y')

        step_difference, n_of_iterations = 1, 0  # define starting values

        curr_point = self._point  # set current point

        self._step_points = []  # list containing each step point used for plotting

        # execute loop if the conditions are met
        while n_of_iterations < self._max_iterations and step_difference > self._epsilon:

            self._step_points.append(list(curr_point))  # append current point

            prev_point = curr_point.copy()  # set previous point as copy of the current one

            # calculate partial derivatives of function
            dfdx = sym.diff(self.function, x)
            dfdy = sym.diff(self.function, y)

            # define gradient
            gradient = self._calculate_gradient([dfdx, dfdy], [x, y], curr_point)

            # calculate new current point
            curr_point = curr_point - self._step_size * gradient

            difference_vector = abs(prev_point - curr_point)  # calculate vector created from points difference
            step_difference = np.linalg.norm(difference_vector, ord=1)  # calculate magnitude of difference vector

            n_of_iterations += 1  # increment number of iterations

        self._p_minimum = curr_point  # set minimum point to attribute

        return curr_point, n_of_iterations


if __name__ == '__main__':
    # If script got proper number of arguments
    if len(sys.argv) == 5:
        step = float(sys.argv[1])
        max_iter = int(sys.argv[2])
        start = [int(sys.argv[3]), int(sys.argv[4])]
    else:
        # run algorithms with default parameters
        step = 0.011  # 0.01
        max_iter = 3000
        start = [-5, -5]  # [5, 2]

    FUNCTION = "(x+2*y-7)**2 + (2*x+y-5)**2"

    # ====================================== #
    #             Newton's method            #
    # ====================================== #

    newton_descent = NewtonMethod(function=FUNCTION,
                                  start_point=start,
                                  step_size=step,
                                  max_iterations=max_iter)

    print(newton_descent)

    start_time = time.time()
    point, n_of_iterations = newton_descent.find_minimum()
    execution_t = time.time() - start_time

    # assign functions for plotting and execute it
    newton_descent.plot_function_descent = plot_function_descent(newton_descent, 'Newton')

    print("\nNewton's method")
    print(f'Method classificated point {point} as the local minimum of function.'
          f'\nNumber of iterations {n_of_iterations}')
    print('Execution time {:0.2f} [s]'.format(execution_t))

    print('---------------------------------------')

    # ====================================== #
    #   Steepest Gradient Descent method     #
    # ====================================== #

    gradient_descent = SteepestGradientDescent(function=FUNCTION,
                                               start_point=start,
                                               step_size=step,
                                               max_iterations=max_iter)

    print(gradient_descent)

    start_time = time.time()
    point, n_of_iterations = gradient_descent.find_minimum()
    execution_t = time.time() - start_time

    # assign functions for plotting and execute it
    # plots are saved automatically in the script directory
    gradient_descent.plot_function_descent = plot_function_descent(gradient_descent, 'Gradient')

    print("\nSteepest gradient descent method")
    print(f'\nMethod classificated point {point} as the local minimum of function.'
          f'\nNumber of iterations {n_of_iterations}')
    print('Execution time {:0.2f} [s]'.format(execution_t))
