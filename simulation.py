import numpy
import matplotlib.pyplot as pyplot
from math import fabs


"""This is where all the heavy lifting is done. There are three big classes.

BVPBoundaryCondition, which implements boundary conditions and is responsible for updating the vector of function 
values at the edges after each time-step.

BVPDescriptor, which describes a Boundary Value Problem in an abstract mathematical sense. It does not concern itself
with vectors of function values, but it does provide functions to handle them. Contains information about the partial
differential equation as well as the boundary conditions through a BVPBoundaryCondition

BVPSimulation, which take a BVPDescriptor and actually makes a concrete instance of a numerical solution to the problem,
complete with coordinates and functions to update the state of the simulation.

There is also a function sim_and_plot which takes a BVPSimulation and renders the solution in realtime as it is solved.
"""


class BVPBoundaryCondition(object):
    """Class provides an abstract description of the conditions imposed at the boundary of a Boundary Value Problem
    (BVP) and ways of applying it to a concrete instance of the problem. It contains the parameters for both x = 0 and
    for maximal X

    All the valid boundary value problems are of the form
        ay + by' = c(t),
    where a and b cannot both be zero. The parameter c is allowed to be time dependent. The typical use is that one of
    a or b are 0 and the other 1, with c(t) handling the dependence on time. In that case the interpretation is simple.
    For Dirichlet conditions (a=1, b=0), the function value at the boundary is set to c(t), while for Neumann conditions
    (a=0, b=1) the derivative at the boundary is set to c(t).

    Mixing of the two types of boundary conditions are also possible.

    Attributes:
        left_dirichlet_parameter (float): The parameter a for x = 0
        right_dirichlet_parameter (float): The parameter a for maximal x.
        left_neumann_parameter (float): The parameter b for x = 0
        right_neumann_parameter (float): The parameter b for maximal x.
        left_constant (float): The parameter c for x = 0
        right_constant (Callable): The parameter c for maximal x.

    Raises:
        ValueError: If both a and b are zero at either boundary."""
    def __init__(self, left_dirichlet_parameter, left_neumann_parameter, right_dirichlet_parameter,
                 right_neumann_parameter, left_constant, right_constant):
        self.left_dirichlet_parameter = left_dirichlet_parameter
        self.right_dirichlet_parameter = right_dirichlet_parameter
        self.left_neumann_parameter = left_neumann_parameter
        self.right_neumann_parameter = right_neumann_parameter
        self.left_constant = left_constant
        self.right_constant = right_constant

        if self.left_dirichlet_parameter == 0 and self.left_neumann_parameter == 0:
            raise ValueError("Combination of boundary conditions is singular! Both left boundary parameters are zero!")
        if self.right_dirichlet_parameter == 0 and self.right_neumann_parameter == 0:
            raise ValueError("Combination of boundary conditions is singular! Both right boundary parameters are zero!")

    def apply_boundary_conditions(self, t, value_vector, dx):
        """Applies the boundary condition at a given time to the supplied vector.

        Args:
            t (float): The time to supply the constant function c(t). Usually the time in the simulation.
            value_vector (float[]): An array of values containing the function. Is modified by the function.
            dx (float): The spacing between the points in the grid.
            """
        value_vector[0] = (self.left_constant(t) - self.left_neumann_parameter*value_vector[1]/dx) \
            / (self.left_dirichlet_parameter - self.left_neumann_parameter/dx)
        value_vector[-1] = (self.right_constant(t) + self.right_neumann_parameter*value_vector[-2]/dx) \
            / (self.right_dirichlet_parameter + self.right_neumann_parameter/dx)

    @classmethod
    def create_dirichlet_boundary_conditions(cls, left_constant, right_constant):
        """
        Args:
            left_constant (Callable): The function c(t) applying to x=0
            right_constant (Callable): The function c(t) applying for maximal x.
        Return: A pure Dirichlet boundary condition (i.e. with b = 0 at both ends.)
            """
        return BVPBoundaryCondition(1, 0, 1, 0, left_constant, right_constant)

    @classmethod
    def create_neumann_boundary_conditions(cls, left_constant, right_constant):
        """
        Args:
            left_constant (Callable): The function c(t) applying to x=0
            right_constant (Callable): The function c(t) applying for maximal x.
        Return: A pure Neumann boundary condition (i.e. with a = 0 at both ends.)
            """
        return BVPBoundaryCondition(0, 1, 0, 1, left_constant, right_constant)


BOUNDARY_CONDITION_HOMOGENEOUS_DIRICHLET = BVPBoundaryCondition.create_dirichlet_boundary_conditions(lambda t: 0,
                                                                                                     lambda t: 0)
"""A ready made boundary condition representing a homogeneous Dirichlet boundary condition at both ends. This means that
a = 1, b = 0 and c(t) = 0 at both ends."""

BOUNDARY_CONDITION_HOMOGENEOUS_NEUMANN = BVPBoundaryCondition.create_neumann_boundary_conditions(lambda t: 0, lambda t: 0)
"""A ready made boundary condition representing a homogeneous Neumann boundary condition at both ends. This means that
a = 0, b = 1 and c(t) = 0 at both ends."""


class BVPDescriptor(object):
    """ Class providing the abstract mathematical description of a Boundary Value Problem (BVP), and how to solve it
    mumerically. It is essentially a container of functions needed in order to solve a concrete discretised BVP. This is
    an abstract class, and the function update_function has to be provided.

    A more sophisticated piece of software would probably split this up into a BVPDescriptor and a dedicated BVPSolver,
    but this will do for this demo.

    Attributes:
       boundary_conditions (BVPBoundaryCondition): A container of the boundary conditions for this BVP. Can by of any
       kind.
       source_function (Callable): A function which takes two floats, time and position. It should not be vectorised.
    """
    def __init__(self, boundary_conditions, source_function):
        self.boundary_conditions = boundary_conditions
        self.source_function = lambda t, x: numpy.vectorize(lambda scalar_x: source_function(t, scalar_x))(x)
        # The above convoluted mess makes source_function into a function vectorised in x.

    def update_function(self, t, dt, x, y, y_previous, laplacian):
        """The function that determines how the boundary value problem behaves, as well as how it is solved numerically.
        This method is not implemented in the base BVPDescriptor class.

        Its mathematical function is simple, given a vector y of function values the vector y_updated in the following
        timestep is given by
          y_updated = update_function.

        As of now, the function only accepts the Laplacian operator, because we only consider the heat equation and the
        wave equation in this demo. If we want to consider also operators like dy/dx or d^2y/dxdt we would have to
        create some sort of Geometry class which handles this. It is not necessary for this demo.

        Args:
            t (float): The time in the simulation.
            dt (float) The time-step.
            x (float[]): A vector containing the coordinates.
            y (float[]): A vector containing the current function values.
            y_previous (float[]): A vector containing the function values in the previous time-step.
            laplacian: A numpy matrix which calculates an approximation of the second derivative in x of the y vector.
        """
        raise NotImplementedError("This is just an abstract BVPDescriptor")


class WaveEquationDescriptor(BVPDescriptor):
    """A sample BVPDescriptor for the Wave Equation,
         d^2y/dt^2 = c^2 d^2y/dx^2 + source_function.
    Defaults to homogeneous Neumann boundary conditions (zero derivative at the edges. This corresponds to no flux out
    of the rod, for most physical interpretations.), with a wave-speed of 1 and no source function.

    Attributes:
        wave_speed (float): The speed at which waves travel, the parameter c in the equation."""
    def __init__(self, wave_speed=1, boundary_conditions=BOUNDARY_CONDITION_HOMOGENEOUS_NEUMANN,
                 source_function=lambda t, x: 0):
        super().__init__(boundary_conditions, source_function)
        self.wave_speed = wave_speed

    """ The update function for this solver of the wave equation. It uses a central finite difference approximation of
    the double time derivative in order to perform the timestep,
         d^2y/dt^2 ~= (y_previous - 2y + y_next)/dt^2"""
    def update_function(self, t, dt, x, y, y_previous, laplacian):
        return 2*y - y_previous + ((dt*self.wave_speed)**2)*numpy.dot(laplacian, y) + (dt**2)*self.source_function(t, x)


class HeatEquationDescriptor(BVPDescriptor):
    """A BVPDescriptor for the heat Equation,
        dy/dt = a d^2y/dx^2 + source_function.
    Defaults to homogeneous Neumann boundary conditions (zero derivative at the edges. Physically this corresponds to
    perfect insulators at both ends), a diffusivity of 1 and no source function.

    This implementation is currently extremely unstable, and requires a small diffusivity in order to not explode.

    Attributes:
        diffusivity (float): The parameter a in the heat equation. A large diffusivity means that heat spreads quickly.
        """
    def __init__(self, diffusivity=1, boundary_conditions=BOUNDARY_CONDITION_HOMOGENEOUS_NEUMANN,
                 source_function=lambda t, x: 0):
        super().__init__(boundary_conditions, source_function)
        self.diffusivity = diffusivity

    def update_function(self, t, dt, x, y, y_previous, laplacian):
        """ The update function for this solver of the heat equation. It uses a simple Euler forward method for
        for performing the time-step,
            dy/dt = (y_next - y)/dt.
        This is an extremely unstable method, and demands that either the diffusivity is small or that the spacing
        between the points is large. (Basically, diffusivity*dt/dx^2 has to be small.)"""
        return y + self.diffusivity*dt*numpy.dot(laplacian, y) + dt*self.source_function(t, x)


class BVPSimulation(object):
    """Class handling simulation of a concrete instance of a Boundary Value Problem (BVP). This differs from the
    BVPDescriptor in that it contains an actual instance of the BVP, with function values and a specified geometry.

    Attributes:
        time_step (float): The length of each time_step.
        length (float): The length of the string
        number_of_points (int): The number of coordinate points to use.
        bvp_descriptor (BVPDescriptor): The abstract description of the problem to solve.
        x (float[]): A vector containing coordinate values
        dx (float): The spacing between coordinate points, which is constant in this demo.
        time (float): The time that has passed in the simulation.
        y (float[]): A vector containing the current function values.
        y_previous (float[]): A vector containing the function values as they were in the previous time-step.
    """
    def __init__(self, time_step, length, number_of_points, bvp_descriptor, initial_conditions,
                 pre_initial_conditions=None):
        """
        Args:
            time_step (float):
            length (float):
            number_of_points (int):
            bvp_descriptor (BVPDescriptor):
            initial_conditions (Callable): A function that specifies what the function values at t=0 should be. Should
                not be vectorised.
            pre_initial_conditions (Callable): A function that specifies what the function values where at t=-time_step.
                If the BVP is second order in time this is relevant. If it is None it is set to the same
                values as the initial_conditions. Defaults to None.
        """
        self.bvp_descriptor = bvp_descriptor
        self.time_step = time_step
        self.length = length
        self.x, self.dx = numpy.linspace(0, self.length, number_of_points, retstep=True)

        self.time = 0
        self.y = numpy.vectorize(initial_conditions)(self.x)
        if pre_initial_conditions is None:
            self.y_previous = self.y
        else:
            self.y_previous = numpy.vectorize(pre_initial_conditions(self.x))

        self._laplacian_matrix = numpy.eye(number_of_points)
        self._laplacian_matrix[range(1, number_of_points - 1), range(1, number_of_points - 1)] = -2
        self._laplacian_matrix[0, 0] = -1
        self._laplacian_matrix[-1, -1] = -1
        self._laplacian_matrix[range(1, number_of_points), range(0, number_of_points - 1)] = 1
        self._laplacian_matrix[range(0, number_of_points - 1), range(1, number_of_points)] = 1
        self._laplacian_matrix /= (self.dx**2)
        """The laplacian matrix implementing the second order x derivative. Also known as the stiffness matrix."""

    def step_simulation(self):
        """Brings the simulation one time-step forward, using the method provided by the BVPDescriptor. Updates the
        time, y, and y_previous."""
        new_previous = self.y
        self.y = self.bvp_descriptor.update_function(self.time, self.time_step, self.x, self.y, self.y_previous,
                                                     self._laplacian_matrix)
        self.bvp_descriptor.boundary_conditions.apply_boundary_conditions(self.time, self.y, self.dx)
        self.time += self.time_step
        self.y_previous = new_previous


def sim_and_plot(simulation, max_time=-1):
    """Runs a BVPSimulation for some time (or until the program is closed) and plots the results using pyplot. This
    function assumes that no other pyplot windows are opened while it is running.

    Args:
        simulation (BVPSimulation): The instance we're going to be simulating.
        max_time (float): The time to end the simulation. If this is negative the simulation runs until the window is
            closed. Defaults to -1 (and thus shows the window until it is closed.)
        """
    max_initial_deviation = max(numpy.vectorize(fabs)(simulation.y))
    if max_initial_deviation == 0:  # Basically a safe-guard for when we start with nothing.
        max_initial_deviation = 1
    # max_initial_deviation determines how big the vertical scale should be in the axis.

    # Creates a window and gives it the right settings
    pyplot.axis([0, max(simulation.x), -2*max_initial_deviation, 2*max_initial_deviation])
    pyplot.xlabel("x")
    pyplot.ylabel("y(t,x)")
    pyplot.ion()  # Makes the plot interactive, meaning we can redraw it and do other things while it's up.
    points = pyplot.plot(simulation.x, simulation.y)[0]
    # points cointains the points of our graph, and saves them so we can update the graph dynamically later. This is
    # better than clearing the axes and calling pyplot.plot again, since that means we'd have to recreate all elements
    # of the graph.

    fignum_count = len(pyplot.get_fignums())
    # Used for knowing when the window has been closed. This number will change when the window has closed.
    show_after = False
    while len(pyplot.get_fignums()) == fignum_count:
        simulation.step_simulation()  # Advance the simulation
        points.set_data(simulation.x, simulation.y)  # This is faster than calling plot. See above.
        pyplot.title('String at t={:.3f}'.format(simulation.time))
        pyplot.pause(simulation.time_step)  # Also draws the plot!

        if 0 < max_time < simulation.time:
            show_after = True
            break

    # If we didn't close the window, show it till it closes.
    if show_after:
        pyplot.ioff()
        pyplot.show()
