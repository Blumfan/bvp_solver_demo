import simulation
import math

# This is the file to run in order to give this demo a spin. It should work out of the box.
#
# Some parameters can be changed, such as initial conditions, boundary conditions, and which physical problem you want
# to solve. As of now there is only the wave equation and the heat equation, but if you're really ambitious I guess
# you could add your own. Adding a dissipative wave equation wouldn't be that hard, just add a dy/dt term.

# SIMULATION PARAMETERS
string_length = 1
"""The length of the string. This is not really a relevant parameter for any problem; both the heat equation and the
wave equation are scale-invariant, so we can always scale the problem to have length 1. But I've included this parameter
for the sake of completion."""
number_of_points = 100
"""The number of x-coordinates in the simulation. Making this too large leads to instabilities."""
time_step = 0.01
"""The time in between each time-step. It is also used for controlling the frame-rate of the animation."""
# END SIMULATION PARAMETERS


# USEFUL FUNCTIONS GENERATORS
def constant(value=0):
    """A flat initial condition. Rather boring, you probably want a source or some interesting boundary conditions
    for this to be fun!

    Args:
        value (float): The value the function takes everywhere. Defaults to 0."""
    return lambda x: value


def gaussian_wave_packet(amplitude=1, position=0.5, width=0.1):
    """Creates a gaussian wave-packet with the given parameters. This is a good standard displacement for an initial
    condition, since it is very smooth. Do not use a box pulse; they are not realistic and do not play well with
    numerical solvers.

    Args:
        amplitude (float): The height of the wave-packet. Defaults to 1.
        position (float): The center of the wave-packet. Is measured in units of the string_length (i.e., if you want
            to have the packet in the middle of the string, the parameter should be 0.5). Defaults to 0.5.
        width (float): The width of the wave-packet. This is the standard deviation of the wave-packet, measured in
            units of the string length. Defaults to 0.1"""
    return lambda x: amplitude*math.exp(-0.5*((x - position*string_length)/(width*string_length))**2)


def standing_wave(amplitude=1, node_count=3, base_period=string_length):
    """Creates a function representing a standing sine-wave. Mostly useful for the wave equation but it could be fun
     also in the heat equation, I suppose.

     Args:
         amplitude (float): The amplitude of the standing wave. The largest value the function attains. Defaults to 1.
         node_count (int): The number of nodes in the standing wave. Defaults to 3.
         base_period (float): The length of a single period at node_count 1. Defaults to the string_length."""
    return lambda x: amplitude*math.sin(node_count*math.pi*x/base_period)
# END USEFUL FUNCTION GENERATORS


# PROBLEM CONDITIONS
initial_conditions = constant()  # Change this to change the initial conditions.
"""The choice of initial condition. This should be a function which takes a single real argument and produces a 
number."""

general_boundary_condition = simulation.BVPBoundaryCondition(
    left_dirichlet_parameter=1,
    right_dirichlet_parameter=0,
    left_neumann_parameter=0,
    right_neumann_parameter=1,
    left_constant=gaussian_wave_packet(),
    right_constant=constant())
homogeneous_dirichlet = simulation.BOUNDARY_CONDITION_HOMOGENEOUS_DIRICHLET
homogeneous_neumann = simulation.BOUNDARY_CONDITION_HOMOGENEOUS_NEUMANN

boundary_conditions = general_boundary_condition  # Change this to change the boundary conditions


def source(t, x):
    """The function used as source function for the BVP. This is a function of both time and position. Change this to
    add a heat source for the heat-equation, or a charge for the wave-equation (if we think in terms of physics) or a
    force (if we think of it as a string.)"""
    return 0


wave_equation_descriptor = simulation.WaveEquationDescriptor(wave_speed=1, boundary_conditions=boundary_conditions,
                                                             source_function=source)
heat_equation_descriptor = simulation.HeatEquationDescriptor(diffusivity=0.01, boundary_conditions=boundary_conditions,
                                                             source_function=source)
# The diffusivity of the heat equation is low to allow convergence.

problem_descriptor = wave_equation_descriptor  # Change this to change equation.
""" Selects the problem we actually want to solve."""
# END PROBLEM CONDITIONS

# THE ACTUAL SIMULATION
# No changes needed here, all parameters should be changed at the appropriate place above.
simulation.sim_and_plot(simulation.BVPSimulation(
    length=string_length, time_step=time_step, number_of_points=number_of_points, bvp_descriptor=problem_descriptor,
    initial_conditions=initial_conditions))
