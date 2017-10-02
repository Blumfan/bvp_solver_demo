# bvp_solver_demo
This is a demo showcasing how one might use Python to solve partial differential equations. The purpose of me writing
this is primarily in order for me to learn the basics of Python, and to have something to showcase while I'm looking for
a job.

This is not software you want to be using for actually solving these problems; there are plenty of issues in this code
that need to be fixed before it would be even remotely competitive. with proffesional solvers. (For instance, the heat
equation needs an algorithm which is more stable than a unicycle.)

To run just run main.py, it should work out of the box provided you have a modern version of python, and numpy and matplotlib.

Without changing anything, the program simulates the wave equation. The easiest interpretation is that of an elastic
string. In the simulation the position on the left side is fixed, while the string is allowed to move freely
in the vertical direction on the right-most side. The left edge is moved up for a short period of time, and then
keeps it fixed at zero for the remained of the simulation.

A slightly more complicated setup (but somewhat more realistic) is to interpret the function as a voltage in a cable.
In that interpretation a short pulse is sent in via a perfect conductor on the left, and meets a perfect insulator on
the right.

All of the complicated stuff happens in simulation.py, which I've tried to keep as neat and straightforward as possible. It is
however, slightly more complicated than need be since I wanted to do some object oriented stuff.


Enjoy!
Jesper Johansson
