from fenics import *
from mshr import *
from math import pi, sin, cos
T = 2.0            # final time
num_steps = 50     # number of time steps
dt = T / num_steps # time step size
alpha = 3          # parameter alpha
beta = 1.2         # parameter beta

# Create mesh and define function space
R = 1.1
r = 0.4
t = 10
x1 = R*cos(float(t) / 180 * pi)
y1 = 0
z1 = R*sin(t)

# Create geometry
s1 = Sphere(Point(0, 0, 0), 1)
s2 = Sphere(Point(x1, y1, z1), r)
b1 = Box(Point(-2, -2, -0.03), Point(2, 2, 0.03))
geometry = s1 - s2 - b1

# Create mesh
mesh = generate_mesh(geometry, 32)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition
u_D = Expression('-1 + exp(x[1]) + beta*exp(x[2]) + alpha*t*t', degree=2, alpha=alpha, beta=beta, t=0)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# Define initial value
u_n = interpolate(u_D, V)
#u_n = project(u_D, V)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(beta - 2 - 2*alpha)

F = u*v*dx + dt*dot(grad(u), grad(v))*dx - (u_n + dt*f)*v*dx
a, L = lhs(F), rhs(F)

res_file = File('heat/solution.pvd')

# Time-stepping
u = Function(V)
t = 0
for n in range(num_steps):

    # Update current time
    t += dt
    u_D.t = t

    # Compute solution
    solve(a == L, u, bc)

    # Save solution to VTK
    res_file << u

    # Update previous solution
    u_n.assign(u)