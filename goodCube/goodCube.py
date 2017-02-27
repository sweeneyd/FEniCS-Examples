"""
Created on Fri Jan 27 10:10:19 2017

@author: Dan

To run: 1.) Install FEniCS in Docker
        2.) Navigate to the ~/home menu in the terminal
        3.) docker run -ti -v $(pwd):/home/fenics/shared quay.io/fenicsproject/stable
        4.) cd shared/Google\ Drive/Papers/Single\ Pulse/Modeling/Cell
"""
from fenics import *
import os
import numpy as np
import matplotlib.pyplot as plt

## Convert mesh from GMSH format
ifilename = 'cellMesh3.msh'
ofilename = ifilename.split('.')[0]+'.xml'
if not os.path.isfile('%s'%ofilename):
    os.system('dolfin-convert %s %s'%(ifilename, ofilename))

# Import the mesh and determine relevant boundaries/subdomains
mesh = Mesh(ofilename)
subdomains = MeshFunction("size_t", mesh, "%s_physical_region.xml"%ofilename.split('.')[0])
boundaries = MeshFunction("size_t", mesh, "%s_facet_region.xml"%ofilename.split('.')[0])

# Rename subdomain numbers from GMSH
rename_subdomains = CellFunction("size_t", mesh)
rename_subdomains.set_all(0)
rename_subdomains.array()[subdomains.array()==0] = 1
rename_subdomains.array()[subdomains.array()==1] = 2

# Rename boundary numbers from GMSH
rename_boundaries = FacetFunction("size_t", mesh)
rename_boundaries.set_all(0)
rename_boundaries.array()[boundaries.array()==55] = 1
rename_boundaries.array()[boundaries.array()==56] = 2
rename_boundaries.array()[boundaries.array()==57] = 3

# print([len(rename_boundaries.array()[rename_boundaries.array()==3]),
#        len(rename_boundaries.array()[rename_boundaries.array()==2]),
#        len(rename_boundaries.array()[rename_boundaries.array()==1]),
#        len(rename_boundaries.array()[rename_boundaries.array()==0])])
#
# print([len(rename_subdomains.array()[rename_subdomains.array()==2]),
#        len(rename_subdomains.array()[rename_subdomains.array()==1]),
#        len(rename_subdomains.array()[rename_subdomains.array()==0])])

# Rename interface numbers from GMSH
rename_interface = FacetFunction("size_t", mesh)
rename_interface.set_all(0)
rename_interface.array()[boundaries.array()==33] = 1

# Generate test/trial functions in function space
V = FunctionSpace(mesh, 'CG', 1)
u = TrialFunction(V)
v = TestFunction(V)
dx = Measure("dx", domain=mesh, subdomain_data=rename_subdomains)
ds = Measure("ds", domain=mesh, subdomain_data=rename_boundaries)

## Define boundary conditions
source = 1.0
sink = 0.0

bcs = [ DirichletBC(V, source, rename_boundaries, 1),
        DirichletBC(V, sink, rename_boundaries, 2),
        DirichletBC(V, sink, rename_boundaries, 3)]

# Define material properties
pi = 1e-9
po = 1.0

## Assemble parts of problem
f = Constant(0)
g = Constant(0)
a0 = inner(po*nabla_grad(u), nabla_grad(v))*dx(1)
a1 = inner(pi*nabla_grad(u), nabla_grad(v))*dx(2)
a = a0 + a1
L = f*v*dx(1) + f*v*dx(2)

## Redefine u as a function in function space V for the solution
u = Function(V)

## Solve PDE and store solution in u
set_log_level(PROGRESS)
solve(a == L, u, bcs)

## Save solution to file in VTK format
ofile = 'Cell/StaticCellModel_asdf.pvd'
print('  [+] Output to %s'%(ofile))
vtkfile = File(ofile)
vtkfile << u
