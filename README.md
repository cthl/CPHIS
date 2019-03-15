# CHPIS
## Composable p-hierarchical solver

CPHIS is a composable implementation of p-hierarchical methods for the
solution of linear systems in discontinuous Galerkin discretizations with
hierarchical basis functions.
It is composable in the sense that CPHIS can not only be used within other
scientific software, but also that third-party linear algebra and solver
packages can be used within CPHIS.

CPHIS itself is entirely written in C99 and has no further dependencies,
although it will benefit from an OpenMP-capable compiler.
However, interfaces to third-party libraries might require additional compilers
and software installations.
