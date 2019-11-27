![CPHIS icon](cphis.png)

# CHPIS
## Composable p-hierarchical solver

CPHIS is a composable implementation of p-multigrid methods for the
solution of linear systems in discontinuous Galerkin discretizations with
hierarchical basis functions.
It is composable in the sense that CPHIS can be used within other
scientific software, and third-party linear algebra and solver
packages can be used within CPHIS.

CPHIS itself is entirely written in C99 and has no further dependencies,
although it will benefit from an OpenMP-capable compiler.
However, interfaces to third-party libraries might require additional compilers
and software installations.

Besides its stand-alone OpenMP-based linear algebra backend, CPHIS currently
supports the Tpetra package of the Trilinos project for hybrid parallel
linear algebra.
