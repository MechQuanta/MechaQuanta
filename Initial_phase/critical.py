from scipy import interpolate
from numpy import zeros
from numpy.linalg import inv
from numpy import (
	dot,
	linspace,
	argmax,
	abs,
	clip,
	sin,
	cos,
	pi,
	amax,
	arctan2,
	sqrt,
	sum,
)
import numpy as np
from warnings import warn

def find_critical(R,Z,discard_xpoints = True):
	f = interpolate.RectBivariateSpline(R[:,0],Z[0,:],psi)
	Bp2 = (f(R,Z,dx = 1,grid = False)**2+ f(R,Z,dy=1,grid = False)**2)/R**2
