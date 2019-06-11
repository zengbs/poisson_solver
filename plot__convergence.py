from math import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


# accuracy tables
# ============================================================
table1   = 'accuracy_SOR'
table2   = 'accuracy_GS'
table3   = 'accuracy_Jacobi'

# settings
# ============================================================
FileOut = 'summary'


f, ax = plt.subplots( 2, 2, sharex=False, sharey=False )
f.subplots_adjust( hspace=0.1, wspace=0.2 )
f.set_size_inches( 13.0, 13.0 )

# line styles
# ============================================================
LStyle_Dot     = [1, 2]
LStyle_Dash    = [4, 2]
LStyle_DashDot = [4, 2, 1, 2]


# load data
# ============================================================
SOR       = np.loadtxt( table1,   usecols=(0,1,2,3,4), unpack=True )
GS        = np.loadtxt( table2,   usecols=(0,1,2,3,4), unpack=True )
Jacobi    = np.loadtxt( table3,   usecols=(0,1,2,3,4), unpack=True )
N         = np.loadtxt( table3,   usecols=(0)        , unpack=True )

# plot the performance
# ============================================================
#L1 error
ax[0][0].plot( N,    SOR[1]            , 'r-o', mec='r', lw=2, ms=20,                label='SOR' )
ax[0][0].plot( N,     GS[1]            , 'g-v', mec='g', lw=2, ms=15,                label='Gauss-Seidel' )
ax[0][0].plot( N, Jacobi[1]            , 'b-x', mec='b', lw=2, ms=20,                label='Jacobi' )
ax[0][0].plot( N, Jacobi[1][0]*(N/N[0])**-2.0, 'k-',   lw=2, ms=5, dashes=LStyle_DashDot, label='$N^{-2}$' )

#elapsed time
ax[1][0].plot( N,    SOR[2]            , 'r-o', mec='k', lw=2, ms=6,                label='SOR' )
ax[1][0].plot( N,     GS[2]            , 'g-o', mec='k', lw=2, ms=6,                label='Gauss-Seidel' )
ax[1][0].plot( N, Jacobi[2]            , 'b-o', mec='k', lw=2, ms=6,                label='Jacobi' )
ax[1][0].plot( N, 0.1*(N/N[0])**4.0, 'k-',     lw=2, ms=5, dashes=LStyle_DashDot, label='$N^{4}$' )
ax[1][0].plot( N, SOR[2][0]*(N/N[0])**2.0, 'k>',     lw=2, ms=5, dashes=LStyle_DashDot, label='$N^{2}$' )

#iterations
ax[0][1].plot( N,    SOR[3]            , 'r-o', mec='k', lw=2, ms=6,                label='SOR' )
ax[0][1].plot( N,     GS[3]            , 'g-o', mec='k', lw=2, ms=6,                label='Gauss-Seidel' )
ax[0][1].plot( N, Jacobi[3]            , 'b-o', mec='k', lw=2, ms=6,                label='Jacobi' )
ax[0][1].plot( N, 1e4*(N/N[0])**2.0, 'k-',     lw=2, ms=5, dashes=LStyle_DashDot, label='$N^{2}$' )
ax[0][1].plot( N, 300*(N/N[0]), 'kv',     lw=2, ms=5, dashes=LStyle_DashDot, label='$N^{1}$' )

#performance
ax[1][1].plot( N,    SOR[4]            , 'r-o', mec='k', lw=2, ms=6,                label='SOR' )
ax[1][1].plot( N,     GS[4]            , 'g-o', mec='k', lw=2, ms=6,                label='Gauss-Seidel' )
ax[1][1].plot( N, Jacobi[4]            , 'b-o', mec='k', lw=2, ms=6,                label='Jacobi' )
ax[1][1].plot( N, 1e4*(N/N[0])**-2.0, 'k-',     lw=2, ms=5, dashes=LStyle_DashDot, label='$N^{-2}$' )

# set axis
# ============================================================
ax[0][0].set_xscale( 'log' )
ax[0][0].set_yscale( 'log' )
ax[0][0].set_xlim( 10, 1000 )
ax[0][0].set_xlabel( 'number of cells ($N$)', fontsize='large' )
ax[0][0].set_ylabel( 'L1 error $(D)$',       fontsize='large' )
ax[0][0].tick_params( which='both', tick2On=True, direction='in' )

ax[1][0].set_xscale( 'log' )
ax[1][0].set_yscale( 'log' )
ax[1][0].set_xlim( 10, 1000 )
ax[1][0].set_xlabel( 'number of cells ($N$)', fontsize='large' )
ax[1][0].set_ylabel( 'elapsed time (sec)',       fontsize='large' )
ax[1][0].tick_params( which='both', tick2On=True, direction='in' )

ax[0][1].set_xscale( 'log' )
ax[0][1].set_yscale( 'log' )
ax[0][1].set_xlim( 10, 1000 )
ax[0][1].set_xlabel( 'number of cells ($N$)', fontsize='large' )
ax[0][1].set_ylabel( 'number of iterations',       fontsize='large' )
ax[0][1].tick_params( which='both', tick2On=True, direction='in' )

ax[1][1].set_xscale( 'log' )
ax[1][1].set_yscale( 'log' )
ax[1][1].set_xlim( 10, 1000 )
ax[1][1].set_xlabel( 'number of cells ($N$)', fontsize='large' )
ax[1][1].set_ylabel( 'performance (cells/sec)',       fontsize='large' )
ax[1][1].tick_params( which='both', tick2On=True, direction='in' )
# add legend
# ============================================================
ax[0][0].legend( loc='lower left', numpoints=1, labelspacing=0.1, handletextpad=0.4,
              borderpad=0.4, handlelength=2.7, fontsize='large' )

ax[1][0].legend( loc='upper left', numpoints=1, labelspacing=0.1, handletextpad=0.4,
              borderpad=0.4, handlelength=2.7, fontsize='large' )

ax[0][1].legend( loc='upper left', numpoints=1, labelspacing=0.1, handletextpad=0.4,
              borderpad=0.4, handlelength=2.7, fontsize='large' )

ax[1][1].legend( loc='lower left', numpoints=1, labelspacing=0.1, handletextpad=0.4,
              borderpad=0.4, handlelength=2.7, fontsize='large' )

# save/show figure
plt.savefig( FileOut+".png", bbox_inches='tight', pad_inches=0.05 )
#plt.savefig( FileOut+".pdf", bbox_inches='tight', pad_inches=0.05 )
#plt.show()
