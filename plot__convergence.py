from math import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


# accuracy tables
# ============================================================
table   = 'accuracy'

# settings
# ============================================================
FileOut = 'fig__PlaneWave_Test'


ax = plt.axes( )


# line styles
# ============================================================
LStyle_Dot     = [1, 2]
LStyle_Dash    = [4, 2]
LStyle_DashDot = [4, 2, 1, 2]


# load data
# ============================================================
accuracy   = np.loadtxt( table,   usecols=(1), unpack=True )
N          = np.loadtxt( table,   usecols=(0), unpack=True )


# plot the performance
# ============================================================
resolution=np.asarray( [2**n for n in range(4,12)] )

ax.plot( N, accuracy                  , 'r-o', mec='k', lw=2, ms=6,                        label='number density $(D)$' )
ax.plot( N, accuracy[0]*(N/N[0])**-1.0, 'k-',  mec='r', lw=2, ms=6, dashes=LStyle_DashDot, label='$N^{-1.0}$' )
ax.plot( N, accuracy[0]*(N/N[0])**-2.0, 'k-',  mec='g', lw=2, ms=5, dashes=LStyle_DashDot, label='$N^{-2.0}$' )
ax.plot( N, accuracy[0]*(N/N[0])**-3.0, 'k-',  mec='b', lw=2, ms=4, dashes=LStyle_DashDot, label='$N^{-3.0}$' )
ax.plot( N, accuracy[0]*(N/N[0])**-4.0, 'k-',  mec='k', lw=2, ms=4, dashes=LStyle_DashDot, label='$N^{-4.0}$' )

# set axis
# ============================================================
ax.set_xscale( 'log' )
ax.set_yscale( 'log' )
ax.set_xlim( 10, 1000 )
#ax.set_ylim( 1e-5, 1e-1 )
ax.set_xlabel( 'number of cells', fontsize='large' )
ax.set_ylabel( 'L1 error $(D)$',       fontsize='large' )
ax.tick_params( which='both', tick2On=True, direction='in' )

# add legend
# ============================================================
ax.legend( loc='lower left', numpoints=1, labelspacing=0.1, handletextpad=0.4,
              borderpad=0.4, handlelength=2.7, fontsize='large' )




# save/show figure
#plt.savefig( FileOut+".png", bbox_inches='tight', pad_inches=0.05 )
#plt.savefig( FileOut+".pdf", bbox_inches='tight', pad_inches=0.05 )
plt.show()
