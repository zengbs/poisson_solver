from math import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


# accuracy tables
# ============================================================
table_032  = 'Output__ComputingSpeed_vs_NumThreads_NUM_NX_32'
table_064  = 'Output__ComputingSpeed_vs_NumThreads_NUM_NX_64'
table_128  = 'Output__ComputingSpeed_vs_NumThreads_NUM_NX_128'
table_256  = 'Output__ComputingSpeed_vs_NumThreads_NUM_NX_256'
table_512  = 'Output__ComputingSpeed_vs_NumThreads_NUM_NX_512'
table_1024  = 'Output__ComputingSpeed_vs_NumThreads_NUM_NX_1024'
table_2048  = 'Output__ComputingSpeed_vs_NumThreads_NUM_NX_2048'

# settings
# ============================================================
FileOut = 'Output'


f, ax = plt.subplots( 1, 2, sharex=False, sharey=False )
f.subplots_adjust( hspace=0.1, wspace=0.2 )
f.set_size_inches( 13.0, 7.0 )


# load data
# ============================================================
table_032  = np.loadtxt(table_032 ,   usecols=(0,1), unpack=True )
table_064  = np.loadtxt(table_064 ,   usecols=(0,1), unpack=True )
table_128  = np.loadtxt(table_128 ,   usecols=(0,1), unpack=True )
table_256  = np.loadtxt(table_256 ,   usecols=(0,1), unpack=True )
table_512  = np.loadtxt(table_512 ,   usecols=(0,1), unpack=True )
table_1024 = np.loadtxt(table_1024 ,  usecols=(0,1), unpack=True )
table_2048 = np.loadtxt(table_2048 ,  usecols=(0,1), unpack=True )



# plot computing speed
# ============================================================
ax[0].plot( table_1024[1], table_1024[0], 'b-o', mec='b', lw=2, ms=8, label='NumGrids=1024' )
ax[0].plot( table_1024[1], table_1024[0][0]*table_1024[1] , 'k-.', mec='k', lw=2, ms=5, label=r'$\propto N$' )

# plot parallel efficiency
# ============================================================
ParallalEfficiency = [0]*6
ParallalEfficiency[0]  = table_064[0][15]/table_064[0][0]   # speed-up
ParallalEfficiency[0] /= table_064[1][15]                   # number of threads

ParallalEfficiency[1]  = table_128[0][15]/table_128[0][0]   # speed-up
ParallalEfficiency[1] /= table_128[1][15]                   # number of threads

ParallalEfficiency[2]  = table_256[0][15]/table_256[0][0]   # speed-up
ParallalEfficiency[2] /= table_256[1][15]                   # number of threads

ParallalEfficiency[3]  = table_512[0][15]/table_512[0][0]   # speed-up
ParallalEfficiency[3] /= table_512[1][15]                   # number of threads

ParallalEfficiency[4]  = table_1024[0][15]/table_1024[0][0] # speed-up
ParallalEfficiency[4] /= table_1024[1][15]                  # number of threads

ParallalEfficiency[5]  = table_2048[0][15]/table_2048[0][0] # speed-up
ParallalEfficiency[5] /= table_2048[1][15]                  # number of threads

NumGrid = [64, 128, 256, 512, 1024, 2048]

ax[1].plot( NumGrid, ParallalEfficiency, 'r-o', mec='r', lw=2, ms=8, label='number of threads=16' )



# set axis
# ============================================================
ax[0].set_yscale( 'log' )
ax[0].set_xscale( 'log' )
ax[0].set_xlim( 1, 16 )
ax[0].set_xlabel( 'number of threads ($N$)',  fontsize='large' )
ax[0].set_ylabel( 'cells/sec',     fontsize='large' )
ax[0].tick_params( which='both', tick2On=True, direction='in' )

ax[1].set_xscale( 'log' )
ax[1].set_xlabel( 'number of grids along one side ($N$)', fontsize='large' )
ax[1].set_ylabel( 'parallel efficiency',       fontsize='large' )
ax[1].tick_params( which='both', tick2On=True, direction='in' )



# add legend
# ============================================================
ax[0].legend( loc='upper left', numpoints=1, labelspacing=0.1, handletextpad=0.4,
              borderpad=0.4, handlelength=2.7, fontsize='large' )

ax[1].legend( loc='upper left', numpoints=1, labelspacing=0.1, handletextpad=0.4,
              borderpad=0.4, handlelength=2.7, fontsize='large' )


# save/show figure
# ============================================================
#plt.savefig( FileOut+".png", bbox_inches='tight', pad_inches=0.05 )
#plt.savefig( FileOut+".pdf", bbox_inches='tight', pad_inches=0.05 )
plt.show()
