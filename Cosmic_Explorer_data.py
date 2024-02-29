#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 09:31:44 2022

@author: flj
"""

import numpy as np
import gwpy
import csv
import pandas as pd
import numpy as np
import math
import sys
import pickle as pkl

#from gwpy.timeseries import TimeSeries
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
from gwpy.astro import sensemon_range
#from gwpy.astro import inspiral_range

#from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import FlatLambdaCDM, z_at_value
from astropy import units as u
cosmo = FlatLambdaCDM(H0=67.3 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.315)
from collections import OrderedDict # for python 2.7

from scipy import stats as st
from scipy import interpolate as interp
from scipy import integrate, interpolate
from scipy.stats import gaussian_kde
from scipy.integrate import tplquad,dblquad,quad, trapz, cumtrapz
from scipy import stats as st
from scipy.integrate import cumtrapz, trapz 
from scipy import special
# import scipy.optimize as opt
# import emcee
import matplotlib as mpl
import matplotlib.pyplot as plt
# import corner

from time import time  
from numpy import random 

from multiprocess import Pool
from functools import partial

import Kit




def z2Mpc(z):
    return cosmo.luminosity_distance(z).value   

def Mpc2z(dL_val):
    zvalue = z_at_value(cosmo.luminosity_distance,
           dL_val * u.Mpc, zmax=30)   
    return float(zvalue)

#lum_distance_data = z2Mpc(random_z)

def mpl_inline():
  rc = plt.rcParams.copy()
  # matplotlib inline
  plt.rcParams.update(rc)
mpl_inline()        

mpl.rcParams.update({'font.size': 20, 'font.family': 'Calibri', 'mathtext.fontset': 'stix', 'lines.linewidth':3,'lines.markersize':10,\
                     'axes.linewidth':3, 'axes.labelsize': 'large', 'axes.labelweight': 'normal',\
                     'figure.figsize': (5, 3), 'ytick.major.size' : 6, 'xtick.major.size' : 6, \
                     'ytick.direction': 'in', 'xtick.direction': 'in', 'axes.labelpad': 15.0, \
                     'xtick.bottom': True, 'xtick.top': True, 'xtick.top': True, 'xtick.major.size': 9.5, \
                     'xtick.minor.bottom': False, 'xtick.minor.size': 5.0, 'xtick.minor.visible': True, \
                     'xtick.major.width': 1.8,'xtick.minor.width': 1.8,'ytick.left': True,'ytick.right': True, \
                     'ytick.minor.left': False, 'ytick.minor.right': False,'ytick.minor.size': 3.0, \
                     'ytick.major.width': 1.8,'ytick.minor.width': 1.8, 'ytick.minor.visible': True, \
                     'ytick.major.size': 9.5, 'axes.titlepad': 12.0})
#==============================================================================
####################################################
# Grid for model
####################################################

CE_data = 'cosmic_explorer.txt'
q = np.loadtxt(CE_data,unpack=True)
f=q[0]
asd=q[1]  # amplitude spectral density
psd = asd**(2) # power spectral density
CE_psd_data = gwpy.frequencyseries.FrequencySeries(psd, unit=None, f0=None, df=None, frequencies=f, name=None, epoch=None, channel=None)

def snr_from_distance(input_dist, input_psd, mass1, mass2):           
    hor_h1 =[]
    snr_space = np.logspace(math.log10(0.01), math.log10(2000), 500)
    for s in snr_space:
        hor_h1.append(sensemon_range(input_psd, snr=s, mass1=mass1, mass2=mass2, fmin=30, horizon=True).value)
    return np.interp(input_dist, hor_h1[::-1], snr_space[::-1])

dist_vector = np.logspace(math.log10(10), math.log10(z2Mpc(30)), 200)
mass_vector = np.logspace(math.log10(1), math.log10(72.5), 100)

snr_array = []
for i in range(len(mass_vector)):
    snr_array.append(snr_from_distance(dist_vector, CE_psd_data, mass_vector[i], mass_vector[i]))
    
snr_array = np.array(snr_array)
print(snr_array.shape)
print(mass_vector.shape)
print(dist_vector.shape) 

a = {'dist_vector':dist_vector, 'mass_vector':mass_vector, 'snr_array': snr_array}
with open('snr_from_dist_mass_for_CE.pickle', 'wb') as handle:
    pkl.dump(a, handle, protocol=pkl.HIGHEST_PROTOCOL)
    
    
####################################################
# Simulate data
####################################################

months = 12
[zC, ez0_C, dRdz_C, dR_C, U_RATE_C, PDF_C, CUM_PDF_C] = Kit.cosmology_2015(rate_Gpc_yr=30, z_range=30, sfmodel='md', met_val=0, delay=[], del_pow=-1.0, inc=0.03, display=1)
total_events_C = round((months*3600*24*30)*U_RATE_C)
print('Total Events  = ', total_events_C)
random_uniform_C = []
random_uniform_C = random.uniform(0, 1, round(total_events_C)) 
random_z_C = []
REAL_CUM_C = [0]
for i in range(len(CUM_PDF_C)):
    REAL_CUM_C.append(CUM_PDF_C[i])
CUM_PDF_C = REAL_CUM_C

for i in range(len(random_uniform_C)):
    random_z_C.append(np.interp(random_uniform_C[i], CUM_PDF_C, zC, left=None, right=None, period=None))

lum_distance_data_C = z2Mpc(random_z_C)

chirp_mass_mean = 10.5          
chirp_mass_sd = 1.3
mc_low = 1.2
mc_up = 63

rand_mc_C = np.random.normal(loc=chirp_mass_mean, scale=chirp_mass_sd, size=len(random_z_C))
randz_and_mc_C = []
for i in range(len(random_z_C)):
    randz_and_mc_C.append([random_z_C[i], rand_mc_C[i]])

def iterable_snr_from_dist_C(z_mc):
    z, mc = z_mc
    m1 = Kit.mass1_from_mchirp_q(mchirp=mc, q=1)
    try:
        snr = snr_from_distance(input_dist=z2Mpc(z), input_psd=CE_psd_data, mass1=m1*(1+z), mass2=m1*(1+z))
        return snr
    except:
        pass

pool = Pool()
snr_data_C = pool.map(iterable_snr_from_dist_C, randz_and_mc_C)
pool.close()
pool.join()

for i in range(len(snr_data_C)-1, -1, -1):
    if snr_data_C[i] == None:
        del snr_data_C[i]

import csv
with open('SNR data for CE with 30 rate', 'w') as snrf:
    writer = csv.writer(snrf)
    writer.writerow(snr_data_C)
