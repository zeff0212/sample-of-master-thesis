#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 17:33:57 2022

@author: flj
"""

import math
import numpy as np
from scipy import integrate
from scipy import special
import scipy.optimize as opt
import emcee
import matplotlib.pyplot as plt
import corner
import Kit
import matplotlib as mpl
from time import time  
from numpy import random
from scipy import interpolate
from scipy.stats import gaussian_kde
import os
import pickle as pkl
import gwpy
from gwpy.astro import sensemon_range
from scipy import stats as st
from multiprocessing import Pool
from multiprocessing import cpu_count
from functools import partial
import csv
os.environ["OMP_NUM_THREADS"] = "1"

def mpl_inline():
  rc = plt.rcParams.copy()
  #matplotlib inline
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




# rate_true = 1210
# met_true = 0.3
# delay_true = 10
# [zB, ez0_B, dRdz_B, dR_b, U_RATE_B, PDF_B, CUM_PDF_B] = cosmology_2015(rate_Gpc_yr = 1210, z_range = 10, sfmodel = 'md', met_val = 0.3, delay = 10, inc = 0.02, display = 1)
# plt.plot(zB, PDF_B, 'k')
# plt.show()
# xdata = zB
# ydata = PDF_B
# sigma = np.sqrt(6)

###############################################################################
# TIME THE FUNCTION
###############################################################################
from astropy.cosmology import Planck15 as cosmo

def z2Mpc(z):
    return cosmo.luminosity_distance(z).value   

with open('snr_from_dist_mass_for_CE.pickle', "rb") as f:
    snr_data = pkl.load(f) # latin1 option only needed as .pkl prodiced using python version 2.7 on CIT
dist_vector = snr_data['dist_vector']
mass_vector = snr_data['mass_vector']                  
snr_array = snr_data['snr_array']

chirp_mass_mean = 10.5          
chirp_mass_sd = 1.3
mc_low = 1.2
mc_up = 63

# Chirp mass distribution
# plt.figure(figsize=(8,6))
# plt.plot(mc_vect,mc_dist, 'k');
def mass_frac(mass, mc_mean, mc_std):
    return st.norm.cdf(mass, loc=mc_mean, scale=mc_std)
#  Normalisation factor
norm_val = mass_frac(mc_up, chirp_mass_mean, chirp_mass_sd)-mass_frac(mc_low, chirp_mass_mean, chirp_mass_sd)
con = 1/norm_val 

def snr_from_distance_grid_CE(input_mass_dist):    
    distance_mass_tuple = [mass_vector, dist_vector]
    snr_value = interpolate.interpn(distance_mass_tuple, snr_array, input_mass_dist, bounds_error=False, fill_value=0, method='linear')[0]    
    return snr_value

def chirp_mass_from_distance_and_snr(snr_low, snr_up, distance):
    snr_input = [snr_low, snr_up]
    chirp_mass_space= np.logspace(math.log10(1.2), math.log10(63), 100)
    mass1_array = [Kit.mass1_from_mchirp_q(mchirp, 1) for mchirp in chirp_mass_space]
    mass1_dist_array = [[mass1, distance] for mass1 in mass1_array]
    mass1_dist_array = np.array(mass1_dist_array)
    snr_test = []
    for i in range(len(mass1_dist_array)):
        snr_test.append(snr_from_distance_grid_CE(mass1_dist_array[i]))
    snr_test = np.array(snr_test)
    snr_max = np.where(snr_test == max(snr_test))
    chirp_low = np.interp(snr_input, snr_test[:snr_max[0][0]+1], chirp_mass_space[:snr_max[0][0]+1])
    chirp_up = np.interp(snr_input, snr_test[len(snr_test):snr_max[0][0]-1:-1], chirp_mass_space[len(snr_test):snr_max[0][0]-1:-1])
    return chirp_low, chirp_up

def CE_differential_snr_2021(snr_range, zB, dRdz_B, mc_mean, mc_std):
    #-------------------------------------------------------------------------------------
    # Determine the normalisation factor for the LF based on alp and bet input parameters    
    #-------------------------------------------------------------------------------------
    snr1, snr2 = snr_range
    
    z_vect = zB
    #snr_spacing = np.linspace(1,50, 10)
    
    dN_dSNR_year = []
    # print('#######################################')
    for j in range(0,len(z_vect)):
        # print('redshift',zB[j] )
        try:
            mc_from_dis_and_snr = chirp_mass_from_distance_and_snr(snr1, snr2, distance=z2Mpc(z_vect[j]))
            chirp_mass_1 = mc_from_dis_and_snr[0][0]/(1+z_vect[j])
            chirp_mass_2 = mc_from_dis_and_snr[0][1]/(1+z_vect[j])
            chirp_mass_3 = mc_from_dis_and_snr[1][1]/(1+z_vect[j])
            chirp_mass_4 = mc_from_dis_and_snr[1][0]/(1+z_vect[j])
            # print('chirp mass 1 ',chirp_mass_1 )
            # print('chirp mass 2 ',chirp_mass_2 )
            # print('chirp mass 3 ',chirp_mass_3 )   
            # print('chirp mass 4 ',chirp_mass_4 )   
            # mc_vect_new = np.logspace(math.log10(chirp_mass_1),math.log10(chirp_mass_2),40)
            # mc_dist2 = st.norm.pdf( mc_vect_new, loc=chirp_mass_mean, scale=chirp_mass_sd) 
            fraction_accessible_chirp_mass_dist = con * (mass_frac(chirp_mass_2, mc_mean, mc_std)-mass_frac(chirp_mass_1, mc_mean, mc_std)
                                                         +mass_frac(chirp_mass_4, mc_mean, mc_std)-mass_frac(chirp_mass_3, mc_mean, mc_std))  
        except:
            fraction_accessible_chirp_mass_dist = 0
            # print('exception')
        # print('Mass frac',fraction_accessible_chirp_mass_dist)
        # print('dR',dRdz_B[j]*(3600*365*24))
        # print('total',fraction_accessible_chirp_mass_dist*dRdz_B[j]*(3600*365*24))
        # print('===================================')
        dN_dSNR_year.append(fraction_accessible_chirp_mass_dist*dRdz_B[j]*(3600*365*24))
    # print('#######################################')
    total_snrs=np.trapz(dN_dSNR_year,z_vect)
    # print('total_snrs= ',total_snrs)       
    return total_snrs

with open('SNR_data_for_CE_with_30_rate', mode='r') as f:
    reader = csv.reader(f)
    for row in reader:
        snr_data_csv = row

snr_data_C = []
for i in snr_data_csv:
    snr_data_C.append(float(i))

CE_xdata_range = np.logspace(math.log10(1),math.log10(600),11)
plt.figure(figsize=(8, 6))
n2, bins, patches = plt.hist(snr_data_C, bins=CE_xdata_range, color='#0504aa', alpha=0.7, rwidth=0.85, density=False)
plt.grid(axis='y', alpha=0.75)
plt.xscale('log')
plt.xlabel('SNR')
plt.ylabel('Number')
plt.title('SNR Histogram (Cosmic Explorer)')

xdata_C = np.log10(CE_xdata_range)[:len(CE_xdata_range)-1]+np.diff(np.log10(CE_xdata_range))/2
sigma_C = []
for i in range(0, len(n2)):
    sigma_C.append(math.sqrt(n2[i]))
sigma_C = np.array(sigma_C)
 
xdata = np.logspace(math.log10(1),math.log10(600),21)[1::2]
ydata = n2
ydata = np.array(ydata)
sigma = sigma_C
for i in range(len(sigma)):
    if sigma[i] == 0.0:
        sigma[i] = 1
  
def log_prior(theta):
    rate, chirpmass, mass_std = theta
    if rate > 0.0 and chirpmass > 0.0 and mass_std > 0.5:
        return 0.0
    return -np.inf

def mode1(x, theta):
    rate, chirpmass, mass_std = theta
    [z, ez0, dRdz, dR, U_RATE, PDF, CUM_PDF] = Kit.cosmology_2015(rate_Gpc_yr=rate, z_range=30, sfmodel='md', met_val=0, delay=[], del_pow=-1.0, inc=0.3, display=[])
    x = np.array(x)
    x_axis = np.logspace(math.log10(1), math.log10(600), len(x)+1)
    delta_num = []
    for i in range(len(x_axis)-1):
        delta_num.append(CE_differential_snr_2021(snr_range=[x_axis[i], x_axis[i+1]], zB=z, dRdz_B=dRdz, mc_mean=chirpmass, mc_std=mass_std))
    return delta_num

def log_likelihood(theta, x, y, sigma):
    rate, chirpmass, mass_std = theta
    model = mode1(x, [rate, chirpmass, mass_std])
    residual = y - model
    sigma = sigma
    like = np.sum(- (residual / sigma) ** 2 / 2 - np.log(2 * np.pi * sigma ** 2) / 2)
    return like

def log_prob(theta, x, y, sigma):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, sigma)

###############################################################################
# MCMC parameters
###############################################################################

thinning = 1 #500
num_iter = 10000
nwalkers = 256 # 256 #256

# thinning = 128 #500

###############################################################################
# Set up output folder
###############################################################################

rate_true = 30
mass_true = chirp_mass_mean
mass_std_true = chirp_mass_sd
initial_mcmc_values = [rate_true, mass_true, mass_std_true]
ndim = len(initial_mcmc_values)

data_mcmc_chain_pkl = 'CE_LogN_LogSNR_mcmc_' + format(num_iter) +'_thinning_'+ format(thinning) + '_nwalkers_' + format(nwalkers) + '.pkl'
print(data_mcmc_chain_pkl)

plots_filename = 'CE_LogN_LogSNR_mcmc_' + format(num_iter) +'_thinning_'+ format(thinning) + '_nwalkers_' + format(nwalkers)
info_text_file = 'CE_LogN_LogSNR_mcmc_' + format(num_iter) +'_thinning_'+ format(thinning) + '_nwalkers_' + format(nwalkers) + '.txt'
h5_filename = 'CE_LogN_LogSNR_mcmc_' + format(num_iter) +'_thinning_'+ format(thinning) + '_nwalkers_' + format(nwalkers) +'.h5'
print(h5_filename)

folder = 'CE_LogN_LogSNR_mcmc_' + format(num_iter) +'_thinning_'+ format(thinning) + '_nwalkers_' + format(nwalkers) +''
os.mkdir(folder)

path = os.path.join(os.getcwd(), folder)

backend = emcee.backends.HDFBackend(os.path.join(os.getcwd(), folder, h5_filename))
backend.reset(nwalkers, ndim)
###############################################################################
# MCMC PART
###############################################################################

total_samples = nwalkers * (num_iter / thinning)
print('Total samples is: ', total_samples)

pos = [[rate_true, mass_true, mass_std_true]]*nwalkers + 0.01 * np.random.randn(nwalkers, ndim)
print('********************************')
print('init positions: ')
print(pos)
print('********************************')

with Pool(cpu_count()) as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(xdata, ydata, sigma), pool=pool, backend=backend)
    start_time = time()
    sampler.run_mcmc(pos, num_iter, thin=thinning, progress=True, store=True)
    end_time = time()
    multi_time = end_time - start_time

# print("Multiprocessing took {0:.1f} seconds".format(multi_time))

###############################################################################
# Output results to text file
###############################################################################

print(info_text_file)


labels = ["rate", "chirp_mass_mean", "chirp_mass_std"]

info_text_file = os.path.join(path, info_text_file)
f = open(info_text_file, "a")

f.write('#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
f.write('# SUMMARY\n')
f.write('#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
f.write('\nTime taken to run emcee with emcee ' + format(emcee.__version__) + ' is:\n' + format(multi_time) + ' seconds\n' + format(multi_time/60) + ' mins\n'+format(multi_time/3600) + ' hours'+'\n' 'num_iter = ' + format(num_iter) +'\nthinning= ' + format(thinning) + '\nnwalkers= ' + format(nwalkers) + '\n')
f.write('Number of cores: ' + format(cpu_count()))
f.write('\nacceptance fraction =' + format(sampler.acceptance_fraction))
f.write('\nMean acceptance fraction =' + format(np.mean(sampler.acceptance_fraction)) + '\n\n')
#f.write(prior_inf)
f.write('\n#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
f.write('# RESULTS\n')
f.write('#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n')
flatchain = sampler.get_chain(discard=1, thin=1, flat=True)
for i in range(ndim):
    mcmc = np.percentile(flatchain[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    f.write(txt)
    f.write('\n')
f.write('\n#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
f.write('# DATA\n')
f.write('#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n')
f.write(h5_filename + '\n')
f.write(data_mcmc_chain_pkl + '\n')
f.write(plots_filename+'_corner.png' + '\n')
f.write(plots_filename+'_chain.png' + '\n')
f.close()

###############################################################################
# ACCEPTANCE FRACTION
###############################################################################
frac = np.mean(sampler.acceptance_fraction)
if (frac < 0.2) or (frac > 0.5):
#        if (frac < 0.2) or (frac > 0.5):
    print('Undesirable acceptance fraction(= {}).'.format(frac)\
          + ' Please run the MCMC code again and increase num_iter and/or thinning.')
else:
    print('Good acceptance fraction(= {}).'.format(frac) + '')

###############################################################################
# Save data to pkl
###############################################################################

samples = sampler.get_chain()
b = {'Acceptance': frac, 'mcmc_chain': flatchain, 'samples_unflat': samples}
with open(os.path.join(os.getcwd(), folder, data_mcmc_chain_pkl), 'wb') as handle:
    pkl.dump(b, handle, protocol=pkl.HIGHEST_PROTOCOL)

###############################################################################
# PLOT CHAINS
###############################################################################

fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")

###############################################################################
# Corner plot
###############################################################################

print(flatchain.shape)
fig = corner.corner(flatchain, labels=labels, truths=[rate_true, mass_true, mass_std_true])
name = os.path.join(path, '.'.join((plots_filename+'_corner_plot', 'png')))
plt.show()
plt.savefig(name, facecolor='white')
plt.close()

tau = sampler.get_autocorr_time()
print(tau)
