"""看起来像库函数的未知的都被我放在这里了"""
import numpy as np
import math
import time
from scipy import integrate
from scipy import special
from scipy import interpolate
import copy
import numpy
def formatreturn(arg, input_is_array=False):
    """If the given argument is a numpy array with shape (1,), just returns
    that value."""
    if not input_is_array and arg.size == 1:
        arg = arg.item()
    return arg

def ensurearray(*args):
    """Apply numpy's broadcast rules to the given arguments.

    This will ensure that all of the arguments are numpy arrays and that they
    all have the same shape. See ``numpy.broadcast_arrays`` for more details.

    It also returns a boolean indicating whether any of the inputs were
    originally arrays.

    Parameters
    ----------
    *args :
        The arguments to check.

    Returns
    -------
    list :
        A list with length ``N+1`` where ``N`` is the number of given
        arguments. The first N values are the input arguments as ``ndarrays``s.
        The last value is a boolean indicating whether any of the
        inputs was an array.
    """
    input_is_array = any(isinstance(arg, numpy.ndarray) for arg in args)
    args = numpy.broadcast_arrays(*args)
    args.append(input_is_array)
    return args

def primary_mass(mass1, mass2):
    """Returns the larger of mass1 and mass2 (p = primary)."""
    mass1, mass2, input_is_array = ensurearray(mass1, mass2)
    mp = copy.copy(mass1)
    mask = mass1 < mass2
    mp[mask] = mass2[mask]
    return formatreturn(mp, input_is_array)



def secondary_mass(mass1, mass2):
    """Returns the smaller of mass1 and mass2 (s = secondary)."""
    mass1, mass2, input_is_array = ensurearray(mass1, mass2)
    if mass1.shape != mass2.shape:
        raise ValueError("mass1 and mass2 must have same shape")
    ms = copy.copy(mass2)
    mask = mass1 < mass2
    ms[mask] = mass1[mask]
    return formatreturn(ms, input_is_array)


def mass1_from_mchirp_q(mchirp, q):
    """Returns the primary mass from the given chirp mass and mass ratio."""
    mass1 = (q**(2./5.))*((1.0 + q)**(1./5.))*mchirp
    return mass1

def eta_from_mass1_mass2(mass1, mass2):
    """Returns the symmetric mass ratio from mass1 and mass2."""
    return mass1*mass2 / (mass1+mass2)**2.

def mchirp_from_mass1_mass2(mass1, mass2):
    """Returns the chirp mass from mass1 and mass2."""
    return eta_from_mass1_mass2(mass1, mass2)**(3./5) * (mass1+mass2)

def q_from_mass1_mass2(mass1, mass2):
    """Returns the mass ratio m1/m2, where m1 >= m2."""
    return primary_mass(mass1, mass2) / secondary_mass(mass1, mass2)

def mass1_from_mchirp_q(mchirp, q):
    """Returns the primary mass from the given chirp mass and mass ratio."""
    mass1 = (q**(2./5.))*((1.0 + q)**(1./5.))*mchirp
    return mass1

def wp_sgrb_rate(z):
    # %z = logspace(log10(0.002),log10(5),100)
    rate1 = []
    for i in range(0, len(z)):
        if z[i] < 0.9:
            rate1.append(math.exp((z[i] - 0.9) / 0.39))
        else:
            rate1.append(math.exp(-(z[i] - 0.9) / 0.26))

    # % for i=1:length(z)
    # % sfr(i) = rate1(i). / rate1(1);
    # % end

    # % figure
    # % semilogy(z, sfr. / sfr(1), 'b');
    # % set(gca, 'XScale', 'Lin');
    # % set(gca, 'YScale', 'Log');
    # % set(gca, 'Ylim', [0.01 5]);

    return rate1

def MadauDickinson2014(z):
    """

    % equation 15 - https://arxiv.org/abs/1403.0007
%-----------------------------------------------
% z=0.002:0.2:10;
% s = MadauDickinson2014(z);
% figure
% plot(z,s./s(0),'b');
%-----------------------------------------------
    :param z:
    :return:sfr0
    """
    a = lambda z: 0.005 * ((1 + z) ** 2.7)
    b = lambda z: 1 + ((1 + z) / 2.9) ** 5.6
    sfr = lambda z: a(z) / b(z)
    sfr0 = sfr(z)
    return sfr0

def SFR_Vangioni(z):
    # 似乎是文件缺少了
    return 0

def wanderman_sfr(z):
    """
    参数 z 是一个一维向量（list）
    返回值 rate 是一个一维向量（list）
    % z1 = 3;
    % for i=1:length(z)
    % if (z(i) < z1)
        % rate(i) = (1 + z(i)). ^ (2.1);
    % else
    % rate(i) = (1 + z1). ^ (2.1 - -1.4). * ((1 + z(i)). ^ (-1.4));
    % end
    % end

    % ------------------------------------
    % Wanderman values
    """
    z1 = 3
    n1 = 2.1
    n2 = -1.4
    # %------------------------------------
    # % % Lien values best
    z1 = 3.6
    n1 = 2.07
    n2 = -0.70

    # % z1=3.6;n1=2.07;n2=-1.4;
    # % upper
    # z1=3.6;n1=2.1;n2=-3.5;
    # % lower
    # z1=3.6;n1=`1.95;n2=-0.0;
    # %------------------------------------
    rate = []
    for i in range(0, len(z)):
        if z[i] < z1:
            rate.append((1 + z(i)) ** n1)
        else:
            rate.append((1 + z1) ** (n1 - n2) * ((1 + z(i)) ** n2))

    return rate

def Gpcyr2Mpcsec(rate_Gpc_yr):
    """
    % convert into Mpc

%rate_Gpc_yr = 1.5  ;
%rate_Mpc_sec = rate_Gpc_yr./(10^(9).*3600*24*365);


% Gpcyr2Mpcsec(1.5)
% Gpcyr2Mpcsec(50)

%rate_Gpc_yr = rate_Mpc_sec.*(10^(9).*3600*24*365);


%   Mpcsec2Gpcyr(9.5*10.^(-15))%
% Mpcsec2Gpcyr(5*10.^(-13))
    """
    return rate_Gpc_yr / (10 ** 9 * 3600 * 24 * 365)


def remove_elements(ind, vect):
    new_vect = []
    vv = np.isnan(ind)
    for i in range(0, len(ind)):
        if not vv[i]:
            new_vect.append(vect[i])
            
    return new_vect


def cosmology_2015(rate_Gpc_yr, z_range, sfmodel, met_val, delay, del_pow, display, inc):
    """
    参数 rate_Gpc_yr, z_range, sfmodel, met_val, delay, display, inc
    返回值 z ez0 dRdz dR U_RATE PDF CUM_PDF
    %---------------------------------------------------
% Input
%---------------------------------------------------
% rate /Gpc3/yr
%---------------------------------------------------
% Output
%---------------------------------------------------
% dRdz dR - /Mpc3/sec
% U_RATE - /sec

% To display parameters make display=1
%---------------------------------------------------------------------
% See also HOGG.m, cosmological_volume.m, z2LBT.m, LumD_mpc
%---------------------------------------------------------------------
    """

    # %---------------------------------------------------------------------
    # % Cosmology
    # %---------------------------------------------------------------------
    # % h= 0.673  ;
    # % Om_m = 0.315;
    # % Om_v = 0.685;


    h = 0.679
    Om_m = 0.306
    Om_v = 0.694

    if display:
        print('----- Cosmological Parameters -----')
        print(' h=', h)
        print(' Om_m=', Om_m)
        print(' Om_v=', Om_v)
        print('-----------------------------------')

        print('----- Source Population Parameters -----')
        print(' SFR=', sfmodel)
        print(' Metalicity Cutoff=', met_val)
        print(' Min delay time (Myr)=', delay)
        print(' Rate (Gpc-3/yr-1)=', rate_Gpc_yr)
        print('-----------------------------------')

    z = np.linspace(inc, z_range, int(z_range / inc))
    rate_Mpc_sec = Gpcyr2Mpcsec(rate_Gpc_yr)

    H0 = h * 100000

    # % Constants
    # % H = 71000 / (10 ^ (6) * 3.0856 * 10 ^ (16)); % in m

    H = H0 / (10 ** 6 * 3.0856 * 10 ** 16)  # % in m
    # % C = 3 * 10 ^ (8);
    dmpc = 10 ** 6 * 3.0856 * 10 ** 16  # % Mpc in m
    cm2mpc = 100 * 10 ** 6 * 3.0856 * 10 ** 16  # % Mpc in cm
    # % Om_m = 0.2;
    # % Om_v = 0.8
    C = 2.99792458 * 10 ** 8  # % was in cm 10 ^ (10)
    yr = 3.156 * 10 ** 7
    Myr = yr * 10 ** 6
    Gyr = yr * 10 ** 9

    # % ---------------------------------------------------------------------
    # % Cosmology Functions
    # % ---------------------------------------------------------------------
    DH = C / H  # % Hubble distance in m
    Ez = lambda z: (Om_m * (1 + z) ** 3 + Om_v) ** 0.5
    Ez0 = lambda z: (Om_m * (1 + z) ** 3 + Om_v * (1 + z) ** 2) ** 0.5
    Ez1 = lambda z: 1 / Ez(z)
    Ez2 = lambda z: 1 / ((1 + z) * Ez(z))
    Ez3 = lambda z: integrate.quad(Ez1, 0, z)[0]

    # % luminosity distance
    dL = lambda z: DH * (1 + z) * Ez3(z)  # % dL in meters
    dM = lambda z: dL(z) / (1 + z)

    # % Volume element - alternative version of dimensionless
    phir = lambda z: Ez3(z)
    phiV = lambda z: phir(z) ** 2 * Ez1(z)

    # % volume element in Mpc ^3
    dvdz_mpc = lambda z: 4 * np.pi * DH * dL(z) ** 2 * Ez1(z) * (1 + z) ** (-2) / dmpc ** 3

    # % in Mpc^3 per steridian
    # % dvdz_mpc = @(z)1*(C/H).*dL(z).^(2)*Ez1(z)*(1 + z)^(-2)./(dmpc)^(3);
    beaming = lambda ang: (1 - math.cos(ang * np.pi / 180)) ** (-1)

    K = 1 / H / Gyr  # % Hubble time in Gyr
    LBT = lambda z: K * integrate.quad(Ez2, 0, z)[0]  # % look back time in Gyr

    # % ---------------------------------------------------------------------
    # % Metalicity
    # % ---------------------------------------------------------------------
    # % met_val = 0.3
    # % met_val = [];

    if not met_val:
        met_val = -1.0
        met=1
    else:
        alp = -1.16
        bet = 2
        eps1 = met_val
        A = alp + 2
        X = lambda z: eps1 ** bet * 10 ** (0.15 * bet * z)
        met = lambda z: special.gammainc(X(z), A)

        # met = gam_inc;
        # met = gam_inc./gam; % old function - scipy already includes '/gam'
        # function
        # figure;plot(z,met(z));
        # hold;
        # plot(z,met1,'--m');

    # % ---------------------------------------------------------------------
    # % Delay
    # % ---------------------------------------------------------------------
    # % if isempty(delay)
    #    % delay = -1.0;
    # % end
    # % ---------------------------------------------------------------------

    # % ---------------------------------------------------------------------
    # % SFR
    # MODELS
    # % ---------------------------------------------------------------------
    if sfmodel == 'sf2':
        if met_val > 0:
            sfr = lambda z: met(z) * (Ez(z) / Ez0(z) * 0.15 * (math.exp(3.4 * z) / (np.exp(3.4 * z) + 22)))
        else:
            sfr = lambda z: Ez(z) / Ez0(z) * 0.15 * (np.exp(3.4 * z) / (math.exp(3.4 * z) + 22))
        sfr0 = lambda z: sfr(z) / sfr(0)

    if sfmodel == 'constant':
        sfr0 = lambda z: 1
        sfr = lambda z: 1

    if sfmodel == 'hb':
        if met_val > 0:
            sfr = lambda z: met(z) * (h * (0.017 + 0.13 * z) / (1 + (z / 3.3) ** 5.3))
        else:
            sfr = lambda z: (h * (0.017 + 0.13 * z) / (1 + (z / 3.3) ** 5.3))
        sfr0 = lambda z: sfr(z) / sfr(0)

    if sfmodel == 'hbs':
        if met_val > 0:
            sfr = lambda z: met(z) * (h * (0.017 + 0.13 * z) / (1 + (z / 3.3) ** 4.3))
        else:
            sfr = lambda z: h * (0.017 + 0.13 * z) / (1 + (z / 3.3) ** 4.3)
        sfr0 = lambda z: sfr(z) / sfr(0)

    if sfmodel == 'li':
        a = 0.0157
        b = 0.118
        c = 3.23
        d = 4.66
        if met_val > 0:
            sfr = lambda z: met(z) * ((a + (b * z)) / (1 + (z / c) ** d))
        else:
            sfr = lambda z: (a + (b * z)) / (1 + (z / c) ** d)
        sfr0 = lambda z: sfr(z) / sfr(0)

    if sfmodel == 'wanderman':
        sfr = lambda z: wanderman_sfr(z)
        sfr0 = lambda z: wanderman_sfr(z)

    if sfmodel == 'kistler':
        a = 3.4
        b = -0.3
        c = -2.5
        z1 = 1
        z2 = 4
        B = (1 + z1) ** (1 - a / b)
        C = ((1 + z1) ** ((b - a) / c)) * ((1 + z2) ** (1 - b / c))
        zeta = -10
        # %B=5160;C=11.5
        if met_val > 0:
            sfr = lambda z: met(z) * (
                    (1 + z) ** (a * zeta) + ((1 + z) / B) ** (b * zeta) + ((1 + z) / C) ** (c * zeta)) ** (1 / zeta)
        else:
            sfr = lambda z: ((1 + z) ^ (a * zeta) + ((1 + z) / B) ** (b * zeta) + ((1 + z) / C) ** (c * zeta)) ** (
                    1 / zeta)
        sfr0 = lambda z: sfr(z) / sfr(0.0)

    if sfmodel == 'kistler08':
        eta = -10
        a2 = 3.4
        b2 = -0.3
        c2 = -3.5
        BB = 5000
        CC = 9
        if met_val > 0:
            sfr = lambda z: met(z) * (
                    ((1 + z) ** (a2 * eta)) + (((1 + z) / BB) ** (eta * b2)) + (((1 + z) / CC) ** (eta * c2))) ** (
                                    1 / eta)
        else:
            sfr = lambda z: (((1 + z) ** (a2 * eta)) + (((1 + z) / BB) ** (eta * b2)) + (
                    ((1 + z) / CC) ** (eta * c2))) ** (1 / eta)
        sfr0 = lambda z: sfr(z) / sfr(0.0)

    if sfmodel == 'kistler13':
        eta = -10
        a2 = 3.4
        b2 = -0.3
        c2 = -2.5
        BB = 5160
        CC = 11.5
        if met_val > 0:
            sfr = lambda z: met(z) * (
                    ((1 + z) ** (a2 * eta)) + (((1 + z) / BB) ** (eta * b2)) + (((1 + z) / CC) ** (eta * c2))) ** (
                                    1 / eta)
        else:
            sfr = lambda z: (((1 + z) ** (a2 * eta)) + (((1 + z) / BB) ** (eta * b2)) + (
                    ((1 + z) / CC) ** (eta * c2))) ** (1 / eta)
        sfr0 = lambda z: sfr(z) / sfr(0.0)

    if sfmodel == 'yuksel08':
        eta = -10
        a2 = 3.4
        b2 = -0.3
        c2 = -3.5
        BB = 5000
        CC = 9
        if met_val > 0:
            sfr = lambda z: met(z) * (
                    ((1 + z) ** (a2 * eta)) + (((1 + z) / BB) ** (eta * b2)) + (((1 + z) / CC) ** (eta * c2))) ** (
                                    1 / eta)
        else:
            sfr = lambda z: (((1 + z) ** (a2 * eta)) + (((1 + z) / BB) ** (eta * b2)) + (
                    ((1 + z) / CC) ** (eta * c2))) ** (1 / eta)
        sfr0 = lambda z: sfr(z) / sfr(0.0)

    if sfmodel == 'md':
        if met_val > 0: 
            sfr = lambda z: MadauDickinson2014(z)
        else:
            sfr = lambda z: MadauDickinson2014(z)
        sfr0 = lambda z: sfr(z) / sfr(0.0)

    if sfmodel == 'wp2015':
        sfr = lambda z: wp_sgrb_rate(z)
        sfr0 = lambda z: wp_sgrb_rate(z)

    # %--------------------------------------------------------------------------

    # % sfr = @(z)  (h*(0.017+0.13*z)./(1+(z./3.3).^5.3))   ; % HB SFR
    # %   lll=length(z);
    # %--------------------------------------------------------------------------
    # % Calculate dR/dz with delay time
    # %--------------------------------------------------------------------------

    if not delay:
        # % disp(['empty'])
        # %--------------------------------------------------------------------------
        # %  dR/dz with no delay time
        # %--------------------------------------------------------------------------
        ez = []
        for i in range(0, len(z)):
            ez.append(sfr(z[i]))
        ez0 = []
        for i in range(0, len(z)):
            ez0.append(ez[i] / ez[0])
        # %     figure
        # %     plot(z,ez,'r');
        dRdz = []
        for i in range(0, len(z)):
            #print(met(z[i]), ez0[i], dvdz_mpc(z[i]), z[i])
            dRdz.append(rate_Mpc_sec * ez0[i] * dvdz_mpc(z[i]) / (1 + z[i]))
    else:
        # %   disp(['not empty'])
        tau = delay * 10 ** (-3)  # 100 Myr to Gyr
        # % convert to look back time

        # z3=0.001:0.001:z_range*3;

        delay_power = del_pow
        t = []
        for i in range(0, len(z)):
            t.append(LBT(z[i]))
        merger_rate = []
        for i in range(0, len(t)):
            if delay < 20:
                td = np.linspace(tau, LBT(max(z)) - t[i], 2000)
            elif delay < 30:
                td = np.linspace(tau, LBT(max(z)) - t[i], 1000)
            elif delay < 100:
                td = np.linspace(tau, LBT(max(z)) - t[i], 500)
            else:
                td = np.linspace(tau, LBT(max(z)) - t[i], 200)
            tf = [j + t[i] for j in td]
            if tf[0] < max(t):
                zf_fun = interpolate.interp1d(t, z)
                zf = []
                for j in range(0, len(tf)):
                    if tf[j] > max(t):
                        tf[j] = max(t)
                    zf.append(zf_fun(tf[j]))
                sfr_shifted = []
                for j in range(0, len(zf)):
                    sfr_shifted.append(sfr(zf[j]) / (1 + zf[j]))
                del_sfr = []
                for j in range(0, len(sfr_shifted)):
                    del_sfr.append(sfr_shifted[j] * (td[j] ** delay_power))
                merger_rate.append(np.trapz(del_sfr, td))
            else:
                merger_rate.append(np.nan)
            # % this is the NaN problem
            # % merger_rate(i) = trapz(td,sfr_shifted.*(1./td));
        ez0_merger = [i / merger_rate[0] for i in merger_rate]
        # %---------------------------------------------------
        # % extra code to correct for NaN problem

        ez0_merger2 = remove_elements(ez0_merger, ez0_merger)
        z2 = remove_elements(ez0_merger, z)
        ez0_merger_fun = interpolate.interp1d(z2, ez0_merger2)
        z = z2
        ez0_merger = [ez0_merger_fun(i) for i in z]
        ez0 = ez0_merger
        # %---------------------------------------------------

        dRdz = []
        for i in range(0, len(z)):
            dRdz.append(rate_Mpc_sec * ez0_merger[i] * dvdz_mpc(z[i]) / (1 + z[i]))

    # % figure
    # % plot(z,dRdz,'--k');
    # %--------------------------------------------------------------------------
    dR = integrate.cumtrapz(dRdz, z)
    U_RATE = max(dR)
    PDF = [i / U_RATE for i in dRdz]
    CUM_PDF = integrate.cumtrapz(PDF, z)

    # %--------------------------------------------------------------------------
    # % Extra code to remove NaNs resulting from delay time distribution
    # %--------------------------------------------------------------------------
    """
    %---------------------------------------------------------------------
% Examples
%---------------------------------------------------------------------


%---------------------------------------------------------------------
% % Compare with previous function
%---------------------------------------------------------------------
% tic
% [z3a ez03a DR3a dRdZ_final3a U_RATE3aa PDF3 CUM_PDF3a] =cosmology_metal_delay(BBH_rate_Gpc3yr,10,'sf2',0.3, []);
% toc
% 
% tic
% [z ez0 dRdz dR U_RATE PDF CUM_PDF] =cosmology_2015(BBH_rate_Gpc3yr,10,'sf2',0.3, []);
% toc
% 
% figure
% plot(z3a,dRdZ_final3a,'--r');hold
% plot(z, dRdz,'--k');

%---------------------------------------------------------------------
% % Compare with SFR no evolution
%---------------------------------------------------------------------
% BBH_rate_Gpc3yr=70
% Gpcyr2Mpcsec(BBH_rate_Gpc3yr)
% 
% [zv dV_rate_MpcSec dVdz_rate_MpcSec lum_d_Mpc lum_d_m  dVdz_m dVdz_Mpc dV_Mpc lum_d_hogg dVdz_hogg] = cosmological_volume( 0.673,0.315 ,0.685 ,10, Gpcyr2Mpcsec(BBH_rate_Gpc3yr), 'low' )
% 
% [z ez0 dRdz dR U_RATE PDF CUM_PDF] =cosmology_2015(BBH_rate_Gpc3yr,10,'constant',[], []);
% 
% 
% figure
% plot(zv,dVdz_rate_MpcSec,'--r');hold
% plot(z, dRdz,'--k');


%------------------------------------------------------------------------

% tic
% rate_Gpc3yr = 10
% [z ez0 dRdz dR U_RATE PDF CUM_PDF] =cosmology_2015(rate_Gpc3yr,10,'sf2',0.1, []);
% toc
% U_rate_day = U_RATE.*3600*24
% U_rate_year_LIGO = interp1(z,dR,Mpc2z(200)).*3600*24*365
% 
% figure
% plot(z3a,dRdZ_final3a,'--r');hold
% plot(z, dRdz,'--k');

%------------------------------------------------------------------------
    """
    return z, ez0, dRdz, dR, U_RATE, PDF, CUM_PDF
