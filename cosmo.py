"""
Assorted cosmology functions
caps => simplified functions assuming flat cosmology ie omega_k,omega_r=0 and omega_lambda=1-omega_m
lowercase => no assumptions about parameters
by Meryl
"""
import numpy as np
from scipy.integrate import quad
c=299792.458 #speed of light(km/s)



def scalefactor(z):
    """
    Scale Factor
    
    Parameters
    ====

    z: float or array-like
            redshift (dimensionless)

    Returns
    ====
    a: float or array-like
             scale factor a=1/(1+z)
    
    """
    return 1/(1+z)

def dH(H0):
    """
    Hubble Distance

    Parameters
    ====

    H0: float or array-like
            Hubble constant, with different prior depending on SHoES or Planck measurement ((km/s)/Mpc)

    Returns
    ====
    d_H: float or array-like
             Hubble distance, d_H defined as the speed of light times the Hubble time: c/H_0

    """
    return c/H0

"""
Flat lambdacdm 
===========================================================================================
"""

def EZ(z,omm):

    """
    The Hubble Function

    Parameters
    ====
            
    
    omm: float or array-like
            omega_m, total matter density (dimensionless)

            
    z: float or array-like
            redshift (dimensionless)

    Returns
    ====
    E(z): float or array-like
            a function in z
            
    """
    omla=1-omm
    
    oms= omm*(1+z)**3 + omla
    
    E=np.sqrt(oms)
    
    return E

def HZ(H0,omm,z):
    
    """
    Hubble Parameter

    Parameters
    ====

    H0: float or array-like
            Hubble constant, with different prior depending on SHoES or Planck measurement ((km/s)/Mpc)
    
    omm: float or array-like
            omega_m, total matter density (dimensionless)
            
    z: float or array-like
            redshift (dimensionless)

    Returns
    ====
    H(z): float or array-like
            The Hubble parameter H(z)=H_0*E(z)

    """

    return EZ(omr,omla,omm,omk,z)*H0

def Integrand(x,omm):
    
    """
    Function to be integrated in order to find the comoving distance

    Parameters
    ====
    
    x: float or array-like
            Dummy variable for z
                
    
    omm: float or array-like
            omega_m, total matter density (dimensionless)
            

    Returns
    ====
    f(x): float or array-like
            f(x)=1/E(x)

    """
    
    return 1/EZ(x,omm)

def DM(z,H0,omm):
    
    """
    Function to be integrated in order to find the comoving distance

    Parameters
    ====
    
    H0: float or array-like
            Hubble constant, with different prior depending on SHoES or Planck measurement ((km/s)/Mpc)

    
    omm: float or array-like
            omega_m, total matter density (dimensionless)
            

    z: float or array-like
            redshift (dimensionless)

    Returns
    ====
    D_c(z): float or array-like
            the comoving distance of a source with redshift z (Mpc)
            For flat cosmology D_c=D_m
            D_c(z)=d_H*I(z)
            where F(z) is the integral of f(x) from 0 to z

    """
       
    integral= quad(Integrand, 0, z, args=(omm)) #integrate f(x) from x=0 to z
    
    DM=dH(H0)*integral[0] #d_H*F(z)
    
    return DM

def DL(z,H0,omm):
    
    """
    Luminosity Distance

    Parameters
    ====

    H0: float or array-like
            Hubble constant, with different prior depending on SHoES or Planck measurement ((km/s)/Mpc)

    
    omm: float or array-like
            omega_m, total matter density (dimensionless)
            

    z: float or array-like
            redshift (dimensionless)

    Returns
    ====
    DL: float or array-like
            dEML, electromagnetic luminosity distance (Mpc)
            
            dEML=(1+z)*dm
            
            where dm is the transverse comoving distance
    """
    
    dm=DM(z,H0,omm)
        
    return (1+z)*dm

def VC(z,H0,omm):
    
    """
    Comoving Volume

    Parameters

    ====

    H0: float or array-like
            Hubble constant, with different prior depending on SHoES or Planck measurement ((km/s)/Mpc)


    omm: float or array-like
            omega_m, total matter density (dimensionless)
            
            
    z: float or array-like
            redshift (dimensionless)

    Returns
    ====
    vc: float or array-like
            V_c, comoving volume (Mpc**3)

    """
                                   
    dm=DM(z,H0,omm) #transverse comoving distance
    vc = (4*np.pi*dm**3)/3

        
    return vc

"""
Full (unsimplified) functions
===========================================================================================
"""

def ez(omr,omla,omm,omk,z):

    """
    The Hubble Function

    Parameters
    ====
                
    omr: float or array-like
            omega_rad, total radiation energy density (dimensionless)
            
    omla: float or array-like
            omega_lambda, dark energy density parameter (dimensionless)
    
    omm: float or array-like
            omega_m, total matter density (dimensionless)
            
    omk: float or array-like
            omega_k, curvature of the universe, omega_k=1-omega_m-omega_lambda (dimensionless)
            
    z: float or array-like
            redshift (dimensionless)

    Returns
    ====
    E(z): float or array-like
            a function in z
            
    """
    
    oms=omr*(1+z)**4 + omm*(1+z)**3 + omk*(1+z)**2 + omla
    
    E=np.sqrt(oms)
    
    return E

def hz(H0,omr,omla,omm,omk,z):
    
    """
    Hubble Parameter

    Parameters
    ====

    H0: float or array-like
            Hubble constant, with different prior depending on SHoES or Planck measurement ((km/s)/Mpc)
                
    omr: float or array-like
            omega_rad, total radiation energy density (dimensionless)
            
    omla: float or array-like
            omega_lambda, dark energy density parameter (dimensionless)
    
    omm: float or array-like
            omega_m, total matter density (dimensionless)
            
    omk: float or array-like
            omega_k, curvature of the universe, omega_k=1-omega_m-omega_lambda (dimensionless)
            
    z: float or array-like
            redshift (dimensionless)

    Returns
    ====
    H(z): float or array-like
            The Hubble parameter H(z)=H_0*E(z)

    """

    return ez(omr,omla,omm,omk,z)*H0

def integrand(x, omr, omla, omm, omk):
    
    """
    Function to be integrated in order to find the comoving distance

    Parameters
    ====
    
    x: float or array-like
            Dummy variable for z
                
    omr: float or array-like
            omega_rad, total radiation energy density (dimensionless)
            
    omla: float or array-like
            omega_lambda, dark energy density parameter (dimensionless)
    
    omm: float or array-like
            omega_m, total matter density (dimensionless)
            
    omk: float or array-like
            omega_k, curvature of the universe, omega_k=1-omega_m-omega_lambda (dimensionless)

    Returns
    ====
    f(x): float or array-like
            f(x)=1/E(x)

    """
    
    return 1/ez(omr,omla,omm,omk,x)

def dc(H0,omr,omla,omm,omk,z):
    
    """
    Function to be integrated in order to find the comoving distance

    Parameters
    ====
    
    H0: float or array-like
            Hubble constant, with different prior depending on SHoES or Planck measurement ((km/s)/Mpc)
                
    omr: float or array-like
            omega_rad, total radiation energy density (dimensionless)
            
    omla: float or array-like
            omega_lambda, dark energy density parameter (dimensionless)
    
    omm: float or array-like
            omega_m, total matter density (dimensionless)
            
    omk: float or array-like
            omega_k, curvature of the universe, omega_k=1-omega_m-omega_lambda (dimensionless)
    
    z: float or array-like
            redshift (dimensionless)

    Returns
    ====
    D_c(z): float or array-like
            the comoving distance of a source with redshift z (Mpc)
            D_c(z)=d_H*I(z)
            where F(z) is the integral of f(x) from 0 to z

    """
       
    integral= quad(integrand, 0, z, args=(omr,omla,omm,omk)) #integrate f(x) from x=0 to z
    
    DC=dH(H0)*integral[0] #d_H*F(z)
    
    return DC

def dm(H0,omr,omla,omm,omk,z):
    
    """
    Transverse Comoving Distance

    Parameters
    ====

    H0: float or array-like
            Hubble constant, with different prior depending on SHoES or Planck measurement ((km/s)/Mpc)
                
    omr: float or array-like
            omega_rad, total radiation energy density (dimensionless)
            
    omla: float or array-like
            omega_lambda, dark energy density parameter (dimensionless)
    
    omm: float or array-like
            omega_m, total matter density (dimensionless)
            
    omk: float or array-like
            omega_k, curvature of the universe, omega_k=1-omega_m-omega_lambda (dimensionless)
            
    z: float or array-like
            redshift (dimensionless)

    Returns
    ====
    dm: float or array-like
            d_M, transverse comoving distance (Mpc)

    """
                                   
    Dc=dc(H0,omr,omla,omm,omk,z) #comoving distance
    
    dh=dH(H0) #hubble distance
    
    #transverse comoving distance depends on curvature of the universe
    
    if omk == 0: #flat universe
        DM = Dc
        
    if omk < 0: #closed universe (note: not injective)
        DM = dh/(np.sqrt(abs(omk)))*np.sin(np.sqrt(abs(omk))*Dc/dh)
    
    if omk > 0: #open universe
        DM = dh/(np.sqrt(omk))*np.sinh(np.sqrt(omk)*Dc/dh)
        
    return DM

def dl(H0,omr,omla,omm,omk,z):
    
    """
    Luminosity Distance

    Parameters
    ====

    H0: float or array-like
            Hubble constant, with different prior depending on SHoES or Planck measurement ((km/s)/Mpc)
                
    omr: float or array-like
            omega_rad, total radiation energy density (dimensionless)
            
    omla: float or array-like
            omega_lambda, dark energy density parameter (dimensionless)
    
    omm: float or array-like
            omega_m, total matter density (dimensionless)
            
    omk: float or array-like
            omega_k, curvature of the universe, omega_k=1-omega_m-omega_lambda (dimensionless)
            
    z: float or array-like
            redshift (dimensionless)

    Returns
    ====
    DL: float or array-like
            dEML, electromagnetic luminosity distance (Mpc)
            
            dEML=(1+z)*dm
            
            where dm is the transverse comoving distance
    """
    
    Dm=dm(H0,omr,omla,omm,omk,z)
        
    return (1+z)*Dm

def vc(H0,z,omr,omla,omm,omk):
    
    """
    
    !!There is an error in here. Investigate later
    
    
    Comoving Volume

    Parameters
    DM(H0,omr,omla,omm,omk,z)
    ====

    H0: float or array-like
            Hubble constant, with different prior depending on SHoES or Planck measurement ((km/s)/Mpc)
                
    omr: float or array-like
            omega_rad, total radiation energy density (dimensionless)
            
    omla: float or array-like
            omega_lambda, dark energy density parameter (dimensionless)
    
    omm: float or array-like
            omega_m, total matter density (dimensionless)
            
    omk: float or array-like
            omega_k, curvature of the universe, omega_k=1-omega_m-omega_lambda (dimensionless)
            
    z: float or array-like
            redshift (dimensionless)

    Returns
    ====
    vc: float or array-like
            V_c, comoving volume (Mpc**3)

    """
       
    Dl=dl(H0,omr,omla,omm,omk,z)
    
    dm=Dl/(1+z) #transverse comoving distance
    
    dh=dH(H0) #hubble distance

    #comoving volume depends on curvature of the universe
    
    if omk == 0: #flat universe
        Vc = (4*np.pi*dm**3)/3
        
    if omk < 0: #closed universe (note: not injective)
        Vc = -((4*np.pi*dh**3)/(2*omk)) * (dm/dh)*(np.sqrt(1+omk*((dm**2)/(dh**2))) - (1/(np.sqrt(np.abs(omk))))*np.arcsin(np.sqrt(np.abs(omk))*(dm/dh)) )
    
    if omk > 0: #open universe
        Vc = ((4*np.pi*dh**3)/(2*omk)) * (dm/dh)*(np.sqrt(1+omk*((dm**2)/(dh**2))) - (1/(np.sqrt(np.abs(omk))))*np.arcsinh(np.sqrt(np.abs(omk))*(dm/dh)) )
        
    return vc