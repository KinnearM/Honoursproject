"""
Class objects to be used for constaining the number of dimensions experienced by gravity + some other useful functions
by Meryl
"""

import cosmo

import numpy as np

import scipy.interpolate as inter

import astropy.visualization as vis
import emcee
import corner


import matplotlib.pylab as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif',size=18)
rc('legend', fontsize=12) 

c=299792.458 #speed of light(km/s)

class leakageparam:
    """
    "leakageparam" object, takes distance posterior samples and converts to likelihood function. Can then sample for gamma or whatever
    data is a .txt file containing distance posterior samples for the GW event.
    z is the redshift of the EM event, a tuple of mean value and sigma.

    distprior is the luminosity distance prior used for GW parameter estimation can be 'Euclidean' or 'Planck15'. Defaults to Euclidean.
    Priors is the set of priors to use for sampling for gamma. Can be 'flat', 'Planck15', 'Planck18', 'Shoes', or 'Shoes'. Defaults to Planck18.
    
    Once class is initialised, you can use:
    self.EMdist to return the distance of the EM source at the mean redshift, with the cosmology model specified in 'priors'
    
    self.plotlikelihood(EM=False) will plot the renormalised distance likelihood distribution, setting EM=True will also plot EMdist as a black line.
    (then self.like returns x,y cublic spline interpolation of likelihood function)
    self.sample(nsamples, nwalkers) uses emcee to sample the posterior function specified by the class options.
    
    Once the sampler has run you can immediately return:
    self.chain 
    self.samples: all samples
    self.gammasamples: just the marginalised gamma samples
    
    or run:
    self.ratio() to print the acceptance ratio
    self.corner() to plot a corner plot suing corner
    self.results() to print mean gamma +/- 68% confidence interval
    self.plot() to plot marginalised gamma posterior with 90% confindence intervals.
    """

    def __init__(self, data, z=None, distprior='Euclidean', priors='Planck18'):
        self.data = np.loadtxt(data) #distance posterior samples
        self.z=z #redshift: mean, sigma
        self.flat=False
        
        self.logprob=self.Logprob #setting cosmo to False allows sampling in H0 and omega_matter
        self.sample=self.Sample #samples in 4 parameters
        self.corner=self.Corner #plots in 4 parameters
        
        self.gmin=0.75
        self.gmax=1.2

        if priors=='Planck18':
            self.H=67.37,0.54
            self.omm=0.3147,0.0074
        
        if priors=='Planck15':
            self.H=67.81, 0.92
            self.omm= 0.308, 0.012
            
        if priors=='SH0ES19':
            self.H=73.20, 1.3
            self.omm=0.30, 0.13
            
        if priors=='flat':
            self.flat=True
            self.logprob=self.logprobflat
            self.H=160,80
            self.omm=0.6,0.3
            self.gmin=0.5
            self.gmax=2.5
                   
        if distprior=='Euclidean':
            self.transprior=self.dsq
            
        if distprior== 'Planck15':
            tp=np.loadtxt('dvddl.txt') #Planck15 dist prior uniform in comoving volume (not time) 
            self.transprior = inter.interp1d(tp[0],tp[1]) #prior used to get p(x|d) from p(d|x)
            
        self.minz=self.z[0]-5*self.z[1]
        self.maxz=self.z[0]+5*self.z[1]
        self.minH=self.H[0]-5*self.H[1]
        self.maxH=self.H[0]+5*self.H[1]
        self.minomm=self.omm[0]-5*self.omm[1]
        self.maxomm=self.omm[0]+5*self.omm[1]
                
        self.EMdist = cosmo.DL(self.z[0], self.H[0], self.omm[0]) #finds mean distance of EM signal
        if self.flat==True:
            self.EMdist = cosmo.DL(self.z[0], 67.37, 0.3147)

        ##Get posterior ditribution from samples. Is there a better way to do this?
            
        self.n, self.bins=histo(self.data) #turn posterior samples into posterior distribution
        self.shift=(self.bins[1]-self.bins[0])/2 #Add on to find center of bin. bins are labelled by lowest distvalue contained in bin ie the left
        
        ##Convert to likelihood from posterior dist
       
        for i, I in enumerate(self.n): #why am I so unsure of this?
            self.n[i]=I/self.transprior(self.bins[i]+self.shift) #for each p(d|x) (d center of bin) divide by p(d) to get distribution propto p(x|d)
        
        ##renormalise
        summ=np.sum(self.n)
        width=self.bins[1]-self.bins[0]
        for i, I in enumerate(self.n):
            self.n[i]=I/(summ*width)
        
        ##Find peak luminosity distance
        
        elem = np.argmax(self.n) #find index of bin the distribution peaks in
        self.peak=self.bins[elem] +self.shift  #find bin, centre on bin to find peak d
        
        ##Find peak gamma to give emcee a starting point on mode
        
        self.gammas=np.zeros(10000) #array to fill with P(gamma|theta) where theta is the most probable value of each other parameter
        gammarange=np.linspace(self.gmin,self.gmax,10000)
        
        for i,I in enumerate(gammarange):
            
            theta= I, self.H[0], self.omm[0], self.z[0]            
            self.gammas[i]=self.logprob(theta)[0] #P(gamma|theta)
                
        maxarg=np.argmax(self.gammas)
        self.gamma0P=gammarange[maxarg] #gamma peak
        
    def plotlikelihood(self, EM=False):
        
        """
        Plot distance likelihood function against luminosity distance
        
        matplotlib plot with standard settings for report
        
        EM=True plots d^EM_L as a black line
        """

        rc('font', family='serif',size=18)
        rc('legend', fontsize=12) 

        fig = plt.figure(figsize=(7.1, 5)) #7.1cm is textwidth for default one column aastex template
        
        ax = fig.add_subplot(111)
        
        if EM==True:
            ax.plot(np.ones(100)*self.EMdist,np.linspace(-1,10,100),c='black',lw=3)

        ran=np.linspace(self.bins[0],self.bins[-1],10000)
        P=inter.CubicSpline(self.bins[:-1],self.n)
        ax.plot(ran,P(ran),c='C0',lw=3)

        ax.set_xlabel('$d_L$ (Mpc)')
        ax.set_ylabel('$\propto p(x|d_L)$')

        ax.set_ylim(-max(self.n)/10,max(self.n)+max(self.n/10))
        #ax.set_xlim(self.bins[0],self.bins[-2])

        if self.EMdist>1000:
            ax.xaxis.set_minor_locator(MultipleLocator(500))
        if 1000>=self.EMdist>100:
            ax.xaxis.set_minor_locator(MultipleLocator(50))
        if self.EMdist<=100:
            ax.xaxis.set_minor_locator(MultipleLocator(2.5))

        ax.tick_params('both',direction='in', which='major', width=2, length=8)

        ax.tick_params(direction='in', which='minor', width=1, length =5)

        plt.show()
        
        self.like = ran, P(ran)
            
        
        
        
    def  prior(self,gamma,H0,omm,z):
        
        """
        p(gamma,H0,z,omm)=p(gamma)p(H0)p(z)p(omm)
        
        """
        
        H0p=gaussx(self.H[0],self.H[1],H0) #gaussian
        
        ommp=gaussx(self.omm[0],self.omm[1],omm) #gaussian centred on omm given on class initiation
        
        zp=zprior(z, H0, omm) #uniform in comoving volume time
              
        return H0p*ommp*zp
        
        
    def zlike(self,zt):
        
        """
        !! sigmaz is extremely small. Is there a better way to do this?
        
        p(z|zt)=N(zt,sigmaz)(z)
        
        likelihood of measuring z given a true value of zt
        """
            
        Pz=gaussx(zt,self.z[1],self.z[0]) #likelihood is a gaussian centred at zt with sigmaz, evaluated at the observed value of z
    
        return Pz
        
    def likefn(self,gamma,H0,omm,z):
        
        """
        p(gamma,H0,d,omm|z)=p(x|d(gamma,H0,omm,z))p(x|z))p0(z|gamma,H0,omm)
        =p(x|d(gamma,H0,omm,z))p(x|z))p0(gamma,H0,omm)
        
        posterior function
        """
        dEM=cosmo.DL(z, H0, omm)
        
        dGW=dEM**gamma
        
        Pxd= GWD(dGW, self.n, self.bins)

        Pz=self.zlike(z)
    
        priors=self.prior(gamma,H0,omm,z)

        post=Pxd*Pz*priors*beta(gamma, z, H0, omm)
    
        return  post
    
    def likefnflat(self,gamma,H0,omm,z):
        
        """
        p(gamma,H0,d,omm|z)=p(x|d(gamma,H0,omm,z))p(x|z))p0(z|gamma,H0,omm)
        =p(x|d(gamma,H0,omm,z))p(x|z))p0(gamma,H0,omm)
        
        posterior function
        """
        dEM=cosmo.DL(z, H0, omm)
        
        dGW=dEM**gamma
        
        Pxd= GWD(dGW, self.n, self.bins)

        Pz=self.zlike(z)
    
        priors=zprior(z, H0, omm)

        post=Pxd*Pz*priors*beta(gamma, z, H0, omm)
    
        return  post
    

   
    def Logprob(self,theta):
        
        """       
        logprob function for emcee for far away sources
        
        inputs theta, an array of length ndim
        
        returns log posterior and log prior as a tuple
        """
    
        gamma, H0, omm, z = theta
    
        loglike= -np.inf
        
        logprior=-np.inf
        
        #prob=0 for silly parameter values
    
        if self.gmin<gamma<self.gmax and self.minH<H0<self.maxH and self.minomm<=omm<=self.maxomm and self.minz<z<self.maxz:

            oms=omm*(1+z)**3 + (1-omm)
        
            if oms>0:
                
                loglike = np.log(self.likefn(gamma,H0,omm,z))
                logprior=np.log(self.prior(gamma,H0,omm,z))
                
        return loglike, logprior
    
    def logprobflat(self,theta):
        
        """       
        logprob function for emcee
        
        inputs theta, an array of length ndim
        
        returns log posterior and log prior as a tuple
        """
    
        gamma, H0, omm, z = theta
    
        loglike= -np.inf
        
        logprior=-np.inf
        
        #prob=0 for silly parameter values
    
        if self.gmin<gamma<self.gmax and 20<H0<300 and 0.2<=omm<=1 and self.minz<z<self.maxz:
        
            oms=omm*(1+z)**3 + (1-omm)
        
            if oms>0:
                
                loglike = np.log(self.likefnflat(gamma,H0,omm,z))
                logprior=np.log(zprior(z, H0, omm))
                
        return loglike, logprior
       
    
    def dsq(self,d):
        
        return 4*np.pi*d**2
    
        
    def Sample(self,sampleno, nwalkers):
        
        """
        run emcee sampler for far away sources in 4 parameters
        """
        
        self.ndim = 4
        self.sampleno=sampleno
        self.nwalkers=nwalkers
            
        p0=np.zeros([self.nwalkers,self.ndim])
        
        if self.flat==True:
            gamerr=1e-3
            self.gamma0P=1
        else:
            gamerr=1e-6
            
        p0=np.zeros([self.nwalkers,self.ndim])
        for i in range(self.nwalkers):
            p0[i][0]=self.gamma0P + gamerr*np.random.randn() #this is ridiculous, please fix this.
            p0[i][1]=self.H[0] + self.H[1]*np.random.randn()
            p0[i][2]=self.omm[0] + self.omm[1]*np.random.randn()
            p0[i][3]=self.z[0] + self.z[1]*np.random.randn()

            
        sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.logprob)
        
        sampler.run_mcmc(p0,self.sampleno)
        
        self.chain=sampler.chain
        self.acceptance=sampler.acceptance_fraction       
        self.samples=self.chain[:, 100:, :].reshape((-1, self.ndim))
        self.gammasamples=self.samples.T[0]
                        
        self.histosamp=np.reshape(self.chain[:,100:,0],self.nwalkers*(self.sampleno-100))
        
        num=len(self.histosamp)
        if num<10000:
            binnum=20
        if 10000<=num<20000:
            binnum=25
        if 20000<=num:
            binnum=45
        
        n, bins, patch = vis.hist(self.histosamp,bins=binnum,range=(self.gmin,self.gmax))
        plt.clf() #dinnae
        
        summ=np.sum(n)
        width=bins[1]-bins[0]
        for i, I in enumerate(n):
            n[i]=I/(summ*width)
        
        self.Pgamma= n, bins
        
    def ratio(self):
        
        print("Mean acceptance fraction: {0:.3f}".format(np.mean(self.acceptance)))
        
    def Corner(self):
        
        fig = corner.corner(self.samples, labels=["$\gamma$", "$H_0$","$\Omega_{m}$","$z$" ])

    
    def results(self):
        total=sum(self.Pgamma[0])
        n=self.Pgamma[0]
        bins=self.Pgamma[1]
        
        self.shiftt=(bins[1]-bins[0])/2
        
        CDF=np.zeros(len(n))
        for i, I in enumerate(self.Pgamma[0]):
            summ=0
            for j, J in enumerate(np.arange(i)):
                summ+=n[j]/total
                CDF[i]=summ
        self.cdffunc=inter.interp1d(CDF,bins[:-1]) 
        
        self.max=np.argmax(self.Pgamma[0])
        self.Max=self.Pgamma[1][self.max]
        
        self.P16=self.cdffunc(0.16)+self.shiftt
        self.P84=self.cdffunc(0.84)+self.shiftt
     
        P16=self.P16*2+2
        P84=self.P84*2+2
        
        Pup=self.P84-self.Max
        Plow=self.Max-self.P16
               
        DPup=P84-(self.Max*2+2)
        DPlow=(self.Max*2+2)-P16
               
        print('For $\gamma$: {0:.2f}+{1:.2f}-{2:.2f}'.format(self.Max,Pup,Plow))
        
        print('For D: {0:.2f}+{1:.2f}-{2:.2f}'.format(self.Max*2+2,DPup,DPlow))
        
    def plot(self):

        self.P95=self.cdffunc(0.95)

        self.P5=self.cdffunc(0.05)

        bins=self.Pgamma[1][:-1]+self.shiftt
        nP = self.Pgamma[0]
        
        ran=np.linspace(bins[0],bins[-1],10000)
        P=inter.CubicSpline(bins,nP)

        rc('font', family='serif',size=18)
        rc('legend', fontsize=12) 

        fig = plt.figure(figsize=(7.1, 5))
        ax = fig.add_subplot(111)
        ax.plot(np.ones(100)*4,np.linspace(-self.max,self.max*2,100),c='black',lw=3)
        ax.plot(ran*2+2,P(ran),c='C0',lw=3)

        
        MAX=max(nP)

        ax.set_xlabel('$D$')
        ax.set_ylabel('$p(D)$')
        axT = ax.twiny()
        axT.set_xlabel('$\gamma$')

        ax.set_xlim(2*self.gmin+2, 2*self.gmax+2)
        ax.set_ylim(-MAX/10,MAX+MAX/10)
        axT.set_xlim(self.gmin,self.gmax)
        nonsym_lim_low=self.P5
        nonsym_lim_high=self.P95
        ax.xaxis.set_minor_locator(MultipleLocator(0.05))
        axT.xaxis.set_minor_locator(MultipleLocator(0.025))
        ax.tick_params('both',direction='in', which='major', width=2, length=8)
        axT.tick_params(direction='in', which='major', width=2, length =8)
        ax.tick_params(direction='in', which='minor', width=1, length =5)
        axT.tick_params(direction='in', which='minor', width=1, length =5)
        axT.plot([nonsym_lim_low,nonsym_lim_low],[-max(nP),max(nP)*2],'C0',ls='--',lw=3)
        axT.plot([nonsym_lim_high,nonsym_lim_high],[-max(nP),max(nP)*2],'C0',ls='--',lw=3)

   
        plt.show()
    
    


"""
Stats stuffs
==================================================================
"""

def gaussx(m,s,x):
    
    """
    A 1-d gaussian distribution

    Parameters
    ====

    m: float 
            mean
                
    s: float 
            standard deviation
            
    x: float or array-like
            variable

    Returns
    ====
    g(x): float or array-like
            the gaussian with mean m and standard deviation s evaluated at x
    """
    
    gauss = 1/(np.sqrt(2*np.pi)*s)*np.exp((-0.5*((x-m)/s)**2))
    
    return gauss

def histo(data):
    
    """
    Plot data as normalised histogram with 100 bins

    Parameters
    ====
    data: array
            sample data    
    Returns
    ====
    n:  array
            probability in each bin 
    
    bins: array
            histogram bins corresponding to samples
    """        
    num=len(data) #trying to limit noise without losing detail, tried some argorithms for picking bins but not sure if they apply to samples like this
    if num<10000:
        binnum=20
    if 10000<=num<20000:
        binnum=25
    if 20000<=num:
        binnum=45
        
    n, bins, patches = vis.hist(data, bins=binnum) #counts samples in each bin
    
    plt.clf() #don't plot it please
    
    return n, bins #just return the P(A\B) x A array


def GWD(dGWL, n, bins):
    
    """
    P(x|d), ie probability of observing GW data given signal came from distance d

    Parameters
    ====
    dGWL: float or array
        the d at which we wish to evaluate P(x|d)  
         
    n: array
        the P(x|d) values corresponding to bins from histo function
        
    bins: array
        the d values from histo fn
        
    Returns
    ====
    P(x|d):  float or array
        converts continuous dGWL values to discrete bins to allow for evaluation of P(x|d) 

    """ 
    
    prob=0 #likelihood is 0 outside of bins
    
    if bins[0]<=dGWL<=bins[-2]: #otherwise
        
        ind=np.searchsorted(bins, dGWL, side='left') #finds index of bin containing d
        
        prob=n[ind] #likelihood at d
        
    return prob

"""
Priors
===========================================================================================
"""

def dVdz(z, H0, omm):
    #dV_c/dz, 
    if z>0:
        d=cosmo.DL(z,H0,omm)
        dm=d/(1+z)
        ez=cosmo.EZ(z,omm)
        prior=cosmo.dH(H0)*dm**2/ez
    else:
        prior=0
    return prior

def zprior(z, H0, omm):
    #1/(1+z)*dV_c/dz, 
    if z>0:
        d=cosmo.DL(z,H0,omm)
        dm=d/(1+z)
        ez=cosmo.EZ(z,omm)
        prior=cosmo.dH(H0)*dm**2/(ez*(1+z))
    else:
        prior=0
    return prior

def ddLdz(z, H0, omm):
    d=cosmo.DL(z,H0,omm)
    return (1+z)*c/(H0*cosmo.EZ(z,omm))+d/(1+z)

def beta(gamma, z, H0, omm):
    
    d=cosmo.DL(z,H0,omm)
    ez=cosmo.EZ(z,omm)
    
    front=(4*np.pi*d**(3*gamma))/(1+z)**4
    back=gamma-1+(c/H0)*((gamma*(1+z))/(d*ez))
    return front*back/dVdz(z, H0, omm)
