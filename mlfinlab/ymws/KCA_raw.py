# Gemini Test code: https://gemini.google.com/app/73fa1b273cc3e54e
# Gemini KCA general + [pos,vel,acc] output: https://gemini.google.com/app/9aa2d26958b01f3b

"""
    #5) Forecast
    #... (forecast loop as before)...

    # Extract the last forecasted state
    if fwd > 0:
        next_state = x_mean[-1]  # Get the last row of x_mean
        return x_mean, x_std, x_covar, next_state 
    else:
        return x_mean, x_std, x_covar
"""


import numpy as np
from pykalman import KalmanFilter
import statsmodels.stats.diagnostic as sm3
import matplotlib.pyplot as plt
#import kca
#from selectFFT import selectFFT
import statsmodels.nonparametric.smoothers_lowess as sml
from icecream import ic


mainPath = '../../'

#A1
def fitKCA(t, z, q, fwd=0):
    '''
    # Kinetic Component Analysis

    Inputs:
    t: Iterable with time indices
    z: Iterable with measurements
    q: Scalar that multiplies the seed states covariance
    fwd: number of steps to forecast (optional, default=0)
    
    Output:
    x[0]: smoothed state means of position velocity and acceleration
    x[1]: smoothed state covar of position velocity and acceleration
    Dependencies: numpy, pykalman
    '''
    #1) Set up matrices A,H and a seed for Q
    h = (t[-1]-t[0])/t.shape[0]
    
    A = np.array([[1,h,.5*h**2],
                [0,1,h],
                [0,0,1]])
    
    Q = q * np.eye(A.shape[0])
    
    #2) Apply the filter
    kf = KalmanFilter(transition_matrices=A, transition_covariance=Q)
    
    #3) EM estimates
    kf = kf.em(z)
    
    #4) Smooth
    x_mean, x_covar = kf.smooth(z)

    #5) Forecast
    for fwd_ in range(fwd):
        x_mean_, x_covar_ = kf.filter_update(filtered_state_mean=x_mean[-1], \
        filtered_state_covariance = x_covar[-1])
        
        x_mean = np.append(x_mean, x_mean_.reshape(1,-1), axis=0)
        
        x_covar_ = np.expand_dims(x_covar_, axis=0)
        
        x_covar = np.append(x_covar, x_covar_, axis=0)
    
    #6) Std series
    x_std=(x_covar[:,0,0]**.5).reshape(-1,1)
    for i in range(1,x_covar.shape[1]):
        x_std_ = x_covar[:,i,i]**.5
        x_std = np.append(x_std, x_std_.reshape(-1,1), axis=1)
        
    return x_mean, x_std, x_covar


#A2
def selectFFT(series, minAlpha=None):
    """
    # FFT signal extraction with frequency selection
    # Implements a forward algorithm for selecting FFT frequencies
    """
    #1) Initialize variables
    series_ = series
    fftRes = np.fft.fft(series_, axis=0)
    fftRes = {i:j[0] for i,j in zip(range(fftRes.shape[0]),fftRes)}
    fftOpt = np.zeros(series_.shape, dtype=complex)
    lags, crit = int(12*(series_.shape[0]/100.)**.25), None
    
    #2) Search forward
    while True:
        key, critOld = None, crit
        
        for key_ in fftRes.keys():
            fftOpt[key_,0] = fftRes[key_]
            series__ = np.fft.ifft(fftOpt,axis=0)
            series__ = np.real(series__)
            crit_ = sm3.acorr_ljungbox(series_-series__,lags=lags) # test for the max # lags
            crit_ = crit_[0][-1],crit_[1][-1]
            
            if crit == None or crit_[0]<crit[0]:
                crit, key = crit_, key_
            fftOpt[key_,0]=0
            
        if key!=None:
            fftOpt[key,0] = fftRes[key]
            del fftRes[key]
        else:
            break
        
        if minAlpha!=None:
            if crit[1] > minAlpha:
                break
            if critOld != None and crit[0]/critOld[0] > 1-minAlpha:
                break
            
    series_ = np.fft.ifft(fftOpt,axis=0)
    series_ = np.real(series_)
    
    out={'series':series_,'fft':fftOpt,'res':fftRes,'crit':crit}
    
    return out


#A3
def getPeriodic(periods,nobs,scale,seed=0):
    t = np.linspace(0,np.pi*periods/2.,nobs)
    rnd = np.random.RandomState(seed)
    signal = np.sin(t)
    z = signal + scale*rnd.randn(nobs)

    return t, signal, z


def vsFFT():
    """
    # Kinetic Component Analysis of a periodic function
    """
    #1) Set parameters
    nobs, periods=300, 10
    
    #2) Get Periodic noisy measurements
    t, signal, z = getPeriodic(periods, nobs, scale=.5)
    
    #3) Fit KCA
    x_point,x_bands = kca.fitKCA(t, z, q=.001)[:2]
    
    #4) Plot KCA's point estimates
    color=['b','g','r']
    plt.plot(t,z,marker='x',linestyle='',label='measurements')
    plt.plot(t,x_point[:,0],marker='o',linestyle='-',label='position', \
    color=color[0])
    plt.plot(t,x_point[:,1],marker='o',linestyle='-',label='velocity', \
    color=color[1])
    plt.plot(t,x_point[:,2],marker='o',linestyle='-',label='acceleration', \
    color=color[2])
    
    plt.legend(loc='lower left',prop={'size':8})
    plt.savefig(mainPath+'Data/test/Figure1.png')
    
    #5) Plot KCA's confidence intervals (2 std)
    for i in range(x_bands.shape[1]):
        plt.plot(t,x_point[:,i]-2*x_bands[:,i],linestyle='-',color=color[i])
        plt.plot(t,x_point[:,i]+2*x_bands[:,i],linestyle='-',color=color[i])
        
    plt.legend(loc='lower left',prop={'size':8})
    plt.savefig(mainPath+'Data/test/Figure2.png')
    plt.clf(); plt.close() # reset pylab
    
    #6) Plot comparison with FFT
    fft = selectFFT(z.reshape(-1,1),minAlpha=.05)
    plt.plot(t,signal,marker='x',linestyle='',label='Signal')
    plt.plot(t,x_point[:,0],marker='o',linestyle='-',label='KCA position')
    plt.plot(t,fft['series'],marker='o',linestyle='-',label='FFT position')
    plt.legend(loc='lower left',prop={'size':8})
    plt.savefig(mainPath+'Data/test/Figure3.png')
    
    return


#A4
def vsLOWESS():
    """
    # Kinetic Component Analysis of a periodic function
    """
    #1) Set parameters
    nobs, periods, frac = 300, 10, [.5,.25,.1]
    
    #2) Get Periodic noisy measurements
    t, signal, z = getPeriodic(periods,nobs,scale=.5)
    
    #3) Fit KCA
    x_point, x_bands = fitKCA(t,z,q=.001)[:2]
    
    #4) Plot comparison with LOWESS
    plt.plot(t,z,marker='o',linestyle='',label='measurements')
    plt.plot(t,signal,marker='x',linestyle='',label='Signal')
    plt.plot(t,x_point[:,0],marker='o',linestyle='-',label='KCA position')
    
    for frac_ in frac:
        lowess = sml.lowess(z.flatten(), range(z.shape[0]),
                            frac=frac_)[:,1].reshape(-1,1)
        
    plt.plot(t,lowess,marker='o',linestyle='-',label='LOWESS('+str(frac_)+')')
    plt.legend(loc='lower left',prop={'size':8})
    plt.savefig(mainPath+'Data/test/Figure4.png')
    
    return

#-----------------------------------------

# Test code
def test_fitKCA():
    """
    Tests the fitKCA function with synthetic data and various parameters.
    """

    # Test Case 1: Basic test with no noise and no forecasting
    t1 = np.arange(0, 10, 0.1)
    z1 = 0.5 * t1**2  # True trajectory (parabolic)
    q1 = 0.01 
    x_mean1, x_std1, x_covar1 = fitKCA(t1, z1, q1)

    # Assertions for Test Case 1
    assert x_mean1.shape[0] == t1.shape[0], "Output size mismatch in Test Case 1"
    assert x_mean1.shape[1] == 3, "Output should have 3 state variables (position, velocity, acceleration)"

    # Visualization for Test Case 1
    plt.figure(figsize=(10, 5))
    plt.plot(t1, z1, 'b.', label='Noisy Measurements')
    plt.plot(t1, x_mean1[:, 0], 'r-', label='Smoothed Position')
    plt.fill_between(t1, x_mean1[:, 0] - x_std1[:, 0], x_mean1[:, 0] + x_std1[:, 0], color='r', alpha=0.2)
    plt.title("Test Case 1: Basic Test (No Noise, No Forecasting)")
    plt.legend()
    plt.show() #show each plot separately

    # Test Case 2: Test with added noise and no forecasting
    np.random.seed(42)
    noise2 = np.random.normal(0, 1, t1.shape)
    z2 = z1 + noise2
    q2 = 0.1
    x_mean2, x_std2, x_covar2 = fitKCA(t1, z2, q2)

    # Assertions for Test Case 2
    assert x_mean2.shape[0] == t1.shape[0], "Output size mismatch in Test Case 2"

    # Visualization for Test Case 2
    plt.figure(figsize=(10, 5))
    plt.plot(t1, z2, 'b.', label='Noisy Measurements')
    plt.plot(t1, x_mean2[:, 0], 'r-', label='Smoothed Position')
    plt.fill_between(t1, x_mean2[:, 0] - x_std2[:, 0], x_mean2[:, 0] + x_std2[:, 0], color='r', alpha=0.2)
    plt.title("Test Case 2: Test with Noise (No Forecasting)")
    plt.legend()
    plt.show() #show each plot separately

    # Test Case 3: Test with forecasting
    fwd3 = 5
    x_mean3, x_std3, x_covar3 = fitKCA(t1, z2, q2, fwd=fwd3)
    t3 = np.arange(0, 10 + fwd3 * 0.1, 0.1)

    # Assertions for Test Case 3
    assert x_mean3.shape[0] == t1.shape[0] + fwd3, "Output size mismatch in Test Case 3 (Forecasting)"

    # Visualization for Test Case 3
    plt.figure(figsize=(10, 5))
    plt.plot(t1, z2, 'b.', label='Noisy Measurements')
    plt.plot(t3, x_mean3[:, 0], 'r-', label='Smoothed/Forecasted Position')
    plt.fill_between(t3, x_mean3[:, 0] - x_std3[:, 0], x_mean3[:, 0] + x_std3[:, 0], color='r', alpha=0.2)
    plt.title("Test Case 3: Test with Forecasting")
    plt.legend()
    plt.show() #show each plot separately

    print("All test cases passed!")
    ic(x_mean1.shape[0])
    ic(x_mean1.shape[1])
    ic(x_mean2.shape[0])
    ic(x_mean2.shape[1])
    ic(x_mean3.shape[0])
    ic(x_mean3.shape[1])


if __name__ == "__main__":
    test_fitKCA()
#    vsFFT()
#   vsLOWESS()
