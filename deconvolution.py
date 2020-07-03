from pylab import linspace, plot, legend, randn
from scipy import stats, signal, fftpack
from spectrum import aryule, Periodogram, arma2psd, arma_estimate
from math import sin, cos, exp
import matplotlib.pyplot as plt
import numpy as np
from pykalman import KalmanFilter
# Setup:
fs = 11  # Sampling Frequency [Hz]
dt = 20  # Experiment runtime [s]
t = np.linspace(0, dt, dt*fs+1) # Time [s]
np.random.seed(19)
def h(t):
    return 3*exp(-t)*cos(2*t)
#noise
I = np.matrix(np.identity(8))
Q = 0.004
R = 20
wnu = np.random.normal(0, 1, t.size)
#wk = np.random.multivariate_normal([0,0,0,0,0,0,0,0], Q, t.size)
nk = np.random.normal(0, R, t.size) # vk
# impulse response
ht = list(map(h, t))
#u is here generated with parameters a. In the real world these parameters need to be estimated
a = [1, -2.21, 2.94, -2.17, 0.96]
u = signal.lfilter([1], a, [wnu])[0]
# output
yt = signal.convolve(u, ht, mode='full')[:t.size]
yt = list(map(lambda y, n: y+n, yt, nk)) 

# Fourier Transform:
Ys = fftpack.fft(yt)
Hs = fftpack.fft(ht) # Transfer function

#inverse naive method
Ns = fftpack.fft(nk)
Us = list(map(lambda y, h: y/h, Ys, Hs)) 
naive = fftpack.ifft(Us)
naive[0] = 0

#wiener
Hs_conj = list(map(lambda h: np.conj(h), Hs)) 
Hs_squ = list(map(lambda h: h*np.conj(h), Hs)) 

def autocorr(x):
    result = np.correlate(x, x, mode="full")
    return result[result.size // 2:]

Ss = fftpack.fft(autocorr(u))
Ns = fftpack.fft(autocorr(nk))
Gs = list(map(lambda hc, hs, s, n: hc*s / (hs * s + n) , Hs_conj, Hs_squ, Ss, Ns)) 
Us = list(map(lambda g, y: g*y, Gs, Ys)) 
wiener = fftpack.ifft(Us)
wiener[0] = 0

#kalman
A = np.matrix(((0, 1, 0, 0, 0, 0, 0, 0), 
               (-exp(-2), 2*exp(-1)*cos(2), 1, 0, 0, 0, 0, 0), 
               (0, 0, 0, -0.96, 2.17, -2.94, 2.21, 1), 
               (0, 0, 0, 0, 1, 0, 0, 0),
               (0, 0, 0, 0, 0, 1, 0, 0),
               (0, 0, 0, 0, 0, 0, 1, 0),
               (0, 0, 0, -0.96, 2.17, -2.94, 2.21, 1),
               (0, 0, 0, 0, 0, 0, 0, 0)))
B = np.transpose(np.matrix((0,0,0,0,0,0,0,1)))
C = np.matrix((-3*exp(-2),3*exp(-1)*cos(2),3,0,0,0,0,0))
x = np.transpose(np.matrix((0,0,0,0,0,0,0,0)))
kalman = [0]
P = np.matrix(np.identity(8))

for i in range(1, len(t)):
    xpred = A*x
    Ppred = A*P*np.transpose(A) + B*Q*np.transpose(B)
    K = Ppred*np.transpose(C) / (C*Ppred*np.transpose(C)+R)
    x = xpred + K*(yt[i]-C*xpred)
    P = (I-K*C)*Ppred
    kalman.append(np.matrix((0,0,1,0,0,0,0,0))*x)
kalman.pop(0)
kalman.insert(len(kalman)-1, u[len(u)-1])

ht = list(map(lambda h: 20*h, ht))
#errors
ew = list(map(lambda u, w: (u-w)*(u-w), u , wiener)) 
en = list(map(lambda u, n: (u-n)*(u-n), u , naive)) 
ek = list(map(lambda u, k: (u-k)*(u-k), u , kalman)) 
#plt.plot(t, ew, label="wiener error")
#plt.plot(t, en, label="naive error")
#plt.plot(t, ek, label="kalman error")
plt.plot(t, u, label="u input signal")
#plt.plot(t, yt, label="y output signal")
#plt.plot(t, ht, label="h impulse response")
#plt.plot(t, kalman, label="kalman")
#plt.plot(t, naive, label="naive")
plt.plot(t, wiener, label="wiener")
plt.legend()

plt.xlabel("time t")
plt.ylabel("unspecified unit")
plt.show()
#plt.savefig('test.pdf')
