'''
x == the true period that we don't know
y == the modeled IBI period with distribution f(y) = ibi(y)

We know x(t) for t = 0, 1, ..., N
we know y(t) for t = 0, 1, ..., N, and the current estimate N+1

Define e(t) = x(t) - y(t) for all t = 0, 1, ..., N

We want to estimate x(N+1) given y(N+1) and e
'''

# Package imports
import adaptfilt # import adaptfilt
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import scipy.integrate as integrate
from math import sqrt
import scipy.stats

def filt(observedhist, estimatedhist):   
	taps = 2
	step = 1
	projorder = 1
	y,e,w = adaptfilt.ap(observedhist, estimatedhist, taps, step, projorder)
	return y,e,w

def generator(mean, std, bvec, size=50):
	result = []
	val = np.random.normal(mean, std, size=1)[0]
	
	for i in range(size):
		if len(result) >= len(bvec):
			val = 0
			for j in range(len(bvec)):
				val = val + (bvec[j] * result[-1*(j+1)])
		else:
			val = np.random.normal(mean, std, size=1)[0]
			
		result.append(val)
	return result
	
## TODO
## ## Check armu which is the same as in paper # Line 38
# IBI paper https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4278369/
def arfit(inputs, inc, order=10):
	# Check edge cases
	start = 2
	if len(inputs) < 3:
		return [0,0]
	if len(inputs) < order:
		order = len(inputs)-1
   
	# Initialize empty matrices
	A = np.zeros((order-start,order-start))
	b = np.zeros(order-start)
   
   	# Compute fitted weights
	for i in range(start, order):
		for j in range(len(inputs[:i])-1):
			A[i-start][j] = inputs[j]
		b[i-start] = inputs[i]    

	# Solve system of equations
	A = np.array(A)
	b = np.array(b)
	weights = np.linalg.solve(A, b)
	return weights

def armu(periods, inc):
	# Fit model
	mdl = arfit(periods, inc) 

	# Update weights
	if len(mdl) > 2:
		mn = mdl[0]
		for j in range(1, len(mdl)):
			mn = mn + (mdl[j] * periods[j])
		return mn
	else:
		return 0

def ibipdf(t, currentabsolutepeaktime, var, periods, inc, window_size=6):
	# Check edge cases
	if t - currentabsolutepeaktime <= 0:
		diff1 = 1
		diff2 = 0
	else:
		diff1 = ((t-currentabsolutepeaktime)**2)
		diff2 = math.log(t - currentabsolutepeaktime)
	if len(periods) == 0:
		return 0
	if math.sqrt(2 * math.pi * var * diff1) == 0:
		return 0
	if var == 0:
		return 0
	armu_val = armu(periods[:window_size],inc)
	if armu_val == 0:
		return 0

	# Compute distribution
	base = 1.0 / math.sqrt(2 * math.pi * var * diff1)
	expterm = -0.5 * ((diff2 - armu_val)**2) * 1.0 / var
	result = base * math.exp(expterm)
	return result    

# w_(k-j) = w_k - (sum_(i=k-j)^(k) u_i  )
# H[0] - (sum_(i=1)^(j) H[i]  )
# H = array that has absolute observation and set of breath intervals
# t = current time
# a = weight for leaky integrator
# t_horizon = time considered to be infty for next breath
# returns float
def mle_ibi(H, t, inc, a=0.3, t_horizon=100):
	n = len(H) - 1 # Number of peak observations (first element is absolute observation -> count-1)
	w = lambda w_a, w_t, w_u: math.exp(-w_a*(w_t-w_u)) # Weight function (leaky integrator)
	u_n = H[n]

	# Compute MLE summation
	total_sum = 0.0
	for i in range(2, n):
		curr_H = [sum(H[-i:])] + H[-i:]
		abs_obs = curr_H[0] # Absolute observation
		w_k = curr_H[1]
		ibi_val = ibipdf(t, abs_obs, np.var(curr_H[1:]), curr_H[1:],inc)
		sum1 = w(a, t, abs_obs) * ibi_val
		total_sum += sum1
	
	# Check edge case for integral
	end_int = H[0]
	if end_int < t:
		end_int = t_horizon+t
	elif end_int > t_horizon+t:
		end_int = t_horizon+t

	# Compute actual integral and sum to rest of summation
	integral_res = integrate.quad( lambda x: ibipdf(x, H[0], np.var(H[1:]), H[1:],inc), t-H[0], end_int)
	sum2 = w(a, t, u_n) * integral_res[0]
	total_sum += sum2
	return total_sum

# First element is global absolute time of next breath
# Second element is period that would be required to reach absolute time given prior period
# Prior observed periods
def generate_future_H(w_next, curr_H, curr_t):
	res = [w_next+curr_t] + [w_next + curr_t - sum(curr_H[1:])] + curr_H[1:]
	return res


def print_and_plot(peaks):

    observed = peaks
    periods = np.diff(peaks)

    estimated = []
    este = []
    ye = []
    maxxs = []

	# Solve for H that gives maximal mle_val 
    for i in range(3, len(observed)):
		tmp1 = [observed[i-1]] 
		tmp2 = np.diff(observed[:i-1]).tolist()
		H = tmp1 + tmp2
		t = observed[i]

		# Choose max index from generated mle_ibi
		poss_vals = np.linspace(0, 10, 1000)
		ibi_vals = [mle_ibi(generate_future_H(x, H, t), t, i) for x in poss_vals]
		max_indx = np.argmax(ibi_vals)
		max_x = poss_vals[max_indx]

		# Print estimate of next observation
		if max_x > 0:
			print("Current observation is", observed[i-1], "I predict the next peak will be in time", max_x, "at", observed[i-1]+max_x)
			estimated.append(observed[i-1]+max_x)
			maxxs.append(max_x)

	# Filter observations and estimates
    for i in range(len(estimated)-1):
		ee = estimated[i] - observed[i+2]
		este.append(ee)
    y,e,w = filt(estimated, observed)
    for i in range(2, len(y)):
		ee = y[i] - observed[i+1]
		ye.append(ee)

	# Print estimates
    print('estimated', estimated)
    print('observed', observed)
    print('y', y)
    print('e', e)
    print('este', este)
    print('ye', ye)
    print('periods', periods)
    print('maxxs', maxxs)

	# Format estimates
    observeddiff = periods
    ydiff = np.diff(y)
    estimateddiff = maxxs

	# Compute RMS goodness of fit of y to observed for post-filtering, estimated to observed for pre-filtering predicted estimate fit
    burnin=0
    msepre = mean_squared_error(observeddiff[-1*min(len(observeddiff), len(estimateddiff))+burnin:], estimateddiff[-1*min(len(observeddiff), len(estimateddiff))+burnin:])
    msepost = mean_squared_error(observeddiff[-1*min(len(observeddiff), len(ydiff))+burnin:], ydiff[-1*min(len(observeddiff), len(ydiff))+burnin:])
    print("RMS error on predictions pre and post adaptive filtering", math.sqrt(msepre), math.sqrt(msepost))

	# Compute chi square goodness of fit
    chipre = scipy.stats.chisquare(estimateddiff[-1*min(len(observeddiff), len(estimateddiff))+burnin:], f_exp=observeddiff[-1*min(len(observeddiff), len(estimateddiff))+burnin:])
    chipost = scipy.stats.chisquare(ydiff[-1*min(len(observeddiff), len(ydiff))+burnin:], f_exp=observeddiff[-1*min(len(observeddiff), len(ydiff))+burnin:])
    print("Chi Squared goodness of fit pre and post adaptive filtering", chipre, chipost)

	# Plot figures
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Time (sec)")
    ax1.plot(range(len(observeddiff)), observeddiff, 'r-', label="Observed")
    ax1.plot(range(1,len(estimateddiff)+1), estimateddiff, 'go', label="Estimated a priori")
    ax1.plot(range(len(estimated)-len(este)+1, len(ydiff)+len(estimated)-len(este)+1), ydiff, 'b-', label="Corrected by Adaptive Filter")
    ax1.legend(loc=0)
    ax2.plot(range(len(estimated)-len(este), len(estimated)-1), este[1:], 'm-', label="Model Estimate Error")
    ax2.plot(range(len(estimated)-len(este)+1, len(e)+len(estimated)-len(este)+1), e, 'c-', label="Adaptive Error")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Time (sec)")
    ax2.legend(loc=0)
    plt.suptitle("Model Predicted Interbreath Interval (IBI) Times and Adaptive Filter from Observed Error Terms")
    plt.show()




### MAIN CODE ###
if __name__ == '__main__':
	#size=50
	#estimated = generator(10, 2, [0.333, 0.166, 0.502], size=size+1) # vector of prior actual observations and next model estimated value
	#observed = estimated[:len(estimated)-1] + np.random.normal(0, 1, size=len(estimated)-1) # vector of prior model estimates
    peaks = [0.095143333, 0.7292695, 1.628975, 2.390444, 3.5081365, 4.330961, 5.650636, 6.63006825, 7.4078774, 7.929061, 8.5092756, 9.9506358, 11.78660725, 12.5715562, 13.8088214, 14.948019, 15.7698166, 16.930458, 17.6903102, 18.489138, 18.8690024, 19.5098212, 20.2317038, 21.27138675, 22.3121856, 22.9512954, 23.79269667, 24.6706236, 26.1509994, 26.91103475, 28.1305788, 28.8102978, 30.0513306, 30.77065125, 31.9330946, 32.7720166, 34.092839, 35.6099524, 36.5923738, 37.573062, 38.5305922, 39.5112872] 
	
    print_and_plot(peaks)