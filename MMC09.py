#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 12:18:45 2019

@author: tianshu
"""

import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from matplotlib.ticker import MultipleLocator,FormatStrFormatter

path = input('Folderpath: ')
filelist = os.listdir(path)

def peakfind(x):
    diff = x[1][0]
    for n in range(len(x[0])):
        dif =  x[1][n]
        if dif > diff:
            diff = dif
            index = n
    return index

g = globals()
n = 0
y_sum = np.array([0]*1024)
y_sum_1 = np.array([0]*1024)
plt.figure()
for file in filelist:
    
    
    if file[-4:] == '.txt':
        g[file[:-4]]=[]
        f = open(os.path.join(path + "/"+file), 'r')
        lines = f.readlines()
        f.close()
        data_1 = [i for i in lines[2:]]
        data_1 = [i.strip() for i in data_1]
        data_1 = [i.replace('  ', ' ') for i in data_1]
        data_1 = [i.replace('  ', ' ') for i in data_1]
        data_1 = [i.replace('  ', ' ') for i in data_1]
        data_1 = [i.replace('  ', ' ') for i in data_1]
        data_1 = [i.replace('  ', ' ') for i in data_1]
        data_1 = [i.replace('  ', ' ') for i in data_1]
        data_1 = [i.split(' ') for i in data_1]
        x = [float(i[1]) for i in data_1]
        y = [float(i[2]) for i in data_1]
        g[file[:-4]] = [file,[x,y]]
        
        if file[0:6] == 'Sample' and file[8] != 'f':
            n = n + 1
            start = peakfind([x,y])
            baseline = np.average(y[0:start-1])
            baseline_err = np.std(y[0:start-1])/np.sqrt(start-2)
            y_1 = y[start:]+[np.nan]*(1024-len(y[start:]))
            y_sum = y_sum + np.array(y)
            y_sum_1 = y_sum_1 + np.array(y_1)
            x_1 = list(np.array(x[start:])-x[start])+[np.nan]*(1024-len(x[start:]))
            y_ave = y_sum/n
            y_ave_1 = y_sum_1/n
            plt.plot(x,y,'.b', markersize = 0.3)
        sample_1 = [x,y_ave]
        sample = [x_1,y_ave_1]
plt.plot(sample_1[0][0],y_ave[0],'.b', markersize = 0.3, label = 'The data point from 10 measurements')
plt.plot(sample_1[0],y_ave, label = 'The noise reduced singal')
plt.legend()
plt.ylabel('Intensity')
plt.xlabel('Time/s')
plt.xlim(min(sample_1[0]),max(sample_1[0]))
plt.show()
            

def integrate(x):
    y = []
    y_sum = 0
    for n in range(len(x[0])-1):
        y_sum = y_sum + (x[0][n+1]-x[0][n])*x[1][n]
        y.append(y_sum)
    return y

def linear_fit(x,m,b):
    return m*x+b

def propot_fit(x,m):
    return m*x

def peakpick(x,range_l,range_h):
    peak =[]
    for i in range(range_l,range_h):
        if x[i] > x[i+1] and x[i] > x[i-1] and x[i] > 0:
            peak.append(x[i])
    ave = np.average(peak)
    err = np.std(peak)/np.sqrt(len(peak))
    return ave,err,peak
            

# =============================================================================
# Calibration
# =============================================================================
c = [4.22e-3,4.22e-4,2.11e-5]

fig,ax1 = plt.subplots()
ax1.plot(Tempo_1[1][0],np.array(Tempo_1[1][1])*10, lw = 0.8,label ='Original (10 times enlarged)')
data_y2 =  integrate(Tempo_1[1])
ax1.plot(Tempo_1[1][0][:-1],np.array(data_y2)*10, lw = 0.8,label = '1. Integration (10 times enlarged)')
data_y3 = integrate([Tempo_1[1][0],data_y2])
ax1.plot(Tempo_1[1][0][:-1],data_y3, lw = 0.8,label ='2. Integration')
ax1.set_xlabel('Mangetic Feld/G')
plt.ylabel('Intensity')
ax2 = ax1.twiny()
Tempo_2_inte = integrate(Tempo_2[1])
Tempo_3_inte = integrate(Tempo_3[1])
inten = [max(data_y3),max(integrate([Tempo_2[1][0],Tempo_2_inte])),
         max(integrate([Tempo_3[1][0],Tempo_3_inte]))]
ax2.plot(c,inten,'x',label = 'Assigned concentration')
popt_cali,pcov_cali = curve_fit(propot_fit,inten,c)
perr_cali = np.sqrt(np.diag(pcov_cali))
ax2.plot(propot_fit(np.arange(max(inten)),*popt_cali),np.arange(max(inten)),'--k', label = 'Fitting of Calibration')
h1,l1 = ax1.get_legend_handles_labels()
h2,l2 = ax2.get_legend_handles_labels()
h = h1 + h2
l = l1 + l2
plt.legend(h,l,loc = 0)
ax1.set_xlim(min(Tempo_1[1][0][:-1]),max(Tempo_1[1][0][:-1]))
ax2.set_xlim(min(c)*0.9,max(c)*1.1)
ax2.set_xlabel('Concentration/ml mol$^{-1}$')
plt.show()

fig,ax1 = plt.subplots()
#plt.figure('Intensity vs total integral')
#x = [peakpick(data_y2)[0],peakpick(Tempo_2_inte)[0],peakpick(Tempo_3_inte)[0]]
#x_err = [peakpick(data_y2)[1],peakpick(Tempo_2_inte)[1],peakpick(Tempo_3_inte)[1]]
#y = inten

x = []
x_err = []
y = []
peaks = []
for file in filelist:
    if file[0:4] == 'Cali' and file[-4:] == '.txt':
#        inte = integrate(g[file[:-4]][1])
#        baseline = (np.average(g[file[:-4]][1][1][:100])+ np.average(g[file[:-4]][1][1][-100:]))/2
        y_1 = np.array(g[file[:-4]][1][1])
        inte = integrate([g[file[:-4]][1][0],y_1])
        x1 = g[file[:-4]][1][0]
        popt_fix,pcov_fix = curve_fit(linear_fit,x1[:100]+x1[-100:],inte[:100]+inte[-100:])
        y_2 = np.array(inte-linear_fit(np.array(x1),*popt_fix)[:-1])
        x.append(peakpick(y_2,498,507)[0])
        x_err.append(peakpick(y_2,498,507)[1])
        peaks.append(peakpick(y_2,498,507)[2])
        y.append(max(integrate([g[file[:-4]][1][0][:-1],y_2])))
        if len(file) == 8:
            print(file)
            ax2 = ax1.twiny()
            ax2.plot(g[file[:-4]][1][0],g[file[:-4]][1][1], label = 'EPR for the actived sample')
            ax2.plot(g[file[:-4]][1][0][:-1],y_2, label = '1. Integration of sample')   
            ax2.set_xlim(min(g[file[:-4]][1][0][:-1]),max(g[file[:-4]][1][0][:-1]))
        
ax1.errorbar(y,x,yerr = x_err, fmt = 'x', ecolor = 'r', label = 'Assigned single peak intensity')
popt_int,pcov_int = curve_fit(propot_fit,x,y)
perr_int = np.sqrt(np.diag(pcov_int))
ax1.plot(propot_fit(np.arange(min(x)*0.9,max(x)*1.1,0.01),*popt_int),np.arange(min(x)*0.9,max(x)*1.1,0.01),'--k',label = 'Fitting for the single peak intensity')
ax1.set_ylabel('Single peak intensity')
ax1.set_xlim(min(y)*0.9,max(y)*1.1)

ax2.set_xlabel('Mangetic Feld/G')
ax1.set_xlabel('Total intensity')
h1,l1 = ax1.get_legend_handles_labels()
h2,l2 = ax2.get_legend_handles_labels()
h = h1 + h2
l = l1 + l2
plt.legend(h,l,loc = 0)
plt.show()

# =============================================================================
# concentration time profile
# =============================================================================
plt.figure()
#start = peakfind([x,y])
#inten = np.array(sample[1][start:])
inten = np.array(sample[1])
con = propot_fit(propot_fit(inten,*popt_int),*popt_cali)
#time = np.array(sample[0][start:])-sample[0][start]
time = sample[0]
plt.plot(time, con,'.')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlim(min(time),max(time))
plt.ylabel('Concentration/ml mol$^{-1}$')
plt.xlabel('Time/s')
plt.show()

# =============================================================================
# log(c_0/c_t-1) vs log(t) determination of a_l,a_s, and i
# =============================================================================
x = []
y = []
for n in range(len(con)):
    if con[n] > 0 and con[n] < con[0]:    
        x.append(np.log(time[n]))
        y.append(np.log(con[0]/con[n]-1))

plt.figure('Determination of a_l,a_s, and i')
plt.plot(x,y,'.')
x_s = []
y_s = []
for i in range(50):
    if y[i] > -2.7:
        x_s.append(x[i])
        y_s.append(y[i])     
popt_s,pcov_s = curve_fit(linear_fit,x_s,y_s)
perr_s = np.sqrt(np.diag(pcov_s))
plt.plot(np.arange(min(x),min(x)/30,0.1),linear_fit(np.arange(min(x),min(x)/30,0.1),*popt_s),label = 'Fitting for $i < i_c$')
popt_l,pcov_l = curve_fit(linear_fit,x[-400:],y[-400:])
perr_l = np.sqrt(np.diag(pcov_l))
plt.plot(np.arange(min(x)/5,max(x)*1.5,0.1),linear_fit(np.arange(min(x)/5,max(x)*1.5,0.1),popt_l[0]*2,popt_l[1]),label = 'Fitting for $i > i_c$')
plt.legend()
plt.xlabel('ln($t$)')
plt.ylabel('ln($c_R(0)/c_R(t)$-1)')
plt.show()
popt_l = np.array([popt_l[0]*2,popt_l[1]])
t_i = fsolve(lambda x : linear_fit(x,*popt_l) - linear_fit(x,*popt_s),0)
i = 2.51e6*np.exp(-21e3/8.314/294)*0.88/254.414*1e3*np.exp(t_i)
print(i)
# =============================================================================
# 
# =============================================================================
def k0(x):
    return np.log10(1.18e11*np.e**(-19400/294/8.314)*i**(popt_s[0]-1+1-popt_l[0]))-(1-popt_l[0])*np.log10(x)

def k(x):
    if x < i:
        return np.log10(1.18e11*np.e**(-19400/294/8.314))-(1-popt_s[0])*np.log10(x)
    else:
        return k0(x)
x = []
y = []
y2 =[]
for n in range(1,10000):
    x.append(np.log10(n))
    y.append(k(n))
    y2.append(k0(n))
plt.figure()
plt.plot(x,y2,'--',label = 'Hypothetical $k_t$ following the long chain termination')
plt.plot(x,y,label = '$k_t$ development over chain length')
plt.xlim(1,4)
plt.xlabel('log($i$)')
plt.ylabel('log($k_t$)')
plt.legend()
plt.show()
# =============================================================================
# 
# =============================================================================
filename = input('Filename: ') #typ in the file name in the cmd or terminal
f = open(filename, 'r')
lines = f.readlines()
f.close()
data = [i.replace("\n","") for i in lines[1:]]
data = [i.split(",") for i in data]
data_x1 = [float(i[0]) for i in data]
data_y1 = [float(i[1]) for i in data]
data_x2 = [float(i[4])  for i in data if i[4] != '']
data_y2 = [float(i[5]) for i in data if i[4] != '' ]

plt.figure()
plt.plot(data_x2,data_y2,'.',markersize = 1,label = 'Experimental values')
plt.plot(data_x1,data_y1,label = 'Simulation')
plt.xlabel('Time/s')
plt.ylabel('Concentration/ml mol$^{-1}$')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend()
plt.xlim(0,max(data_x1))
plt.ylim(min(data_y1)*0.9,1.1*max(data_y1))
plt.show()

# =============================================================================
# 
# =============================================================================
# =============================================================================
# 
# =============================================================================
filename = input('Filename: ') #typ in the file name in the cmd or terminal
f = open(filename, 'r')
lines = f.readlines()
f.close()
data = [i.replace("\n","") for i in lines[1:]]
data = [i.split(",") for i in data]
data_x1 = [float(i[0]) for i in data]
data_y1 = [float(i[1]) for i in data]
plt.figure()
plt.plot(data_x1,data_y1,label = 'Simulation')
plt.xlabel('Time/s')
plt.ylabel('$k_t$')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend()
plt.xlim(0,max(data_x1))
plt.ylim(min(data_y1)*0.9,1.1*max(data_y1))
plt.show()

