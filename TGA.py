# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 09:00:18 2018

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt

filename =input('Filename: ')
f = open(filename,'r')
lines = f.readlines()
data_series_1 = [i for i in lines[31:-1]]
data_1 = [i.strip() for i in data_series_1]
data_1 = [i.replace(',', '.') for i in data_1]
data_1 = [i.split(';') for i in data_1]
f.close()
#print (data_1)

fig = plt.figure(figsize = (4,4))
ax1 = fig.add_subplot(111)

data_x1 = []
data_y1 = []
data_x2 = []
data_y2 = []
data_x3 = []
data_y3 = []
ax2 = ax1.twinx()
data_y = []

x = []
Temp = []
for n in range(len(data_1)):
    if n < len(data_1)-1:
        data_y.append((float(data_1[n+1][2])-float(data_1[n][2]))/(float(data_1[n+1][1])-float(data_1[n][1])))
        if float(data_1[n][1]) < 52:
            data_x1.append(float(data_1[n][1]))
            data_y1.append(float(data_1[n][2]))
        elif float(data_1[n][1]) > 132:
            data_x3.append(float(data_1[n][1]))
            data_y3.append(float(data_1[n][2]))
        elif float(data_1[n][1]) < 132 or float(data_1[n][1]) > 52:
            data_x2.append(float(data_1[n][1]))
            data_y2.append(float(data_1[n][2]))
    
    x.append(float(data_1[n][1]))
    Temp.append(float(data_1[n][0]))
    
ax2.plot(x[:-1],data_y,'.',markersize = 1,color = 'c',label = 'Differential')
ax1.plot(data_x1,data_y1,'r',label = ' ')
ax1.plot(data_x2,data_y2,'m',label = 'Mass lost')
ax1.plot(data_x3,data_y3,'g',label = ' ')



ax3 = ax1.twinx()
ax3 = plt.gca()
pos1 = ax1.get_position()
pos2 = [pos1.x0-0.05,pos1.y0,pos1.x1/1.5,pos1.y1/1.1]
ax3.plot(x,Temp, linewidth = 0.5,label = 'Temperature')
#ax1.set_xlim(0,180)
ax1.set_xlabel('Time/min')
ax1.set_ylabel('Masslost/mg')
ax2.set_ylabel('Differential')
ax1.set_xlim(0,180)
ax3.set_position(pos2)
ax3.spines['right'].set_position(('data',200))
ax3.set_ylabel('Temperature/°C')
h1,l1=ax1.get_legend_handles_labels()
h2,l2=ax2.get_legend_handles_labels()
h3,l3=ax3.get_legend_handles_labels()
h = h1 + h2 +h3
l = l1 + l2 +l3
plt.legend(h,l,loc = 'center right')
