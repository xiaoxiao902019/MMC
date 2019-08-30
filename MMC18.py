# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 13:11:43 2019

@author: xiaox
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filename1 =input('Filename: ')
f = open(filename1,'r')
lines = f.readlines()
data_series_1 = [i for i in lines[2:]]
data_1 = [i.replace(',', '.') for i in data_series_1]
data_1 = [i.replace('  ', ' ') for i in data_1]
data_1 = [i.replace('  ', ' ') for i in data_1]
data_1 = [i.replace('  ', ' ') for i in data_1]
data_1 = [i.split('\t') for i in data_1]
f.close()


linear_sample = [[],[]]
for i in range(50,61):
    linear_sample[0].append(float(data_1[i][4]))
    linear_sample[1].append(float(data_1[i][7]))

star_sample = [[],[]]
for i in range(61,72):
    star_sample[0].append(float(data_1[i][4]))
    star_sample[1].append(float(data_1[i][7]))
    
plt.figure(1)
plt.plot(linear_sample[0],linear_sample[1],'-x')
plt.ylabel('Size/nm')
plt.xlabel('Temperature/°C')
plt.show()

plt.figure(2)
plt.plot(star_sample[0],star_sample[1],'-x')
plt.ylabel('Size/nm')
plt.xlabel('Temperature/°C')
plt.show()

filename2 =input('Filename: ')
f = pd.read_excel(filename2,sheet_name=None)
lines = list(f.items())
profil=[]
for i in lines:
    name = i[0]
    data_2 = i[1].values.tolist()
    intensity_x = [n[0] for n in data_2]
    intensity_y = [n[1] for n in data_2]
    for n in range(2,len(data_2[0])):
        print(n)
        if np.isnan(data_2[0][n]) == False:
            print(data_2[0][n])
            m = n
            break
    correction_x = [n[m] for n in data_2]
    correction_y = [n[m+1] for n in data_2]
    profil.append([name,intensity_x,intensity_y,correction_x,correction_y])

plt.figure(3)
plt.plot(profil[0][1],profil[0][2],label = 'Folded linear molecules at 29 °C')
plt.plot(profil[1][1],profil[1][2],label = 'Unfolded linear molecules at 37 °C')
plt.ylabel('Distribution')
plt.xlabel('Size/nm')
plt.xlim(0,600)
plt.legend()
plt.show()

plt.figure(4)
plt.plot(profil[2][1],profil[2][2],label = 'Folded star-formed molecules at 22 °C')
plt.plot(profil[3][1],profil[3][2],label = 'Unfolded star-formed molecules at 32 °C')
plt.ylabel('Distribution')
plt.xlabel('Size/nm')
plt.xlim(0,1500)
plt.legend()
plt.show()


fig,ax = plt.subplots()
ax.plot(profil[0][3],profil[0][4],label = 'Folded linear molecules at 29 °C')
ax.plot(profil[1][3],profil[1][4],label = 'Unfolded linear molecules at 37 °C')
plt.ylabel('Correlation function')
ax.set_xscale('log')
plt.xlabel('Time')
plt.legend()
plt.show()

fig,ax = plt.subplots()
plt.plot(profil[2][3],profil[2][4],label = 'Folded star-formed molecules at 22 °C')
plt.plot(profil[3][3],profil[3][4],label = 'Unfolded star-formed molecules at 32 °C')
plt.ylabel('Correlation function')
ax.set_xscale('log')
plt.xlabel('Time')
plt.legend()
plt.show()










