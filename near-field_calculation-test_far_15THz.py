#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np;
import torch as T;
import copy ;
import time;
import matplotlib.pyplot as plt;

import Pyccat2;
from Pyccat2 import field_calculation,field_calculation_far;
from Pyccat2 import Make_fitfuc

import Kirchhoffpy;
from Kirchhoffpy.Spheical_field import spheical_grid;
from Kirchhoffpy.coordinate_operations import Coord;



# In[6]:


# define the parameters input files
inputfile='CCAT_model';
sourcefile='beam'
defocus=[0,0,0];
'''
ad=np.genfromtxt('CCAT_model/fitting_error.txt');
ad_m2=ad[0:5*69];
ad_m1=ad[5*69:];
'''
ad_m2=np.zeros(5*69);
ad_m1=np.zeros((5,77));

Ns=501;
source=spheical_grid(-0.001,0.001,-0.001,0.001,Ns,Ns,FIELD='far')
#source=Coord();
#source0=np.genfromtxt(sourcefile+'/beam.txt');
#source.x=source0[...,0];source.y=source0[...,1];source.z=source0[...,2];
Field_s,Field_fimag,Field_m1,Field_m2=field_calculation_far(inputfile,source,defocus,ad_m2,ad_m1);
Ns=int(np.sqrt(source.x.size));
S=(Field_s.real+1j*Field_s.imag).reshape(Ns,Ns);


# In[9]:
fig=plt.figure(figsize=(8,7));
plt.pcolor(source.x.reshape(Ns,Ns),source.y.reshape(Ns,Ns),20*np.log10(np.abs(S)));
plt.xlabel('near-feild beam in amplitude (dB)',fontsize=18,color='darkred')
plt.colorbar();
plt.savefig('output/far_field_beam.png')

fig=plt.figure(figsize=(8,7));
N_fimag=int(np.sqrt(Field_fimag.real.size))
plt.pcolor(20*np.log10(np.abs(Field_fimag.real+1j*Field_fimag.imag).reshape(N_fimag,-1)));
plt.xlabel('near-feild beam in amplitude (dB)',fontsize=18,color='darkred')
plt.colorbar();
plt.savefig('output/IF_field_beam.png')


# In[14]:


# saveing data;
np.savetxt('output/infocus_far/source_field.txt',np.append(S.real,S.imag).reshape(2,-1).T,delimiter=',');
np.savetxt('output/infocus_far/imaginary_field.txt',np.append(Field_fimag.real,Field_fimag.imag).reshape(2,-1).T,delimiter=',');
np.savetxt('output/infocus_far/m1_field.txt',np.append(Field_m1.real,Field_m1.imag).reshape(2,-1).T,delimiter=',');
np.savetxt('output/infocus_far/m2_field.txt',np.append(Field_m2.real,Field_m2.imag).reshape(2,-1).T,delimiter=',');


# In[ ]:




