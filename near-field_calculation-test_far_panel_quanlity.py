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
#S_init=np.genfromtxt('CCAT_model/error.txt')
#S_init=np.genfromtxt('CCAT_model/adjusters_error.txt')
#S_init=np.genfromtxt('CCAT_model/error_no_adjusters0.txt');
#ad_m2=S_init[0:5*69]
#ad_m1=S_init[5*69:]
ad_m2=np.zeros(5*69);
ad_m1=np.zeros(5*77);
m2_panel='MeasedSurf/M2_error_profile.txt'
m1_panel='MeasedSurf/M1_error_profile.txt'
#m2_panel=None
#m1_panel=None

angle=0.001
Ns=501
u0=defocus[0]/(14400+defocus[2]);
v0=-defocus[1]/(14400+defocus[2]);
source=spheical_grid(-angle+u0,angle+u0,-angle+v0,angle+v0,Ns,Ns,FIELD='far')
#source=Coord();
#source0=np.genfromtxt(sourcefile+'/beam.txt');
#source.x=source0[...,0];source.y=source0[...,1];source.z=source0[...,2];
Field_s,Field_fimag,Field_m1,Field_m2=field_calculation_far(inputfile,source,defocus,ad_m2,ad_m1,m2_panel=m2_panel,m1_panel=m1_panel);
#Ns=int(np.sqrt(source.x.size));
S=(Field_s.real+1j*Field_s.imag).reshape(Ns,Ns);


# In[9]:


fig=plt.figure(figsize=(8,7));
plt.pcolor(source.x.reshape(Ns,Ns),source.y.reshape(Ns,Ns),20*np.log10(np.abs(S)));
plt.xlabel('near-feild beam in amplitude (dB)',fontsize=18,color='darkred')
plt.colorbar();
plt.savefig('output/perror/far_field_beam.png')

fig=plt.figure(figsize=(8,7));
N_fimag=int(np.sqrt(Field_fimag.real.size))
plt.pcolor(20*np.log10(np.abs(Field_fimag.real+1j*Field_fimag.imag).reshape(N_fimag,-1)));
plt.xlabel('near-feild beam in amplitude (dB)',fontsize=18,color='darkred')
plt.colorbar();
plt.savefig('output/perror/IF_field_beam.png')

# In[14]:


# saveing data;
#np.savetxt('output/near_field3/source_field.txt',np.append(S.real,S.imag).reshape(2,-1).T,delimiter=',');
#np.savetxt('output/near_field3/imaginary_field.txt',np.append(Field_fimag.real,Field_fimag.imag).reshape(2,-1).T,delimiter=',');
#np.savetxt('output/near_field3/m1_field.txt',np.append(Field_m1.real,Field_m1.imag).reshape(2,-1).T,delimiter=',');
#np.savetxt('output/near_field3/m2_field.txt',np.append(Field_m2.real,Field_m2.imag).reshape(2,-1).T,delimiter=',');

#np.savetxt('output/panel_quality/gain_after_fitting_900GHz.txt',np.append(S.real,S.imag).reshape(2,-1).T,delimiter=',');
#np.savetxt('output/panel_quality/gain_after_fitting_900GHz_(200,0).txt',np.append(S.real,S.imag).reshape(2,-1).T,delimiter=',');
#np.savetxt('output/panel_quality/gain_before_fitting_900GHz_(200,0).txt',np.append(S.real,S.imag).reshape(2,-1).T,delimiter=',');

#np.savetxt('output/panel_quality/gain_900GHz_(200,0).txt',np.append(S.real,S.imag).reshape(2,-1).T,delimiter=',');
#np.savetxt('output/panel_quality/gain_after_fitting_900GHz_(200,0).txt',np.append(S.real,S.imag).reshape(2,-1).T,delimiter=',');
#np.savetxt('output/panel_quality/gain_before_fitting_900GHz_(200,0).txt',np.append(S.real,S.imag).reshape(2,-1).T,delimiter=',');

#np.savetxt('output/panel_quality/gainloss/gain_900GHz.txt',np.append(S.real,S.imag).reshape(2,-1).T,delimiter=',');
np.savetxt('output/panel_error_effect1500GHz.txt',np.append(S.real,S.imag).reshape(2,-1).T,delimiter=',');

#np.savetxt('output/panel_quality/gainloss/gain_before_fitting_900GHz_only_panelerror.txt',np.append(S.real,S.imag).reshape(2,-1).T,delimiter=',');
#np.savetxt('output/panel_quality/gain_before_fitting_900GHz.txt',np.append(S.real,S.imag).reshape(2,-1).T,delimiter=',');



#np.savetxt('output/panel_quality/gain_820GHz_20um.txt',np.append(S.real,S.imag).reshape(2,-1).T,delimiter=',');
#np.savetxt('output/panel_quality/gain_820GHz_10um.txt',np.append(S.real,S.imag).reshape(2,-1).T,delimiter=',');
#np.savetxt('output/panel_quality/gain_820GHz_0um.txt',np.append(S.real,S.imag).reshape(2,-1).T,delimiter=',');

