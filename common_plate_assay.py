# -*- coding: utf-8 -*-
"""
Created on Wed Feb 08 21:18:24 2017

@author: Eirinn
"""
from __future__ import division
import os.path, argparse
import numpy as np
import pandas as pd
pd.set_option('display.width',120)
parser = argparse.ArgumentParser(description='Process a deltapixel CSV.')
parser.add_argument('datafile')
parser.add_argument('--noled',action='store_true')
parser.add_argument('--inverttreatment',action='store_true',help='Use when the control treatment is on the top rows')
parser.add_argument('--framerate',type=int, default=100)
parser.add_argument('--scalefactor',type=float, default=0.1227)
parser.add_argument('--minactivity',type=float, default=0)
parser.add_argument('--longboutlength',type=float, default=0.5)
args=parser.parse_args()
FRAMERATE=args.framerate
SCALEFACTOR=args.scalefactor
print ' Long term tracking '.center(40,'=')
print '''Framerate = {0.framerate}
Scalefactor = {0.scalefactor}
Ignore LED: {0.noled}'''.format(args)
print ''.center(40,'=')
print 'Loading...',
datafile=args.datafile
data=np.loadtxt(datafile,dtype=float)
print len(data),"frames"
print len(data)/FRAMERATE/60,"minutes"
#%% Conditions
datapath, datafilename = os.path.split(datafile)
datafilename=datafile.replace('.csv','')
platedata=pd.read_csv(os.path.join(datapath,'Plate.csv'),index_col=0)
platedata.columns=platedata.columns.astype(int)
conditions=platedata.stack().reset_index()
conditions.columns=['row','col','genotype']
NUM_WELLS=len(conditions)
print ' Conditions '.center(40,'=')
print NUM_WELLS, " wells specified:"
print platedata
treatmentfile = os.path.join(datapath,'Treatment.csv')
if os.path.exists(treatmentfile):
    treatments = pd.read_csv(treatmentfile,index_col=0)
    print "Treatments:"
    print treatments
    treatments=treatments.stack().reset_index()
    treatments.columns=['row','col','treatment']
    conditions['treatment'] = treatments.treatment
else:
    conditions['treatment'] = 'Control'
if args.inverttreatment:
    treatment_order = conditions.treatment.unique()
else:
    treatment_order = conditions.treatment.unique()[::-1]
print ''.center(40,'=')