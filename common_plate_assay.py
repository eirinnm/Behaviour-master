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
def get_args(default_framerate=100, experiment_type="Undefined experiment"):
    global args, FRAMERATE, SCALEFACTOR
    parser = argparse.ArgumentParser(description='Process a deltapixel CSV.')
    parser.add_argument('datafile')
    parser.add_argument('--noled',action='store_true')
    parser.add_argument('--inverttreatment',action='store_true',help='Use when the control treatment is on the top rows')
    parser.add_argument('--framerate',type=int, default=default_framerate)
    parser.add_argument('--scalefactor',type=float, default=0.1227)
    parser.add_argument('--minactivity',type=float, default=0)
    parser.add_argument('--skipframes',type=int, default=0)
    parser.add_argument('--longboutlength',type=float, default=0.5)
    parser.add_argument('--maxlatency',type=int, default=500, help='cutoff (milliseconds) to classify movement as response')
    parser.add_argument('--usedeltapixels',action='store_true',help='Use delta pixels even for XY(H) scripts')
    args=parser.parse_args()
    FRAMERATE=args.framerate
    SCALEFACTOR=args.scalefactor
    print (' %s ' % experiment_type).center(40,'=')
    print '''Framerate = {0.framerate}
Scalefactor = {0.scalefactor}
Ignore LED: {0.noled}
Use delta pixels: {0.usedeltapixels}
Maximum latency = {0.maxlatency}'''.format(args)
    return args
def load_file():
    print ' Datafile '.center(40,'=')
    global datafile, data
    print 'Loading...',
    datafile=args.datafile
    data=np.loadtxt(datafile,dtype=float)
    print len(data),"frames"
    print len(data)/FRAMERATE/60,"minutes"
    return data
#%% Conditions
def load_conditions():
    global datapath, datafilename, conditions, treatment_order, NUM_TRIALS, NUM_WELLS, trialdata, stimname, genotype_order
    datapath, datafilename = os.path.split(datafile)
    datafilename=datafilename.replace('.csv','')
    platedata=pd.read_csv(os.path.join(datapath,'Plate.csv'),index_col=0)
    platedata.columns=platedata.columns.astype(int)
    conditions=platedata.stack().reset_index()
    conditions.columns=['row','col','genotype']
    NUM_WELLS=len(conditions)
    print ' Conditions '.center(40,'=')
    print NUM_WELLS, " wells specified:"
    print platedata
    ## Set the genotype order
    expected_names = ['Unk','Wt','Het','Hom','Mut']
    actual_names = [name.capitalize() for name in conditions.genotype.unique()]
    genotype_order = [name for name in expected_names if name in actual_names]
    genotype_order+= [name for name in actual_names if name not in expected_names]
    print "Genotypes:",genotype_order
    ### Treatments
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
    ### Stimuli
    trialfile = os.path.join(datapath,'Trials.csv')
    if os.path.exists(trialfile):
        trialdata=pd.read_csv(trialfile)
        stimname=trialdata.columns[0]
        trialdata.columns=['stimulus']
        NUM_TRIALS = trialdata.shape[0]
        print NUM_TRIALS, " trials specified. Stimulus name: ", stimname
    else:
        NUM_TRIALS=1
        trialdata=pd.DataFrame({'stimulus':[0]})
        stimname='stimulus'
    print ''.center(40,'=')
    return conditions, treatment_order