# -*- coding: utf-8 -*-
"""
Created on Wed Feb 08 21:18:24 2017

@author: Eirinn
"""
from __future__ import division
import sys, os.path
print sys.argv
if len(sys.argv)==1:
    raise NameError, "No input file provided"

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#import scipy.signal
#%matplotlib inline
from pylab import rcParams
rcParams['figure.figsize'] = 10, 5
sns.set(context='talk',style='darkgrid',palette='deep',rc={'figure.facecolor':'white'})


datafile = sys.argv[1]
#load the file
data=np.loadtxt(datafile,dtype=float)
datapath, datafile = os.path.split(datafile)
datafile=datafile.replace('.csv','')


## Set experiment conditions
#import itertools
#NUM_WELLS=24
#NUMROWS=4
#NUMCOLS=6
#conditions=pd.DataFrame(list(itertools.product('ABCDEFGH'[:NUMROWS],range(1,NUMCOLS+1))), columns=['row','col'])
#conditions.loc[conditions.col.between(1,3),'genotype']='wt'
#conditions.loc[conditions.col.between(4,6),'genotype']='zwp'
platedata=pd.read_csv(os.path.join(datapath,'Plate.csv'),index_col=0)
conditions=platedata.stack().reset_index()
conditions.columns=['row','col','genotype']
NUM_WELLS=len(conditions)
print NUM_WELLS, " wells specified:"
print platedata

trialdata=pd.read_csv(os.path.join(datapath,'Trials.csv'))
stimname=trialdata.columns[0]
trialdata.columns=['stimulus']
NUM_TRIALS = trialdata.shape[0]
print NUM_TRIALS, " trials specified. Stimulus name: ", stimname
FRAMERATE=500
SCALEFACTOR=16.2/132 #mm / pixel. 16.2/66 is for a 24-well plate at 640x480 resolution
SKIP_FRAMES = 0 #number of frames to skip at the start of each trial

#main function for finding swim bouts in a given set of values
MIN_BOUT_LENGTH = 5 #frames
MIN_BOUT_GAP = 2 #frames

def get_bouts(xy): #returns stats for each bout in this well
    bouts=xy[:,0].nonzero()[0]
    if len(bouts)>0:
        start_gaps=np.ediff1d(bouts,to_begin=99)
        end_gaps=np.ediff1d(bouts,to_end=99)
        breakpoints=np.vstack((bouts[np.where(start_gaps>MIN_BOUT_GAP)],
                               bouts[np.where(end_gaps>MIN_BOUT_GAP)])) #two columns, start and end frame
        boutlengths=np.diff(breakpoints,axis=0)[0]
        #select only bouts longer than a minimum
        breakpoints=breakpoints[:,boutlengths>MIN_BOUT_LENGTH]
        boutlengths=boutlengths[boutlengths>MIN_BOUT_LENGTH]
        #intensities=np.array([np.sum(delta_pixels[start:end]) for start, end in breakpoints.T])
        return boutlengths, breakpoints[0]#, intensities
    else:
        return [],[]

##reshape the data to pair the XY coords
data=np.dstack((data[:,::2],data[:,1::2]))
#def process2(data,num_trials):
trials=np.array_split(data,NUM_TRIALS)
#POST_FRAMES=250
df=[]
response1=[]
response2=[]
rdf=[]
trialframecounter=0
tdf=[]
bdf=[]
maxmovement=np.array(0)
for t,trial in enumerate(trials):
    #find the LED flash, if there is one. It will always be in well 0.
    pulseframes = np.nonzero(trial[SKIP_FRAMES:,0,0])[0]
    flash1 = pulseframes[0] if len(pulseframes) else np.nan
    flash2 = pulseframes[pulseframes>=flash1+20]
    flash2 = flash2[0] if flash2 else np.nan
    tdf.append({'trialstart':trialframecounter,'flash1':flash1,'flash2':flash2,
                    'flash1_abs':trialframecounter+flash1+SKIP_FRAMES})
    trialframecounter = trialframecounter+len(trial)
    #now find the swimming bouts
    for well in range(NUM_WELLS):
        thismovement=trial[SKIP_FRAMES:,well+1] #well+1 because the LED is well 0
        if np.count_nonzero(thismovement)>np.count_nonzero(maxmovement):
            maxmovement=thismovement
            print t, well
        for boutid,(boutlength, startframe) in enumerate(zip(*get_bouts(thismovement))):
            bdf.append({'boutid':boutid,'trial':t,'fish':well,
                        'boutlength_raw':boutlength,
                        'boutlength':boutlength/FRAMERATE,
                        'startframe':startframe,})
                        #'endframe':startframe+boutlength})
tdf=pd.DataFrame(tdf,dtype=int)
bdf=pd.DataFrame(bdf)
meanflash=tdf.flash1.median()
print "Median flash frame was",meanflash,"+-", tdf.flash1.std()
print len(tdf[tdf.flash1.isnull()])," trials had no flash and were given the median value:"
print tdf[tdf.flash1.isnull()].index
tdf.loc[tdf.flash1.isnull(),'flash1']=meanflash
tdf=pd.merge(tdf,trialdata,left_index=True,right_index=True)


#the 'startframe' values for each bout need to be offset with the LED flash value for that trial.
bdf['vid_startframe']=bdf.apply(lambda x: tdf.loc[x.trial].trialstart+x.startframe+SKIP_FRAMES,axis=1)
bdf['startframe']=bdf.apply(lambda x: x.startframe-tdf.loc[x.trial].flash1,axis=1)
bdf['latency']=bdf.startframe/FRAMERATE


##Classify responses
##Make a dataframe of the first bout in each trial
rdf=bdf.groupby(['fish','trial'],as_index=False).agg({'latency':np.min,'boutlength':np.sum})
rdf=pd.merge(rdf,tdf,left_on='trial',right_index=True)
rdf=pd.merge(rdf, conditions, left_on='fish',right_index=True)
rdf.loc[rdf.latency<0,'cat']='already_moving'
rdf.loc[rdf.latency>=0,'cat']='responded'
#print "Dropping", len(rdf[rdf.startframe<=0]), "responses that occured before the LED flash"
#rdf=rdf[rdf.latency>0]
#print len(rdf), "responses remain"

responses = rdf[rdf.cat=='responded'].groupby(['genotype','trial','stimulus']).size().reset_index(name='num_responses')
responses = pd.merge(responses,tdf.groupby('stimulus').size().reset_index(name='num_trials'))
responses = pd.merge(responses,conditions.groupby('genotype').size().reset_index(name='num_fish'))
responses['mean_response']=responses.num_responses/responses.num_fish

## start plotting things
## response rate

sns.barplot(data=responses,y='mean_response', x='stimulus', hue='genotype' )
plt.title("Number of responses in each group, mean across trials")
plt.ylabel("Percent of fish that responded")
plt.xlabel("Stimulus (%s)" % stimname)
plt.text(0.5, 1.08,datafile,
     horizontalalignment='center',
     verticalalignment='top',
     transform = plt.gca().transAxes,
        fontsize=10)
plt.savefig(os.path.join(datapath, datafile+".responserate.png"))
plt.show()

##plot all the latencies
for name, group in rdf.groupby('stimulus'):
    sns.kdeplot(data=group.latency,label=name)
plt.title("Distribution of latencies")
plt.xlabel("Time until first movement (seconds)")
plt.savefig(os.path.join(datapath, datafile+".latencydist.png"))
plt.show()

##plot all the latencies - excluding already_moving
for name, group in rdf[rdf.cat=='responded'].groupby('stimulus'):
    sns.kdeplot(data=group.latency,label=name,clip=(0,0.4))
plt.title("Distribution of latencies - responders only")
plt.xlabel("Time until first movement (seconds)")
plt.savefig(os.path.join(datapath, datafile+".latencydist-responders.png"))
plt.show()

#make a bar graph of latencies
sns.stripplot(data=rdf[rdf.cat=='responded'],y='latency', hue='genotype',x='stimulus',jitter=True)
plt.title("Latency of first bout")
plt.ylabel("Mean latency per bout (seconds)")
plt.xlabel("Stimulus (%s)" % stimname)
plt.savefig(os.path.join(datapath, datafile+".latencybar.png"))
plt.show()
