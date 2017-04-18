# -*- coding: utf-8 -*-
"""
Created on Wed Feb 08 21:18:24 2017

@author: Eirinn
"""
from __future__ import division
import os.path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#import scipy.signal
#%matplotlib inline
from pylab import rcParams
rcParams['figure.figsize'] = 5, 5
sns.set(context='talk',style='darkgrid',palette='deep',rc={'figure.facecolor':'white'})

DEFAULT_FRAMERATE=500
import common_plate_assay as cpa
args=cpa.get_args(DEFAULT_FRAMERATE,'Startle Analysis')
data=cpa.load_file()
conditions, treatment_order = cpa.load_conditions()
datafilename = cpa.datafilename
datapath = cpa.datapath
NUM_WELLS = cpa.NUM_WELLS
NUM_TRIALS = cpa.NUM_TRIALS
FRAMERATE = cpa.FRAMERATE
USEDELTAPIXELS = cpa.args.usedeltapixels
SKIP_FRAMES = cpa.args.skipframes #number of frames to skip at the start of each trial
trialdata = cpa.trialdata
stimname = cpa.stimname
MAX_LATENCY = cpa.args.maxlatency #milliseconds. Movement after this time is not a response to the stimulus
#%%
#main function for finding swim bouts in a given set of values
MIN_BOUT_LENGTH = 5 #frames
MIN_BOUT_GAP = 2 #frames

def get_bouts(movementframes): #returns stats for each bout in this well
    #bouts=xy[:,0].nonzero()[0]
    bouts=movementframes.nonzero()[0]
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

if not USEDELTAPIXELS:
    ##reshape the data to pair the XY coords
    data=np.dstack((data[:,::2],data[:,1::2]))
trials=np.array_split(data,NUM_TRIALS)
#POST_FRAMES=250
df=[]
response1=[]
response2=[]
rdf=[]
trialframecounter=0
tdf=[]
bdf=[]
for t,trial in enumerate(trials):
    #find the LED flash, if there is one. It will always be in well 0.
    if USEDELTAPIXELS:
        pulseframes = np.nonzero(trial[SKIP_FRAMES:,0])[0]
    else:
        pulseframes = np.nonzero(trial[SKIP_FRAMES:,0,0])[0]
    flash1 = pulseframes[0] if len(pulseframes) else np.nan
    flash2 = pulseframes[pulseframes>=flash1+20]
    flash2 = flash2[0] if flash2 else np.nan
    tdf.append({'trialstart':trialframecounter,'flash1':flash1,'flash2':flash2,
                    'flash1_abs':trialframecounter+flash1+SKIP_FRAMES})
    trialframecounter = trialframecounter+len(trial)
    #now find the swimming bouts
    for well in range(NUM_WELLS):
        if USEDELTAPIXELS:
            thismovement=trial[SKIP_FRAMES:,well+1] #well+1 because the LED is well 0
        else:
            thismovement=trial[SKIP_FRAMES:,well+1,0] #just use the X movements
        for boutid,(boutlength, startframe) in enumerate(zip(*get_bouts(thismovement))):
            bdf.append({'boutid':boutid,'trial':t,'fish':well,
                        'boutlength_raw':boutlength,
                        'boutlength':boutlength/FRAMERATE,
                        'startframe':startframe})
                        #'endframe':startframe+boutlength})
tdf=pd.DataFrame(tdf,dtype=int)
bdf=pd.DataFrame(bdf)

## Prepare trial data
meanflash=tdf.flash1.median()
print "Median flash frame was",meanflash,"+-", tdf.flash1.std()
missing_flash_trials = tdf[tdf.flash1.isnull()]
if len(missing_flash_trials)>0:
    print len(missing_flash_trials)," trials had no flash and were given the median value:"
    print tdf[tdf.flash1.isnull()].index
    tdf.loc[tdf.flash1.isnull(),'flash1']=meanflash
else:
    print "Flash found in all",NUM_TRIALS, "trials."
tdf=pd.merge(tdf,trialdata,left_index=True,right_index=True)

#the 'startframe' values for each bout need to be offset with the LED flash value for that trial.
bdf['vid_startframe']=bdf.apply(lambda x: tdf.loc[x.trial].trialstart+x.startframe+SKIP_FRAMES,axis=1)
bdf['startframe']=bdf.apply(lambda x: x.startframe-tdf.loc[x.trial].flash1,axis=1)
bdf['latency']=bdf.startframe/FRAMERATE*1000
## make the fish and trials a category, so missing fish/trials will show up
bdf.fish = bdf.fish.astype("category", categories = np.arange(NUM_WELLS))
bdf.trial = bdf.trial.astype("category", categories = np.arange(NUM_TRIALS))
#%%
##Classify responses
##Make a blank dataframe for each fish/trial combination
from itertools import product
df = pd.DataFrame([{'trial':t,'fish':f} for f,t in product(bdf.fish.cat.categories,bdf.trial.cat.categories)])
## group responses for each fish/trial, getting the first latency and longest bout
rdf=bdf.groupby(['fish','trial'],as_index=False).agg({'latency':np.min,'boutlength':np.max})
## put this back into the blank dataframe
df=pd.merge(df,rdf,left_on=['fish','trial'],right_on=['fish','trial'],how='outer')
df=pd.merge(df,tdf,left_on='trial',right_index=True) ## add the trial conditions
df=pd.merge(df, conditions, left_on='fish',right_index=True) ## add the well (fish) conditions
df['responded']=False
df.loc[(df.latency>=0) & (df.latency<=MAX_LATENCY),'responded']=True
#drop fish with genotype 'x'
df=df[df.genotype<>'x']
#df.loc[df.latency<0,'cat']='already_moving'
#df.loc[df.latency>=0,'cat']='responded'
#df.loc[df.latency.isnull(),'cat']='no_response'
#%% Plot the response per well, ignoring stimuli conditions
fishmeans = df.groupby(['row','col','fish','genotype','treatment','stimulus']).agg({'boutlength':np.mean, 'responded': np.mean})
fishmeanlatency = df[df.responded].groupby(['row','col','fish','genotype','treatment','stimulus']).latency.mean()
fishmeans['latency']=fishmeanlatency
fishmeans=fishmeans.reset_index()

fig,axes=plt.subplots(1,2, sharex=True, sharey=True, figsize=(14,4))
#plt.tight_layout()
annot_kws={"size": 10}
ax=axes[0]
## this will probably break with more than one stimuli; need to take an average before making the heatmap
sns.heatmap(ax=ax,data=fishmeans.pivot('row','col','responded'),annot=True,cbar=False,square=True,annot_kws=annot_kws)
ax.set_title('Response rate')
ax=axes[1]
sns.heatmap(ax=ax,data=fishmeans.pivot('row','col','latency'),annot=True,cbar=False,square=True,annot_kws=annot_kws,fmt=".1f")
ax.set_title('Mean latency (ms)')
#plt.axis('off')
plt.suptitle('Behaviour per well')
plt.savefig(os.path.join(datapath, datafilename+"_plateview.png"))
#%%
#g = sns.FacetGrid(data=fishmeans,row='stimulus', aspect=2, size=5)
#g.map_dataframe(lambda data,color: sns.heatmap(data.pivot('row','col','responded'),
#                                               annot=True,cbar=False,square=True,annot_kws=annot_kws ))
#%% ================= Per-trial response rate =================
## What percentage of fish responded to each stimuli?
trialmeans = df.groupby(['genotype','treatment','stimulus','trial']).agg({'responded': np.mean,
                                                                            'latency': np.mean}).reset_index()
g=sns.factorplot(data=trialmeans,y='responded',x='stimulus',hue='genotype',col='treatment',aspect=0.75,capsize=.1,size=5)
g.set_ylabels('Fraction of fish')
g.set(ylim=(0,1))
#plt.ylim(0,1)
plt.subplots_adjust(top=0.85)
plt.suptitle('Responses per stimulus and treatment')
g.savefig(os.path.join(datapath, datafilename+"_pct_perstimulus.png"))
#%%
g=sns.factorplot(data=trialmeans,y='responded',x='trial',hue='genotype',row='treatment',col='stimulus',aspect=2,size=4)
g.set(ylim=(0,1))
plt.subplots_adjust(top=0.85)
plt.suptitle('Response fraction per trial')
g.savefig(os.path.join(datapath, datafilename+"_pertrial.png"))
#%% ================= Per-fish response rate =================
## Generate per-fish response rates rather than per-trial

## make a facetgrid
g = sns.FacetGrid(data=fishmeans, hue='genotype',col='stimulus',aspect=1, ylim=(0,1),size=5)
g.map(sns.violinplot,'treatment','responded', cut=0, bw=0.2)
g.map(sns.swarmplot,'treatment','responded',color='#333333')
g.set_ylabels('Rate (fraction of trials)')
plt.subplots_adjust(top=0.85)
plt.suptitle('Response rate per stimulus condition')
#plt.ylabel('Fraction of trials')
g.savefig(os.path.join(datapath, datafilename+"_rate_perstimulus.png"))
#%% ================= Latencies =================
## What was the mean latency for all responses? (not fish means)
g = sns.factorplot(data=df[df.responded], kind='box', x='latency',y='treatment',hue='genotype', row='stimulus',aspect=2, size=5,
                   showfliers=False, notch=True,margin_titles=True)
#g.set(xlim=(0,30))
#g.set(xticks=np.arange(0,30,2))
g.map(sns.swarmplot,'latency','treatment',  color='#333333')
#g.map_dataframe(lambda data, color: sns.stripplot(data=data))
plt.subplots_adjust(top=0.85)
plt.suptitle('Latency of responses (ms)')
g.savefig(os.path.join(datapath, datafilename+"_latency_box.png"))

#%% Total distribution
CUTOFF = 100
g = sns.FacetGrid(data=df[df.responded & (df.latency<=CUTOFF)], hue='genotype', row='stimulus',aspect=1.5, size=5)
#g = sns.factorplot(data=df[df.responded & (df.latency<=CUTOFF)], hue='genotype', row='stimulus',aspect=1.5, size=5,
#                           kind='strip',x='latency',y='treatment',order=treatment_order,jitter=True,color='black')

g.map(sns.swarmplot,'latency','treatment',  color='#333333')
g.map(sns.violinplot,'latency', 'treatment',bw=0.05,cut=0,split=True,order=treatment_order)
#g.map(sns.distplot,'latency',bw=1)
#g.map(sns.stripplot,'latency','treatment', jitter=True, color='gray')
g.set(xticks=np.arange(0,100,4))
g.set_xticklabels(np.arange(0,100,4))
plt.subplots_adjust(top=0.85)
plt.suptitle('Distribution of latencies below %sms' % CUTOFF)
g.savefig(os.path.join(datapath, datafilename+"_latency_dist.png"))
