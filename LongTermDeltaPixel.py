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
import scipy.signal, scipy.stats
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 10, 5
sns.set(context='talk',style='darkgrid',palette='deep',rc={'figure.facecolor':'white'})

DEFAULT_FRAMERATE=100
import common_plate_assay as cpa
args=cpa.get_args(DEFAULT_FRAMERATE,'Long term delta pixels')
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
genotype_order = cpa.genotype_order
#%% Analysis functions
MIN_BOUT_LENGTH = 2 #frames
MIN_BOUT_GAP = 2 #frames
LONGBOUT_THRESHOLD = args.longboutlength #seconds
MIN_ACTIVITY_THRESHOLD = args.minactivity #seconds

def get_bouts(delta_pixels):
    bouts=delta_pixels.nonzero()[0]
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

def process(data):
    global treatment_order
    #filter the data to remove noise
    data=scipy.signal.medfilt(data,(3,1))
    bdf=[] #bout dataframe
    for well in range(NUM_WELLS):
        if args.noled: 
            thismovement=data[:,well]
        else:
            thismovement=data[:,well+1]
        #boutlengths, startframes, intensities = 
        for boutlength, startframe in zip(*get_bouts(thismovement)):
            bdf.append({'fish':well,
                        'boutlength':boutlength/FRAMERATE,
                        'startframe':startframe,
                        })
    bdf=pd.DataFrame(bdf)
    bdf=bdf.merge(conditions,left_on='fish',right_index=True)
    #drop fish with genotype 'x'
    bdf=bdf[bdf.genotype<>'x']
    #the above step might have made some empty treatment groups, remove those
    treatment_order=[treatment for treatment in treatment_order if treatment in bdf.treatment.unique()]
    ## make the fish a category, so missing fish/bouts will show up
    bdf.fish = bdf.fish.astype("category", categories = np.arange(NUM_WELLS))
    ## add a time series and make it a category
    bdf['minute'] = bdf.startframe // (FRAMERATE*60)
    bdf.minute = bdf.minute.astype("category", categories = np.arange(bdf.minute.max()+1))
    return bdf

#%% Run analysis
bdf=process(data)
## widen some plots based on the number of conditions
aspect = 0.75+0.25*(len(treatment_order)-1)
#%%
fig=sns.stripplot(data=bdf, jitter=True,y='boutlength',x='treatment',hue='genotype', 
                  split=True, order=treatment_order, hue_order=genotype_order)
#fig.figure.axes[0].set(yscale="log")
#fig.figure.axes[0].set_yscale('log',subsy=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
plt.title("Bout lengths")
plt.savefig(os.path.join(datapath, datafilename+".bout_lengths.png"))
#plt.show()
#%% Get stats per minute
#calculate the mean length and frequency of bouts per fish and per minute
#also count the number of seizures (long bouts)
#longbouts is "seizures per minute"
def analyse_minute(bouts):
    b=bouts.boutlength
    return pd.Series({'bout_length':b.mean(),
                      'total_activity':b.sum(),
                      'bout_freq':len(b),
                     'long_bouts':np.sum(b>LONGBOUT_THRESHOLD)})
df=bdf.groupby(['fish','minute']).apply(analyse_minute)
df.reset_index(inplace=True)
## merge with conditions
df=pd.merge(df,conditions,left_on='fish',right_index=True)
#what percentage of bouts are long bouts?
df.total_activity.fillna(0, inplace=True)
df.long_bouts.fillna(0, inplace=True)
df.bout_freq.fillna(0, inplace=True)
df['long_bout_pct'] = df.long_bouts / df.bout_freq
df.long_bout_pct.fillna(0, inplace=True)
## Calculate the mean over 15 minute periods
bins=np.arange(0,df.minute.max()+10,15)
labels=["{}-{}".format(a,b) for a,b in zip(bins[:-1], bins[1:])]
df['minutes'] = pd.cut(df.minute,bins=bins,labels=labels, include_lowest=True,right=False,).astype(str)
df.fish=df.fish.astype(int)
## Remove inactive fish: those with mean activity less than a threshold
inactive_fish=df.groupby('fish').mean().query('total_activity<@MIN_ACTIVITY_THRESHOLD').index
print "Removing",len(inactive_fish),"inactive fish:"
print conditions.iloc[inactive_fish]
df=df[~df.fish.isin(inactive_fish)]
#%% Plot time courses
def plot_timecourse(variable, title):
    f=sns.factorplot(data=df,x='minute',y=variable,hue='genotype',hue_order=genotype_order,
                     col_order=treatment_order,col='treatment',
                     scale=0.5, aspect=2)
    plt.suptitle(title)
    plt.subplots_adjust(top=0.85)
    f.set(xticks=[])
    plt.savefig(os.path.join(datapath, "%s.%s_per_minute.png" % (datafilename,variable)))
    #plt.show()
plot_timecourse('total_activity','Activity (seconds) per minute')
plot_timecourse('long_bouts','Seizures per minute')
#%% 15 minute bins
tdf=df.groupby(['treatment','genotype','fish','minutes']).mean().reset_index().drop(['minute'],axis=1)
tdf.sort_values(['fish','minutes'],inplace=True)
sns.factorplot(data=tdf,x='minutes',y='long_bouts',hue='genotype',hue_order=genotype_order, col_order=treatment_order,col='treatment',capsize=.1)
plt.suptitle("Seizures per minute, average 15 mins")
plt.subplots_adjust(top=0.85)
plt.savefig(os.path.join(datapath, datafilename+".seizures-15min.png"))
#%% 15 minute bins, by genotype
#tdf=df.groupby(['treatment','genotype','fish','minutes']).mean().reset_index().drop(['minute'],axis=1)
#tdf.sort_values(['fish','minutes'],inplace=True)
sns.factorplot(data=tdf,x='minutes',y='long_bouts',hue='treatment',hue_order=treatment_order,col='genotype',col_order=genotype_order,capsize=.1)
plt.suptitle("Seizures per minute, average 15 mins")
plt.subplots_adjust(top=0.85)
plt.savefig(os.path.join(datapath, datafilename+".seizures-15min-genotype.png"))
#plt.show()
#%%
fishmeans=df.groupby(['treatment','genotype','fish'],as_index=False).mean()
melted=pd.melt(fishmeans, ['treatment','genotype','fish'],
               ['bout_freq','total_activity','bout_length','long_bouts'])
sns.factorplot(data=melted,x='treatment',y='value',hue='genotype',kind='bar',hue_order=genotype_order,
               order=treatment_order,col='variable', col_wrap=2, aspect=aspect, sharey=False,sharex=False)
plt.suptitle("Mean behaviours")
plt.subplots_adjust(top=0.90)
plt.savefig(os.path.join(datapath, datafilename+".behaviours.png"))
#plt.show()
#%% other behaviours over 15 minute windows
for timebin in tdf.minutes.unique():
    melted=pd.melt(tdf[tdf.minutes==timebin], ['treatment','genotype','fish'],
                   ['bout_freq','total_activity','bout_length','long_bouts'])
    sns.factorplot(data=melted,x='treatment',y='value',hue='genotype',kind='bar',hue_order=genotype_order,
                   order=treatment_order,col='variable', col_wrap=2,aspect=aspect, sharey=False,sharex=False)
    plt.suptitle("Behaviour during minutes "+timebin)
    plt.subplots_adjust(top=0.90)
    plt.savefig(os.path.join(datapath, datafilename+".behaviours-"+timebin+"min.png"))
#plt.show()
#%% plate view
## Which fish were affected the most?
plate_summary = df.groupby(['row','col']).mean().reset_index()
fig,axes=plt.subplots(2,2, sharex=False, sharey=True) #figsize=(16,8)
annot_kws={"size": 10}
plt.subplot(2,2,1)
sns.heatmap(data=plate_summary.pivot('row','col','bout_freq'),annot=True,cbar=False,square=True,annot_kws=annot_kws,fmt=".1f")
plt.title("Bouts per minute")
plt.axis('off')
plt.subplot(2,2,2)
sns.heatmap(data=plate_summary.pivot('row','col','total_activity'),annot=True,cbar=False,square=True,annot_kws=annot_kws,fmt=".1f")
plt.title("Activity (seconds per minute)")
plt.axis('off')
plt.subplot(2,2,3)
sns.heatmap(data=plate_summary.pivot('row','col','bout_length'),annot=True,cbar=False,square=True,annot_kws=annot_kws,fmt=".2f")
plt.title("Mean bout length (seconds)")
plt.axis('off')
plt.subplot(2,2,4)
sns.heatmap(data=plate_summary.pivot('row','col','long_bouts'),annot=True,cbar=False,square=True,annot_kws=annot_kws,fmt=".1f")
plt.title("Seizures per minute")
plt.axis('off')
plt.suptitle("Behaviour per well")
plt.savefig(os.path.join(datapath, datafilename+".plateview.png"))
#%% Some basic stats
## UNCORRECTED T TESTS!
#data_to_compare=tdf[tdf.minutes=='75-90']
data_to_compare = fishmeans
mutants = data_to_compare[data_to_compare.genotype<>'wt']
controls=mutants[mutants.treatment=='Control']
for treatment in treatment_order[1:]:
    pvalue = scipy.stats.ttest_ind(controls.long_bouts, mutants[mutants.treatment==treatment].long_bouts).pvalue
    print "Control vs %s: p =" % treatment, pvalue

#%%
variable='long_bouts'
groups=[group[variable] for name, group in mutants.groupby('treatment')]
