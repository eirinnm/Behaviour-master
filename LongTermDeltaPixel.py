# -*- coding: utf-8 -*-
"""
Created on Wed Feb 08 21:18:24 2017

@author: Eirinn
"""

import os.path
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.signal, scipy.stats
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 10, 5
rcParams['pdf.fonttype'] = 42
rcParams['figure.autolayout'] = True
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
MIN_BOUT_LENGTH = 1 #frames
MIN_BOUT_GAP = 1 #frames
LONGBOUT_THRESHOLD = args.longboutlength #seconds
MIN_ACTIVITY_THRESHOLD = args.minactivity #seconds
MIN_BOUT_FREQ = 0.4#2 #bouts per minute to decide if a fish is active enough

def get_bouts(movementframes):
    # Detect individual bouts and return metrics on them in multiple arrays
    bouts=movementframes.nonzero()[0]
    if len(bouts)>0:
        start_gaps=np.ediff1d(bouts,to_begin=99)
        end_gaps=np.ediff1d(bouts,to_end=99)
        breakpoints=np.vstack((bouts[np.where(start_gaps>MIN_BOUT_GAP)],
                               bouts[np.where(end_gaps>MIN_BOUT_GAP)])) #two columns, start and end frame
        boutlengths=np.diff(breakpoints,axis=0)[0]
        #select only bouts longer than a minimum
        breakpoints=breakpoints[:,boutlengths>=MIN_BOUT_LENGTH]
        boutlengths=boutlengths[boutlengths>=MIN_BOUT_LENGTH]
        
        # calculate "vigour"
        bout_deltapixels = [movementframes[start:end] for start, end in breakpoints.T]
        v_max = np.array([np.max(v) for v in bout_deltapixels])
        v_sum = np.array([np.sum(v) for v in bout_deltapixels])
        v_min = np.array([np.min(v) for v in bout_deltapixels])
        v_std = np.array([np.std(v) for v in bout_deltapixels])
        v_mean = np.array([np.mean(v) for v in bout_deltapixels])
        v_maxpoint = np.array([np.argmax(v)/len(v) for v in bout_deltapixels])

        return boutlengths, breakpoints[0], v_max, v_sum, v_min, v_std, v_mean, v_maxpoint
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
        for boutlength, startframe, v_max, v_sum, v_min, v_std, v_mean, v_maxpoint in zip(*get_bouts(thismovement)):
            bdf.append({'fish':well,
                        'startframe':startframe,
                        'endframe':startframe+boutlength,
                        'boutlength':boutlength/FRAMERATE,
                        'v_max':v_max/255,
                        'v_sum':v_sum/255,
                        'v_min':v_min/255,
                        'v_std':v_std/255,
                        'v_mean':v_mean/255,
                        'v_maxpoint':v_maxpoint,
                        'longbout':boutlength/FRAMERATE>LONGBOUT_THRESHOLD
                        })
    bdf=pd.DataFrame(bdf)
    bdf=bdf.merge(conditions,left_on='fish',right_index=True)
    #drop fish with genotype 'x'
    bdf=bdf[bdf.genotype!='x']
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
#drop fish with genotype 'x'
bdf=bdf[bdf.genotype!='X']
if 'X' in genotype_order: genotype_order.remove('X')
## widen some plots based on the number of conditions
aspect = 0.75+0.25*(len(treatment_order)-1)

#%%
#from sklearn.cluster import KMeans
#kmeans = KMeans(n_clusters=15, random_state=0).fit_predict(Y)
#plt.scatter(Y[:,0],Y[:,1], s=2, c=kmeans)
#%%
plt.figure(figsize=(8,5))
#sns.set_context('poster')
#sns.set_style('whitegrid')
ax=sns.stripplot(data=bdf, jitter=True,y='boutlength',x='treatment',hue='genotype', 
                      dodge=True, order=treatment_order, hue_order=genotype_order, size=5, edgecolor='gray', linewidth=0.5)
ax.set_ylabel('Bout length (ms)')
ax.set_xlabel('Treatment')
#handles, labels = ax.get_legend_handles_labels()
#ax.legend(handles, labels)
plt.title("Bout lengths")
plt.savefig(os.path.join(datapath, datafilename+".bout_lengths.png"))
#plt.savefig(os.path.join(datapath, datafilename+".bout_lengths.pdf"))
#plt.show()
#%% Detailed analysis
#def get_movement(bout):
#    return data[bout.startframe:bout.endframe+1,bout.fish+1]
#tracks=[]
#for name, bouts in bdf.sort_values('boutlength').groupby(pd.cut(bdf.boutlength,50)):
#    for bout in bouts[bouts.genotype=='Wt-zx1'].head(10).itertuples():
#        track = scipy.signal.savgol_filter(get_movement(bout),3,1)
#        tracks.append(track)
#alltracks = np.zeros((len(tracks),max(len(track) for track in tracks)))
#for t, track in enumerate(tracks):
#    alltracks[t,:len(track)]=track
#with sns.axes_style('dark'):
#    plt.imshow(alltracks)
#%%
def multisample_jointplot(data, groupvar='genotype', labels=genotype_order, x='boutlength',y='v_max', s=2):
    #create the first jointgrid with the first distribution
    if labels == None:
        labels = np.sort(bdf[groupvar].unique())
    dist0 = data[data[groupvar] == labels[0]]
    g = sns.JointGrid(data=dist0, x=x,y=y, space=0, ratio=2);
    g = g.plot_joint(plt.scatter, s=s, alpha=0.5)
    g = g.plot_marginals(sns.kdeplot, shade=True,label=labels[0])
    #plot the other distributions
    for label in labels[1:]:
        dist = data[data[groupvar] == label] 
        g.x=dist[x]
        g.y=dist[y]
        g = g.plot_joint(plt.scatter, s=s, alpha=0.5)
        g = g.plot_marginals(sns.kdeplot, shade=True, legend=True, label=label)
    g.ax_marg_x.autoscale()
    g.ax_marg_y.autoscale()
    return g

if len(treatment_order)>1:
    plt.figure()
    g = multisample_jointplot(bdf, 'treatment', treatment_order)
    g.ax_joint.set_xlabel('Bout length (s)')
    g.ax_joint.set_ylabel('Peak velocity (pixels)')
    g.savefig(os.path.join(datapath, datafilename+".swimbouts-treatment.png"))
if len(genotype_order)>1:
    plt.figure()
    g = multisample_jointplot(bdf)
    g.ax_joint.set_xlabel('Bout length (s)')
    g.ax_joint.set_ylabel('Peak velocity (pixels)')
    g.savefig(os.path.join(datapath, datafilename+".swimbouts-genotype.png"))
#plt.suptitle("Swim bout analysis")
#plt.subplots_adjust(top=0.85)
#%% PCA
#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA
## Separating out the features
#features = ['v_max','v_sum','v_min','v_std','v_mean','v_maxpoint','boutlength']
#x = StandardScaler().fit_transform(bdf[features].values)
#pca = PCA()
#pc = pca.fit_transform(x)
#pca.explained_variance_ratio_
#bdf['pc1'] = pc[:,0]
#bdf['pc2'] = pc[:,1]
#bdf['pc3'] = pc[:,2]
#if len(treatment_order)>1:
#    plt.figure()
#    g = multisample_jointplot(bdf, 'treatment', treatment_order, x='pc1',y='pc2')
#    g.ax_joint.set_xlabel('PC1')
#    g.ax_joint.set_ylabel('PC2')
#    g.savefig(os.path.join(datapath, datafilename+".pca-treatment.png"))
#if len(genotype_order)>1:
#    plt.figure()
#    g = multisample_jointplot(bdf, x='pc1',y='pc2')
#    g.ax_joint.set_xlabel('PC1')
#    g.ax_joint.set_ylabel('PC2')
#    g.savefig(os.path.join(datapath, datafilename+".pca-genotype.png"))
#%%
#from MulticoreTSNE import MulticoreTSNE as TSNE
#tsne = TSNE(n_jobs=4, n_components=3, perplexity=80)
#Y = tsne.fit_transform(pc)
#bdf['tsne1'] = Y[:,0]
#bdf['tsne2'] = Y[:,1]
#multisample_jointplot(bdf,x='tsne1',y='tsne2')
#%% Bout classification
#from sklearn import mixture
#gmm = mixture.GaussianMixture(n_components=4).fit(pc)
#classes = gmm.predict(pc)
#bdf['boutclass']=classes
#plt.figure()
#if len(genotype_order)>1:
#    class_props = bdf.groupby(['genotype','fish']).boutclass.value_counts(normalize=True).reset_index(name='prop')
#    sns.barplot(data=class_props,x='boutclass',y='prop', hue='genotype', hue_order=genotype_order, dodge=True)
#    plt.title("Proportion of each bout class per fish")
#    plt.ylabel("Proportion of bouts")
#    plt.savefig(os.path.join(datapath, datafilename+".boutclasses.genotype.png"))
#if len(treatment_order)>1:
#    class_props = bdf.groupby(['treatment','fish']).boutclass.value_counts(normalize=True).reset_index(name='prop')
#    sns.barplot(data=class_props,x='boutclass',y='prop', hue='treatment', hue_order=treatment_order, dodge=True)
#    plt.title("Proportion of each bout class per fish")
#    plt.ylabel("Proportion of bouts")
#    plt.savefig(os.path.join(datapath, datafilename+".boutclasses.treatment.png"))    
#%%
# Save the BDF table.
output_df = bdf.copy()
output_df['numfish']=NUM_WELLS
output_df['expt']=datafilename
output_df.to_csv(os.path.join(datapath, datafilename+".bdf.txt"), sep='\t', index=False)
#%% Get stats per minute
#calculate the mean length and frequency of bouts per fish and per minute
#also count the number of seizures (long bouts)
#longbouts is "seizures per minute"
def analyse_minute(bouts):
    b=bouts.boutlength
    return pd.Series({'boutlength':b.mean(),
                      'total_activity':b.sum(),
                      'bout_freq':len(b),
                     'long_bouts':np.sum(b>LONGBOUT_THRESHOLD),
                     })
df=bdf.groupby(['fish','minute']).apply(analyse_minute)
df.reset_index(inplace=True)
## merge with conditions
df=pd.merge(df,conditions,left_on=df.fish.astype(int),right_index=True)
#what percentage of bouts are long bouts?
df.total_activity.fillna(0, inplace=True)
df.long_bouts.fillna(0, inplace=True)
df.bout_freq.fillna(0, inplace=True)
df['long_bout_pct'] = df.long_bouts / df.bout_freq
df.long_bout_pct.fillna(0, inplace=True)
## Calculate the mean over 15 minute periods
bins=np.arange(0,df.minute.cat.categories.max()+10,15)
labels=["{}-{}".format(a,b) for a,b in zip(bins[:-1], bins[1:])]
df['minutes'] = pd.cut(df.minute,bins=bins,labels=labels, include_lowest=True,right=False,)#.astype(str)
df.fish=df.fish.astype(int)
## Remove inactive fish: those with mean activity less than a threshold
inactive_fish=df.groupby('fish').mean().query('total_activity<@MIN_ACTIVITY_THRESHOLD').index
if len(inactive_fish):
    print("Removing",len(inactive_fish),"inactive fish:")
    print(conditions.iloc[inactive_fish])
    df=df[~df.fish.isin(inactive_fish)]
#%% Plot time courses
def plot_timecourse(variable, title):
    f=sns.factorplot(data=df,x='minute',y=variable,hue='genotype',hue_order=genotype_order,
                     row_order=treatment_order,row='treatment',
                     scale=0.5, aspect=2, linewidth=1, ci=None)
    plt.suptitle(title, y=1)
    f.set(xticks=[])
    plt.savefig(os.path.join(datapath, "%s.per_minute_%s.png" % (datafilename,variable)))
    #plt.show()
plot_timecourse('total_activity','Activity (seconds) per minute')
plt.subplots_adjust(top=0.80)
plot_timecourse('long_bouts','Seizures per minute')
#%% 15 minute bins
tdf=df.groupby(['treatment','genotype','fish','minutes']).mean().reset_index()#.drop(['minute'],axis=1)
tdf = tdf[~tdf.col.isnull()]
tdf.sort_values(['fish','minutes'],inplace=True)
if len(tdf.minutes.unique())>1:
    ax=sns.factorplot(data=tdf,x='minutes',y='long_bouts',hue='genotype',hue_order=genotype_order,
                      row_order=treatment_order,row='treatment',capsize=.1, aspect=2)
    plt.suptitle("Seizures per minute, average 15 mins", y=1)
    plt.savefig(os.path.join(datapath, datafilename+".seizures-15min.png"))


#%% 15 minute bins, by genotype
#tdf=df.groupby(['treatment','genotype','fish','minutes']).mean().reset_index().drop(['minute'],axis=1)
#tdf.sort_values(['fish','minutes'],inplace=True)
if len(tdf.minutes.unique())>1:
    sns.factorplot(data=tdf,x='minutes',y='long_bouts',hue='treatment',hue_order=treatment_order,row='genotype',row_order=genotype_order,capsize=.1, aspect=2)
    plt.suptitle("Seizures per minute, average 15 mins", y=1)
    plt.subplots_adjust(top=0.85)
    plt.savefig(os.path.join(datapath, datafilename+".seizures-15min-genotype.png"))
#plt.show()
#%%
#g = sns.FacetGrid(data=tdf,row='treatment',col='genotype', col_order=genotype_order)
#g.map(plt.plot,y='long_bouts',x='minutes')
#%%
fishmeans=df.groupby(['treatment','genotype','fish'],as_index=False).mean()
old_count=len(fishmeans)
fishmeans=fishmeans[fishmeans.bout_freq>=MIN_BOUT_FREQ]
if len(fishmeans)!=old_count: 
    print("Note: removed", old_count-len(fishmeans), "inactive fish. Counts are now:")
    print(fishmeans.groupby(['genotype','treatment']).size().unstack())
melted=pd.melt(fishmeans, ['treatment','genotype','fish'],
               ['bout_freq','total_activity','boutlength','long_bouts'])
ax=sns.factorplot(data=melted,x='treatment',y='value',hue='genotype',kind='bar',hue_order=genotype_order,capsize=.1,
               order=treatment_order,col='variable', col_wrap=2, aspect=aspect, sharey=False,sharex=False, legend_out=False)
ax.fig.tight_layout()

#plt.suptitle("Mean behaviours")
plt.subplots_adjust(top=0.6)
plt.tight_layout()
plt.savefig(os.path.join(datapath, datafilename+".behaviours.png"))
#plt.show()
#%% Seizures per minute, expansion of the smaller plot just generated
plt.figure(figsize=(8,5))
ax=sns.pointplot(data=fishmeans,x='treatment',y='long_bouts',hue='genotype',hue_order=genotype_order,order=treatment_order,capsize=0.1)
plt.title('Long (>0.5s) bouts per minute')
ax.set_ylabel('Long bouts per minute')
#ax.set_xlabel('PTZ concentration')
plt.savefig(os.path.join(datapath, datafilename+".seizures-per-treatment.png"))
#plt.savefig(os.path.join(datapath, datafilename+".seizures-per-treatment.pdf"))
#%% other behaviours over 15 minute windows
#for timebin in tdf.minutes.unique():
#    #fishmeans_thisbin = 
#    melted=pd.melt(tdf[(tdf.minutes==timebin) & (tdf.bout_freq>MIN_BOUT_FREQ)], ['treatment','genotype','fish'],
#                   ['bout_freq','total_activity','boutlength','long_bouts'])
#    sns.factorplot(data=melted,x='treatment',y='value',hue='genotype',kind='bar',hue_order=genotype_order,capsize=.1,
#                   order=treatment_order,col='variable', col_wrap=2,aspect=aspect, sharey=False,sharex=False, legend_out=False)
#    plt.suptitle("Behaviour during minutes "+timebin, y=1)
#    #plt.subplots_adjust(top=0.80)
#    plt.savefig(os.path.join(datapath, datafilename+".behaviours-"+timebin+"min.png"))
#plt.show()
#%% plate view
## Which fish were affected the most?
plate_summary = df.groupby(['row','col']).mean().reset_index()
fig,axes=plt.subplots(2,2, sharex=False, sharey=True,figsize=(16,8))
annot_kws={"size": 10}
plt.subplot(2,2,1)
sns.heatmap(data=plate_summary.pivot('row','col','bout_freq'),annot=True,cmap="coolwarm",cbar=False,square=True,annot_kws=annot_kws,fmt=".1f")
plt.title("Bouts per minute")
plt.axis('off')
plt.subplot(2,2,2)
sns.heatmap(data=plate_summary.pivot('row','col','total_activity'),annot=True,cmap="coolwarm",cbar=False,square=True,annot_kws=annot_kws,fmt=".1f")
plt.title("Activity (seconds per minute)")
plt.axis('off')
plt.subplot(2,2,3)
sns.heatmap(data=plate_summary.pivot('row','col','boutlength'),annot=True,cmap="coolwarm",cbar=False,square=True,annot_kws=annot_kws,fmt=".2f")
plt.title("Mean bout length (seconds)")
plt.axis('off')
plt.subplot(2,2,4)
sns.heatmap(data=plate_summary.pivot('row','col','long_bouts'),annot=True,cmap="coolwarm",cbar=False,square=True,annot_kws=annot_kws,fmt=".1f")
plt.title("Seizures per minute")
plt.axis('off')
plt.suptitle("Behaviour per well", y=1)
plt.subplots_adjust(top=0.50)
plt.savefig(os.path.join(datapath, datafilename+".plateview.png"))
#%% Distplots per fish

# Use scipy to generate a KDE estimate for each fish. Then we can plot them along with a thicker line for the mean KDE per genotype.
def multi_kde(var='boutlength',condition='genotype', gridsize=100):
    kde_range = np.linspace(0, bdf[var].max(),gridsize)
    kdes = []
    for f in range(NUM_WELLS):
        y_vals = bdf[bdf.fish==f][var]
        if len(y_vals)>0:
            this_fish_data = bdf[bdf.fish==f][var]
            if len(this_fish_data)>3:
                kde_func = scipy.stats.gaussian_kde(this_fish_data)
                y = kde_func.evaluate(kde_range)
                kde = pd.DataFrame(data=y,index=kde_range,columns=['density'])
                kde['fish'] = f
                kdes.append(kde)
    kdes = pd.concat(kdes)
    kdes = pd.merge(kdes,conditions, left_on='fish',right_index=True)

    fig = plt.figure()
    ax = fig.gca()
    if condition=='genotype':
        condition_order = genotype_order
    elif condition=='treatment':
        condition_order = treatment_order
    for cond in condition_order:
        kde_wide = kdes[kdes[condition]==cond].pivot(columns='fish', values='density')
        c = sns.color_palette()[condition_order.index(cond)]
        kde_wide.plot(ax=ax, kind='line',color=[c], legend=False, alpha=0.1)
        kde_wide.mean(axis=1).plot(ax=ax, kind='line',color=[c], legend=True, label=cond, lw=5)


if len(genotype_order)>1:
    fig = multi_kde('boutlength', 'genotype')
    plt.title('Distributions of bout length by genotype')
    plt.xlabel('Bout length (seconds)')
    plt.ylabel('Density')
    plt.savefig(os.path.join(datapath, datafilename+".boutlength.genotype.png"))

if len(treatment_order)>1:
    fig = multi_kde('boutlength', 'treatment')
    plt.title('Distributions of bout length by treatment')
    plt.xlabel('Bout length (seconds)')
    plt.ylabel('Density')
    plt.savefig(os.path.join(datapath, datafilename+".boutlength.treatment.png"))


#%% Some basic stats
#import statsmodels.api as sm
#result_table = []
#vars_to_test = ["long_bouts","bout_freq"]
#for var in vars_to_test:
#    for geno in genotype_order:
#        data_to_compare = fishmeans[fishmeans.genotype==geno]
#        control_group=data_to_compare[data_to_compare.treatment=='Control']
#        for treatment in treatment_order[1:]:
#            comparison_group = data_to_compare[data_to_compare.treatment==treatment]
#            pvalue = scipy.stats.ttest_ind(control_group[var], comparison_group[var]).pvalue
#            result_table.append(dict(genotype=geno, variable=var, control_vs=treatment, pvalue=pvalue))
#result_table=pd.DataFrame(result_table)
#result_table['pvalue_adjusted']=sm.stats.multipletests(result_table.pvalue, method='hommel')[1]
#print(result_table)
#%% More advanced stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Test the counts of long_bouts per fish in each treatment group
lm = smf.glm(formula=f'long_bouts ~ C(treatment, Treatment(reference="{treatment_order[0]}"))*C(genotype, Treatment(reference="{genotype_order[0]}"))',
             data=fishmeans, family=sm.families.Poisson()).fit()

print(lm.summary())

#%% Special GLM for ZX1+PTZ
#bdf['PTZ'] = bdf.treatment.str.contains('PTZ')
#bdf['ZX1'] = bdf.treatment.str.contains('ZX1')
#lm=smf.glm(formula='boutlength ~ PTZ*ZX1', data=bdf, family=sm.families.Gamma(link=sm.families.links.log)).fit()
#print(lm.summary())
#%% Mixed effects model
lme = smf.mixedlm(f'boutlength ~ C(treatment, Treatment(reference="{treatment_order[0]}"))*C(genotype, Treatment(reference="{genotype_order[0]}"))', data=bdf, groups=bdf.fish).fit()
print(lme.summary())