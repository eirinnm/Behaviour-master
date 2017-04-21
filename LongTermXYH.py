''' 
Process a CSV file of fish tracking data
Assumed first column is IR LED, then subsequent columns are X, Y, Heading for each fish
'''
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
import scipy.ndimage

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
SCALEFACTOR = cpa.args.scalefactor
USEDELTAPIXELS = cpa.args.usedeltapixels
SKIP_FRAMES = cpa.args.skipframes #number of frames to skip at the start of each trial
trialdata = cpa.trialdata
stimname = cpa.stimname
genotype_order = cpa.genotype_order

'''
We have X, Y and heading for each fish.
What is the velocity and angular velocity at each frame?
If the coords are 0 it means the fish was not found, so velocity should be NaN.
We will also need to filter out spurious jumps.
'''


if args.noled:
    movementdata=data
else:
    movementdata=data[:,1:]
## replace zeros with nans
movementdata[movementdata==0]=np.nan
xy=np.dstack((movementdata[:,::3],movementdata[:,1::3]))
headings=movementdata[:,2::3]
#get the euclidean distance moved this frame
#pixels per frame * mm-per-pixel * frames-per-second = mm per second
velocity=np.linalg.norm(xy[1:]-xy[:-1],axis=2)*SCALEFACTOR*FRAMERATE
#smooth it
velocity=scipy.signal.medfilt(velocity,(3,1))
#%%
MIN_BOUT_LENGTH = 13 #frames
MIN_BOUT_DISTANCE = 0.4 #mm
MAX_PEAK_SPEED = 100 #mm/sec
## divide this movement into bouts.
## each bout should start and stop with a few frames of zero velocity (not nans)
### TODO: figure out why Bonsai exports ints!
bdf=[]
for well in range(NUM_WELLS):
    v=velocity[:,well]
    #find contiguous frames of movement
    boutlabels=scipy.ndimage.label(v)[0]
    boutslices=scipy.ndimage.find_objects(boutlabels)
    bouts = [{'start':bout[0].start,'end':bout[0].stop,'fish':well} for bout in boutslices 
             if (not np.isnan(v[bout]).any()) 
             and bout[0].stop-bout[0].start>=MIN_BOUT_LENGTH]
    bdf.extend(bouts) 
bdf=pd.DataFrame(bdf)
bdf['length']=bdf.end-bdf.start
bdf['boutlength']=bdf.length/FRAMERATE
#%%
## What is the peak velocity and distance travelled in each bout?
def get_track_stats(bout):
    b=bout.astype(int) ##shouldn't have to force this but here we are
    v=velocity[b.start:b.end,int(b.fish)]
    #v=scipy.signal.savgol_filter(v,3,1) ## apply a smoothing filter - maybe not necessary
    boutstats = {'total_distance':v.sum()/FRAMERATE,'mean_speed':v.mean(),'peak_speed':v.max(),'peak_frame':v.argmax()}
    return pd.Series(boutstats)
bdf=bdf.merge(bdf.apply(get_track_stats,axis=1),left_index=True,right_index=True)
bdf=bdf[(bdf.total_distance>=MIN_BOUT_DISTANCE) & (bdf.peak_speed<=MAX_PEAK_SPEED)]
#%%
## Plot the tracks for each fish
## first assign colour coding for the length
#bins=np.arange(0,bdf.boutlength.max()+0.1,0.1)
bins=np.append(np.arange(0,1,0.2),np.inf)
#colours=sns.cubehelix_palette(len(bins), start=0, rot=1)
colours=sns.color_palette("YlOrRd", len(bins))
#bdf['lengthbin']=pd.cut(bdf.boutlength,bins,)
bdf['colourcode']=pd.cut(bdf.boutlength,bins).cat.codes
#plt.subplot(2,3,1)
with sns.axes_style("dark"):
    fig, axes=plt.subplots(2,3, sharex=True,sharey=True,figsize=(14,8))
    for fish in range(NUM_WELLS):
        ax=axes.flat[fish]
        #ax.subplot(2,3,fish+1)
        for bout in bdf[bdf.fish==fish].itertuples():
            thistrack=xy[bout.start:bout.end,fish]
            ax.plot(thistrack[:,0],thistrack[:,1],c=colours[bout.colourcode])
        #ax.set(aspect=1)
        ax.set_aspect('equal', 'datalim')
    ax.invert_yaxis() #invert the axis of the last plot; this will flip them all because sharey=true
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.suptitle("Fish paths")
plt.savefig(os.path.join(datapath, datafilename+".tracks.png"))
#%% Function for aligning plots
FIRST_N_POINTS = 10
def rotate_track(bout):
    thistrack=xy[bout.start:bout.end,bout.fish]
    #plt.plot(thistrack[:,0],thistrack[:,1])
    #get the initial heading
    d=np.linalg.norm(thistrack[1:]-thistrack[:-1],axis=1)
    t=np.diff(thistrack,axis=0)
    cos, sin=t[:,0]/d, t[:,1]/d
    ## take an average of these cos/sin values to get an average heading
    c = cos[:FIRST_N_POINTS].mean()
    s = sin[:FIRST_N_POINTS].mean()
    #use a rotation matrix to transform the points
    R = np.matrix([[s, c], [-c, s]])
    rotated = np.dot(thistrack,R)
    #align it with the origin
    return rotated - rotated[0]
#%% Draw the aligned plots
fig, axes = plt.subplots(2,3,figsize=(14,8),sharex=True,sharey=True)
for fish in range(NUM_WELLS):
    ax=axes.flat[fish]
    for bout in bdf[bdf.fish==fish].sort_values('length',ascending=False).itertuples():
        newtrack = rotate_track(bout)
        ax.plot(newtrack[:,0],newtrack[:,1],c=colours[bout.colourcode])
    #ax.set(aspect=1)
    ax.set_aspect('equal', 'datalim')
    ax.set_xlim(-150,150)
    #ax.set_ylim(-25,100)
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.suptitle("Fish paths in aligned space")
plt.savefig(os.path.join(datapath, datafilename+".tracks-aligned.png"))
#%% Draw some interesting bout velocities
#bdf.groupby(['fish',bdf.start//200]).filter(lambda x: (x.total_distance.sum()>7) and (len(x)>1))
some_bouts=bdf.groupby(['fish',bdf.start//200]).filter(lambda x: (x.total_distance.sum()>10) or (len(x)>2))
some_bouts=some_bouts.groupby('fish').first()
#good_times = [(0,18034),
#              (1,4032),
#              (2,66384),
#              (3,178604),
#              (4,36575),
#              (5,83068)]
fig, axes=plt.subplots(2,3, sharex=True,sharey=True, figsize=(14,6))
with sns.axes_style("white"):
    for bout in some_bouts.itertuples():
        startframe=bout.start-50
        endframe=bout.start+450
        fish=bout.Index
        v=velocity[startframe:endframe,fish]
        ax=axes.flat[fish]
        ax.plot(v,lw=1)
        #plt.ylabel('Velocity (mm/sec)')
        #highlight the bouts
        for bout in bdf.query('fish==@fish and start>=@startframe and start<=@endframe').itertuples():
            ax.axvspan(bout.start-startframe,bout.end-startframe,fc='gold',alpha=0.3)
        ax.set_title("Fish %s" % fish,size=12)
    plt.suptitle("Example velocity plots")
plt.savefig(os.path.join(datapath, datafilename+".velocity.png"))
