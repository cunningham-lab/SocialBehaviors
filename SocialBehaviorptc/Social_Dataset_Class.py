import os
import sys
import pdb
from tqdm import tqdm

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.ndimage import rotate
from scipy.ndimage.morphology import binary_dilation
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from skimage.util import img_as_ubyte

import moviepy
from moviepy.editor import VideoClip,VideoFileClip
from moviepy.video.io.bindings import mplfig_to_npimage

# import tensorflow as tf


### Geometry helper functions
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def unit_vector_vec(vectorvec):
    return vectorvec / np.linalg.norm(vectorvec,axis = 1,keepdims=True)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def angle_between_vec(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector_vec(v1)
    v2_u = unit_vector_vec(v2)
    dotted = np.einsum('ik,ik->i', v1_u, v2_u)
    return np.arccos(np.clip(dotted, -1.0, 1.0))

## Tensorflow Data API helper functions:
## Designate helper function to define features for examples more easily
"""
def _int64_feature_(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature_(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _float_feature_(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
"""

"""
def _write_ex_animal_focused(videoname,i,image,mouse,pos):
    features = {'video': _bytes_feature_(tf.compat.as_bytes((videoname))),
                'frame': _int64_feature_(i),
                'image': _bytes_feature_(tf.compat.as_bytes(image.tobytes())),
                'mouse': _int64_feature_(mouse),
                'position':_bytes_feature_(tf.compat.as_bytes(pos.tobytes()))
                }
    example = tf.train.Example(features = tf.train.Features(feature = features))
    return example
"""

## indexing helper functions:
def sorter(index,bp = 5,cent_ind = 3):
    # return the center index and other body part indices given a mouse index
    center = index*bp+3
    all_body = np.arange(bp)+index*bp
    rel = list(all_body)
    rel.pop(cent_ind)
    return center,rel

## helper function for segmentation:
def find_segments(indices):
    differences = np.diff(indices)
    all_intervals = []
    ## Initialize with the first element added:
    interval = []
    interval.append(indices[0])
    for i,diff in enumerate(differences):
        if diff == 1:
            pass # interval not yet over
        else:
            # last interval ended
            if interval[0] == indices[i]:
                interval.append(indices[i]+1)
            else:
                interval.append(indices[i])
            all_intervals.append(interval)
            # start new interval
            interval = [indices[i+1]]
        if i == len(differences)-1:
            interval.append(indices[-1]+1)
            all_intervals.append(interval)
    return all_intervals

## Helper function to return a segment given relevant information. Handles corner cases:
def segment_getter(trajectories,segs,segind,mouseind):
    ## If segind is negative, return the first element of the trajectory:
    if segind <0:
        segment = trajectories[mouseind][0:1,:]
        return segment
    else:
        segbounds = segs[mouseind][segind]
        segment = trajectories[mouseind][segbounds[0]:segbounds[1]]
        return segment

def interpolate_isnans(trajectories,out_array):
    vgood,mgood = np.where(~np.isnan(out_array[:,0:1]))[0],np.where(~np.isnan(out_array[:,1:2]))[0]
    good_valsv,good_valsm = trajectories[0][vgood,:],trajectories[1][mgood,:]
    vinterp = interp1d(vgood,good_valsv,axis = 0,bounds_error = False,fill_value = 'extrapolate',kind = 'slinear')
    minterp = interp1d(mgood,good_valsm,axis = 0,bounds_error = False,fill_value = 'extrapolate',kind = 'slinear')
    return vinterp,minterp


class social_dataset(object):
    def __init__(self,filepath,vers = 0):
        self.dataset = pd.read_hdf(filepath)
        self.dataset_name = filepath.split('/')[-1]
        self.scorer = 'DeepCut' + filepath.split('DeepCut')[-1].split('.h5')[0]
        self.part_list = ['vtip','vlear','vrear','vcent','vtail','mtip','mlear','mrear','mcent','mtail']
        self.part_index = np.arange(len(self.part_list))
        self.part_dict = {index:self.part_list[index] for index in range(len(self.part_list))}
        self.time_index = np.arange(self.dataset.shape[0])
        self.allowed_index = [self.time_index[:,np.newaxis] for _ in self.part_list] ## set of good indices for each part!
        self.allowed_index_full = [self.simple_index_maker(part,self.allowed_index[part]) for part in self.part_index]
        self.filter_check_counts = [[] for i in self.part_index]
        self.vers = vers ## with multiple animal index or not

    # Helper function: index maker. Takes naive index constructs and returns a workable set of indices for the dataset.
    def simple_index_maker(self,pindex,allowed):
        length_allowed = len(allowed)
        # First compute the part index as appropriate:
        part = np.array([[pindex]]).repeat(length_allowed,axis = 0)

        full_index = np.concatenate((allowed,part),axis = 1)
        return full_index

    # Trajectory selector:
    def select_trajectory(self,pindex):
        # return the x, y coordinates
        part_name = self.part_dict[pindex]
        part_okindex = self.allowed_index[pindex]
        if self.vers == 0:
            rawtrajectory = self.dataset[self.scorer][part_name]['0'].values[:,:2]
        else:
            rawtrajectory = self.dataset[self.scorer][part_name].values[:,:2]
        return rawtrajectory

    # if a trajectory could be influenced by other trajectories:
    # do interperlation for not-ok data
    def render_trajectory_full(self,pindex):
        rawtrajectories = self.dataset[self.scorer].values # np array version of the dataframe
        part_okindex = self.allowed_index_full[pindex]
        time = part_okindex[:,0:1]
        # this is the x,y coordinate of the dataframe
        x = part_okindex[:,1:2]*3
        y = part_okindex[:,1:2]*3+1
        coords = np.concatenate((x,y),axis = 1)
        out = rawtrajectories[time,coords]
        filtered_x,filtered_y = np.interp(self.time_index,part_okindex[:,0],out[:,0]),np.interp(self.time_index,part_okindex[:,0],out[:,1])
        filtered_part = np.concatenate((filtered_x[:,np.newaxis],filtered_y[:,np.newaxis]),axis = 1)
        return filtered_part


    # # if a trajectory could be influenced by other trajectories:
    # def _render_trajectory_full(self,pindex):
    #     rawtrajectories = self.dataset[self.scorer].values
    #     part_okindex = self.allowed_index_full[pindex]
    #     time = part_okindex[:,0:1]
    #     x = part_okindex[:,1:2]*3
    #     y = part_okindex[:,1:2]*3+1
    #     coords = np.concatenate((x,y),axis = 1)
    #     out = rawtrajectories[time,coords]
    #     f = interp1d(part_okindex[:,0],out,axis = 0,kind = 'cubic')
    #     # filtered_x,filtered_y = np.interp(self.time_index,part_okindex[:,0],out[:,0]),np.interp(self.time_index,part_okindex[:,0],out[:,1])
    #     filtered_part = f(self.time_index)
    #     return filtered_part

    def render_trajectory_valid(self,pindex):
        rawtrajectories = self.dataset[self.scorer].values
        part_okindex = self.allowed_index_full[pindex]
        time = part_okindex[:,0:1]
        x = part_okindex[:,1:2]*3
        y = part_okindex[:,1:2]*3+1
        coords = np.concatenate((x,y),axis = 1)
        out = rawtrajectories[time,coords]
        return out,time

    # For multiple trajectories:

    def render_trajectories(self,to_render = None):
        if to_render == None:
            to_render = self.part_index
        part_trajectories = []
        for pindex in to_render:
            part_traj = self.render_trajectory_full(pindex)
            part_trajectories.append(part_traj)

        return(part_trajectories)


    # Now define a plotting function:
    # Also plot a tracked frame on an image:
    def plot_image(self,part_numbers,frame_nb):
        allowed_indices = [frame_nb in self.allowed_index_full[part_number][:,0] for part_number in part_numbers]
        colors = ['red','blue']
        point_colors = [colors[allowed] for allowed in allowed_indices]
        relevant_trajectories = self.render_trajectories(to_render = part_numbers)
        print(relevant_trajectories[0].shape)
        relevant_points = [traj[frame_nb,:] for traj in relevant_trajectories]
        relevant_points = np.array(relevant_points)
        assert np.all(np.shape(relevant_points) == (len(part_numbers),2))
        print(self.movie.duration,self.movie.fps)

        # Now load in video:
        try:
            frame = self.movie.get_frame(frame_nb/self.movie.fps)
            fig,ax = plt.subplots()
            ax.imshow(frame)
            ax.axis('off')
            ax.scatter(relevant_points[:,0],relevant_points[:,1],c = point_colors)
            plt.show()
        except OSError as error:
            print(error)

    # Plot a tracked frame on an image, with raw trackings for comparison:
    def plot_image_compare(self,part_numbers,frame_nb,xlims = [0,-1],ylims = [0,-1],internal = False,figureparams = None):

        print(frame_nb)
        allowed_indices = [frame_nb in self.allowed_index_full[part_number][:,0] for part_number in part_numbers]
        colors = ['blue','red']
        shapes = ['o','v']
        colorsequence = [colors[i >= 5] for i in part_numbers]

        point_colors = [colors[allowed] for allowed in allowed_indices]
        relevant_trajectories = self.render_trajectories(to_render = part_numbers)
        relevant_points = [traj[frame_nb,:] for traj in relevant_trajectories]
        relevant_points = np.array(relevant_points)

        rawtrajectories = self.dataset[self.scorer].values[frame_nb,:]
        a = rawtrajectories.reshape(10,3)
        relevant_raw = np.array([part[:2] for i, part in enumerate(a) if i in part_numbers])

        assert np.all(np.shape(relevant_points) == (len(part_numbers),2))
        assert np.all(np.shape(relevant_raw) == (len(part_numbers),2))

        # Now load in video:
        try:
            frame = self.movie.get_frame(frame_nb/self.movie.fps)
            if figureparams is None:
                fig,ax = plt.subplots()
            else:
                fig,ax = figureparams[0],figureparams[1]

            shape = np.array(np.shape(frame[xlims[0]:xlims[1],ylims[0]:ylims[1]]))[:2].reshape((1,2))
            ax.imshow(frame[xlims[0]:xlims[1],ylims[0]:ylims[1]])
            ax.axis('off')

            relevant_points[relevant_points == 0] += 1
            relevant_points = np.minimum(relevant_points,shape-1)
            ax.scatter(relevant_points[:,0],relevant_points[:,1]-xlims[0],c = colorsequence,marker = shapes[0],s = 100)
            # ax.scatter(relevant_raw[:,0],relevant_raw[:,1]-xlims[0],c = colorsequence,marker = shapes[1])
            if figureparams is not None:
                return fig,ax
            else:
                if internal == False:
                    plt.show()
                else:
                    return mplfig_to_npimage(fig)

        except AttributeError as error:
            print(error,' Plotting frames without video')
            if figureparams is None:
                fig,ax = plt.subplots()
            else:
                fig,ax = figureparams[0],figureparams[1]
            ax.set_xlim([0,630-330])
            ax.set_ylim([480-70,0])
            ax.set_aspect('equal')
            ax.scatter(relevant_points[:,0],relevant_points[:,1],c = colorsequence,marker = shapes[0])
            ax.scatter(relevant_raw[:,0],relevant_raw[:,1],c = colorsequence,marker = shapes[1])
            if figureparams is not None:
                return fig,ax
            else:
                if internal == False:
                    plt.show()
                else:
                    return mplfig_to_npimage(fig)


    # Plot a tracked frame on an image, with raw trackings for comparison:
    def plot_clip_compare(self,part_numbers,frame_sequence,fps):
        ## See how many frames we should worry about:
        length = len(frame_sequence)
        duration = length/fps
        ## Make function to pass to clipwriter:
        framemaker = lambda t: self.plot_image_compare(part_numbers,frame_sequence[int(t*fps)],internal = True)
        animation = VideoClip(framemaker,duration = duration)
        return animation

    def plot_clip_with_ethogram(self,part_numbers,interval,fps,title):
        length = len(interval)
        duration = length/fps

        ## Pull the ethogram we will use:
        vetho = self.nest_ethogram(0)
        detho = self.nest_ethogram(1)
        petho = self.shepherding_ethogram()
        ethox = np.arange(len(vetho))
        fig = plt.figure(figsize = (15,10))
        ax0 = plt.subplot2grid((5, 4), (0, 0), rowspan=5,colspan = 2)
        ax1 = plt.subplot2grid((5, 4), (1, 2), colspan = 2)
        ax2 = plt.subplot2grid((5, 4), (3, 2), colspan = 2)
        ax = [ax0,ax1,ax2]
        ax[1].plot(vetho,color ='blue',label = 'NEST')
        ax[2].plot(detho,color = 'red',label = 'NEST')
        ax[1].fill_between(ethox,0,vetho,color = 'blue')
        ax[2].fill_between(ethox,0,detho,color = 'red')
        ax[1].set_title('Virgin Ethogram')
        ax[2].set_title('Dam Ethogram')
        [ax[i].plot(petho,color = 'orange',label = 'pursuit') for i in [1,2]]
        [ax[i].legend() for i in [1,2]]
        ## define a small function we will call as a lambda function:
        def framemaker(t,vetho,detho,petho):
            index = int(t*fps)
            ax[0].clear()
            fig0,ax0 = self.plot_image_compare(part_numbers,interval[index],figureparams=(fig,ax[0]))
            ## Plot markers based on ethogram too:
            if vetho[interval[index]] or detho[interval[index]] or petho[interval[index]]:
                ax0.tick_params(
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    left = False,
                    labelbottom=False,
                    labelleft=False) # labels along the bottom edge are off
                ax0.axis('on')
            if vetho[interval[index]]:
                ax0.spines['left'].set_linewidth(2)
                ax0.spines['left'].set_color('blue')
            if detho[interval[index]]:
                ax0.spines['right'].set_linewidth(2)
                ax0.spines['right'].set_color('red')
            if petho[interval[index]]:
                ax0.scatter(270,15,color = 'orange',marker = 'o',s = 100)
            ax[1].axvline(x = interval[index],color = 'black',alpha = 0.5)
            ax[2].axvline(x = interval[index],color = 'black',alpha = 0.5)
            return mplfig_to_npimage(fig)

        make_frame = lambda t:framemaker(t,vetho,detho,petho)
        clip = VideoClip(make_frame,duration=duration)
        clip.write_videofile(title,fps = fps)


    def plot_trajectory(self,part_numbers,start = 0,end = -1,cropx = 0,cropy = 0,axes = True,save = False,**kwargs):
        # First define the relevant part indices:
        relevant_trajectories = self.render_trajectories(to_render = part_numbers)
        names = ['Virgin','Dam']
        mouse_id = int(part_numbers[0]/5)
        if axes == True:
            fig,axes = plt.subplots()
            for part_nb,trajectory in enumerate(relevant_trajectories):
                axes.plot(trajectory[start:end,0],-trajectory[start:end,1],label = self.part_list[part_numbers[part_nb]],**kwargs)
            axes.axis('equal')
            plt.legend()
            plt.title(names[mouse_id]+' Trajectories')
#             plt.yticks(0,[])
#             plt.xticks(0,[])
            plt.show()
            plt.close()
            if save != False:
                plt.savefig(save)

        else:
            for part_nb,trajectory in enumerate(relevant_trajectories):
                axes.plot(trajectory[start:end,0]-cropx,trajectory[start:end,1]-cropy,label = self.part_list[part_numbers[part_nb]],**kwargs)

## Closely related to the plotting function is the gif rendering function for trajectories:
    def gif_trajectory(self,part_numbers,start = 0,end = -1,fps = 60.,cropx = 0,cropy = 0,save = False,**kwargs):
        # First define the relevant part indices:
        relevant_trajectories = self.render_trajectories(to_render = part_numbers)
        names = ['Virgin','Dam']
        mouse_id = int(part_numbers[0]/5)
        if end == -1:
            duration = relevant_trajectories.shape[0]-start
        else:
            duration = end - start

        clipduration = duration/fps

        fig,axes = plt.subplots()
        print(relevant_trajectories[0][start:start+2+int(5*fps),0].shape)
        ## Define a frame making function to pass to moviepy:
        def gif_trajectory_mini(t):
            axes.clear()
            for part_nb,trajectory in enumerate(relevant_trajectories):
                axes.plot(trajectory[start:start+2+int(t*fps),0],-trajectory[start:start+2+int(t*fps),1],label = self.part_list[part_numbers[part_nb]],**kwargs)
                axes.plot(trajectory[start+int(t*fps),0],-trajectory[start+int(t*fps),1],'o',markersize = 3,label = self.part_list[part_numbers[part_nb]],**kwargs)
            axes.axis('equal')

            return mplfig_to_npimage(fig)

        animation = VideoClip(gif_trajectory_mini,duration = clipduration)
        # animation.ipython_display(fps=60., loop=True, autoplay=True)
        return animation


####### Quantifying errors:
    ## Look at a patch of the underlying image:
    def make_patches(self,frames,part,radius):
        points = self.render_trajectory_full(part)[frames,:]
        xcents,ycents = points[:,0],points[:,1]
        xsize,ysize = self.movie.size
        all_clipped = np.zeros((len(frames),2*radius,2*radius,3)).astype(np.uint8)
        for i,frame in enumerate(frames):
            image = self.movie.get_frame((frame)/self.movie.fps)
            xcent,ycent = xcents[i],ycents[i]
            xmin,xmax,ymin,ymax = int(xcent-radius),int(xcent+radius),int(ycent-radius),int(ycent+radius)
            ## do edge detection:
            pads  = np.array([[ymin - 0,ysize - ymax],[xmin - 0,xsize - xmax],[0,0]])


            clip = image[ymin:ymax,xmin:xmax]
            # print(clip,'makedocip')
            if np.any(pads < 0):
                topad = pads<0
                padding = -1*pads*topad
                clip = np.pad(clip,padding,'edge')
            all_clipped[i,:,:,:] = clip.astype(np.uint8)
        return all_clipped

    def patch_grandhist(self,frames,part,radius):
        dataarray = self.make_patches(frames,part,radius)
        hists = [np.histogram(dataarray[:,:,:,i],bins = np.linspace(0,255,256)) for i in range(3)]
        return hists

    def patch_hist(self,frames,part,radius):
        dataarray = self.make_patches(frames,part,radius)
        hists = [[np.histogram(dataarray[f,:,:,i],bins = np.linspace(0,255,256)) for i in range(3)]for f in range(len(frames))]
        return hists

    # def patch_outliers
#######

    def calculate_speed(self,pindex):
        rawtrajectory = self.select_trajectory(pindex)
        diff = np.diff(rawtrajectory,axis = 0)
        speed = np.linalg.norm(diff,axis = 1)
        return speed

    def motion_detector(self,threshold = 80):
        ## We will assume the trajectories are raw.
        indices = np.array([0,1,2,3,4])
        mouse_moving = np.zeros((2,))
        for mouse in range(2):
            mouseindices = indices+5*mouse
            trajectories = np.concatenate(self.render_trajectories(list(indices)),axis = 1)

            maxes = np.max(trajectories,axis = 0)
            mins = np.min(trajectories,axis = 0)
            differences = maxes-mins
            partwise_diffs = differences.reshape(5,2)
            normed = np.linalg.norm(partwise_diffs,axis = 1)
            ## See if any of the body parts left
            if np.any(normed <threshold):
                mouse_moving[mouse] = 0
            else:
                mouse_moving[mouse] = 1

        return mouse_moving,normed

    # Now we define filtering functions:
    # Filter by average speed
    def filter_speed(self,pindex,threshold = 10):
        rawtrajectory = self.select_trajectory(pindex)
        diff = np.diff(rawtrajectory,axis = 0)
        speed = np.linalg.norm(diff,axis = 1)
        filterlength = 9

        filterw = np.array([1.0 for i in range(filterlength)])/filterlength
        filtered = np.convolve(speed,filterw,'valid')
        # virg_outs = np.where(filtered > 10)[0]
        outliers = np.where(filtered>threshold)[0]

        okay_indices = np.array([index for index in self.allowed_index_full[pindex] if index[0] not in outliers])

        self.allowed_index_full[pindex] = self.simple_index_maker(pindex,okay_indices[:,0:1])

    # Filter by likelihood
    def filter_likelihood(self,pindex):
        part_name = self.part_dict[pindex]
        part_okindex = self.allowed_index_full[pindex]
        if self.vers == 0:
            likelihood = self.dataset[self.scorer][part_name]['0'].values[:,2]
        else:
            likelihood = self.dataset[self.scorer][part_name].values[:,2]
        outliers = np.where(likelihood<0.95)[0]
        okay_indices = np.array([index for index in self.allowed_index_full[pindex] if index[0] not in outliers])

        self.allowed_index_full[pindex] = self.simple_index_maker(pindex,okay_indices[:,0:1])

    def filter_nests(self):
        try:
            print("NEST bounds are: "+str(self.bounds))
            indices = np.array([0,1,2,3,4])
            for mouse in range(2):
                mouseindices = indices+5*mouse
                ## Bounds are defined as x greater than some value, y greater than some value (flipped on plot)
                self.filter_speeds(mouseindices,9)
                whole_traj = self.render_trajectories(list(mouseindices))
                for part_nb,traj in enumerate(whole_traj):
                    pindex = part_nb+mouse*5
                    ## Discover places where the animal is in its NEST
                    bounds_check = traj-self.bounds
                    checkarray = bounds_check>0
                    in_nest = checkarray[:,0:1]*checkarray[:,1:]
                    nest_array = np.where(in_nest)[0]
                    okay_indices = np.array([index for index in self.allowed_index_full[pindex] if index[0] not in nest_array])
                    self.allowed_index_full[pindex] = self.simple_index_maker(pindex,okay_indices[:,0:1])

        except NameError as error:
            print(error)


    def filter_speeds(self,indices,threshold = 10):
        print('filtering by speed...')
        for index in indices:
            self.filter_speed(index,threshold)
    def filter_likelihoods(self,indices):
        print('filtering by likelihood...')
        for index in indices:
            self.filter_likelihood(index)

    def reset_filters(self,indices):
        print('resetting filters...')
        for index in indices:
            self.allowed_index_full[index] = self.simple_index_maker(index,self.time_index[:,np.newaxis])
        self.filter_check_counts = [[] for i in self.part_index]
############# End of filtering functions. On to processing functions.

###
## Calculates the velocity of a mouse based on centroid:
    def velocity(self,mouse,windowlength,polyorder,filter = False):
        mouse = self.render_trajectories([mouse*5+3])[0]
        diffs = np.diff(mouse,axis = 0)
        if filter == True:
            diffs = savgol_filter(diffs,windowlength,polyorder,axis = 0)
        return diffs
    ## Determines when the velocities of the two mice are tracking each other.
    def tracking(self,windowlength = 15,polyorder = 3,filter = True):
        mice = [0,1]
        vels = []
        for mouse in mice:
            vel = self.velocity(mouse,windowlength,polyorder,filter = True)
            vels.append(vel)
        # Batch dot product
        similarity = np.sum(np.multiply(vels[0],vels[1]),axis = 1)
        return similarity

    ## Not just when mice are tracking, but
    def orderedtracking(self,windowlength = 15,polyorder = 3,filter = True):
        mice = [0,1]
        vels = []
        for mouse in mice:
            vel = self.velocity(mouse,windowlength,polyorder,filter = True)
            vels.append(vel)
        # Batch dot product
        similarity = np.sum(np.multiply(vels[0],vels[1]),axis = 1)
        # Find the average vector between their directions:
        average = np.mean(np.stack(vels),axis = 0)

        ## Give the difference in positions between the two animals
        relvec = self.relative_vector(0) ## Position of dam w.r.t. virgin.

        ## is the dam in front of the virgin or vice versa?
        pose_align = np.sum(np.multiply(average,relvec[:-1,:]),axis = 1)/(np.linalg.norm(average,axis = 1)*np.linalg.norm(relvec[:-1,:],axis = 1))

        return similarity,pose_align

    def shepherding_ethogram(self):
        similarity = self.tracking()
        points = np.where(similarity>10)[0]
        onehot = np.array([i in points for i in range(len(similarity)+1)])
        return onehot

    def nest_ethogram(self,mouse):
        try:
            nest_location = self.bounds
            ## retrieve trajectory:
            out = self.render_trajectories([mouse*5+3])[0]
            xcheck = (out[:,0]<self.bounds[0])
            ycheck = (out[:,1]<self.bounds[1])
            compound = np.logical_not(xcheck + ycheck)
        except AttributeError as error:
            print(error)

        return compound
    ## Give a full ethogram that shows the activity of both mice, and shepherding
    ## events interspersed
    def full_ethogram(self,save = False):
        ## First get the NEST ethograms of each animal:
        in_nest = []
        for mouse in [0,1]:
            in_nest.append(self.nest_ethogram(mouse))
        ## Now get the shepherding ethogram:
        shep = self.shepherding_ethogram()
        fig,ax = plt.subplots(2,1)
        names = ['Virgin','Dam']
        for mouse in [0,1]:
            b = in_nest[mouse]
            b_x = range(len(b))
            ax[mouse].plot(b_x,b,label = 'NEST')
            ax[mouse].fill_between(b_x,0,b)
            ax[mouse].plot(b_x,shep,label = 'pursuit')
            ax[mouse].set_title(names[mouse]+ ' Ethogram')
        plt.legend()
        plt.tight_layout()
        if save == True:
            plt.savefig(self.dataset_name.split('.')[0]+' Ethogram')
        plt.show()

    def part_dist(self,part0,part1):
        traj0,traj1 = self.render_trajectories([part0,part1])
        diff = traj0-traj1
        dist = np.linalg.norm(diff,axis = 1)
        return dist

    def proximity(self):
        virg,moth = self.render_trajectories([3,8])
        diff = virg-moth
        dist = np.linalg.norm(diff,axis = 1)
        return dist

    def proximity_nonest(self):
        close = np.where(self.proximity()<50)[0]
        ## Check where neither mouse is in the NEST:
        notnest = np.where((~self.nest_ethogram(0))*(~self.nest_ethogram(1)))[0]

        close_good = [i for i in range(self.dataset.shape[0]) if i in close and i in notnest]
        close_good_onehot = close_good = [i in close and i in notnest for i in range(self.dataset.shape[0])]
        return close_good_onehot

    def relative_velocity(self,mouse):
        ## First calculate velocity:
        vel = self.velocity(mouse,5,3,False)
        ## Now calculate relative position of other mouse:
        rel = self.relative_vector(mouse)
        ## calculate projection of velocity onto relative position:
        # first calculate scalar projection:
        print(vel.shape,rel.shape)
        inner_prod = np.sum(np.multiply(vel[:self.dataset.shape[0]-1,:],rel[:self.dataset.shape[0],:]),axis = 1)
        normed = inner_prod/np.linalg.norm(rel[:self.dataset.shape[0]-1,:],axis = 1)**2
        projections = normed[:,np.newaxis]*rel[:self.dataset.shape[0]-1,:]
        return projections

    def relative_speed(self,mouse):
        ## First calculate velocity:
        vel = self.velocity(mouse,5,3,False)
        ## Now calculate relative position of other mouse:
        rel = self.relative_vector(mouse)
        ## calculate projection of velocity onto relative position:
        # first calculate scalar projection:

        inner_prod = np.sum(np.multiply(vel[:self.dataset.shape[0],:],rel[:self.dataset.shape[0]-1,:]),axis = 1)
        normed = inner_prod/np.linalg.norm(rel[:self.dataset.shape[0]-1,:],axis = 1)

        return normed

    def relative_speed_normed(self,mouse):
        ## First calculate velocity:
        vel = self.velocity(mouse,5,3,False)
        ## Now calculate relative position of other mouse:
        rel = self.relative_vector(mouse)
        ## calculate projection of velocity onto relative position:
        # first calculate scalar projection:
        print(vel.shape,rel.shape)
        inner_prod = np.sum(np.multiply(vel[:self.dataset.shape[0]-1,:],rel[:self.dataset.shape[0]-1,:]),axis = 1)
        normed = inner_prod/(np.linalg.norm(rel[:self.dataset.shape[0]-1,:],axis = 1)*np.linalg.norm(vel[:self.dataset.shape[0]-1,:],axis = 1))

        return normed

    def interhead_position(self,mouse_id):
        # Returns positions between the head for a given mouse
        part_id = 5*mouse_id
        lear,rear = self.render_trajectories([part_id+1,part_id+2])

        stacked = np.stack((lear,rear))
        centroids = np.mean(stacked,axis = 0)
        return centroids
    # Vector from the center of the head to the tip.
    def head_vector(self,mouse_id):
        # First get centroid:
        head_cent = self.interhead_position(mouse_id)

        vector = self.render_trajectory_full(mouse_id*5)-head_cent
        return vector

    ## Vector from the center of the body to the center of the head.
    def body_vector(self,mouse_id):
        vector = self.render_trajectory_full(mouse_id*5)-self.render_trajectory_full(mouse_id*5+3)
        return vector

    def head_angle(self,mouse_id):
        # First get centroid:
        vector = self.head_vector(mouse_id)
        north = np.concatenate((np.ones((len(self.time_index),1)),np.zeros((len(self.time_index),1))),axis = 1)
        angles = angle_between_vec(north,vector)
        return angles

    def body_angle(self,mouse_id):
        vector = self.body_vector(mouse_id)
        north = np.concatenate((np.ones((len(self.time_index),1)),np.zeros((len(self.time_index),1))),axis = 1)
        sign = np.sign(north[:,1]-vector[:,1])
        angles = angle_between_vec(north,vector)*sign
        return angles

    # Gives the relative position of the other mouse with regard to the mouse provided in mouse_id
    def relative_vector(self,mouse_id):
        #First get centroid:
        head_cent = self.interhead_position(mouse_id)
        # Get other mouse centroid
        other_mouse = abs(mouse_id-1)
        other_mouse_centroid= self.render_trajectory_full(5*other_mouse)
        vector = other_mouse_centroid-head_cent
        return vector

    def gaze_relative(self,mouse_id):
        head_vector = self.head_vector(mouse_id)
        rel_vector = self.relative_vector(mouse_id)
        angles = angle_between_vec(head_vector,rel_vector)

        return angles
    # Import the relevant movie file
    def import_movie(self,moviepath):
        self.movie = VideoFileClip(moviepath)
#         self.movie.reader.initialize()
    # Plot the movie file, and overlay the tracked points
    def show_frame(self,frame,plotting= True):
        image = img_as_ubyte(self.movie.get_frame(frame/self.movie.fps))
        fig,ax = plt.subplots()
        ax.imshow(image[70:470,300:600])
#         ax.imshow(image)
        if plotting == True:
            self.plot_trajectory([0,1,2,3,4,5,6,7,8,9],start = frame,end = frame+1,cropx = 300,cropy = 70,axes = ax,marker = 'o')
        plt.show()

############################ Filtering Primitives
    def deviance_final(self,i,windowlength,reference,ref_pindex,target,target_pindex):
        ## We have to define all relevant index sets first. This is actually where most of the trickiness happens. We do the
        ## following: 1) define a starting point in the TARGET trajectory, by giving an index into the set of allowed axes.
        # First define the relevant indices for interpolation: return the i+1th and the windowlength+i+1th index in the test set:

        sample_indices_absolute = [i+1,windowlength+i+1]
        ref_indices = ref_pindex[:,0]

        target_indices = target_pindex[:,0]

        test_indices_sample_start = target_indices[sample_indices_absolute[0]]
        test_indices_sample_end = target_indices[sample_indices_absolute[-1]]

        ## We have to find the appropriate indices in the reference trajectory: those equal to or just outside the test indices
        start_rel_ref,end_rel_ref = np.where(ref_indices <= test_indices_sample_start)[0][-1],np.where(ref_indices >= test_indices_sample_end)[0][0]

        sample_indices_rel = ref_indices[[start_rel_ref-1,start_rel_ref,end_rel_ref-1,end_rel_ref]]

        ## Define the relevant indices for comparison in the test trajectory space:
        comp_indices_rel_test = target_indices[sample_indices_absolute[0]:sample_indices_absolute[-1]]



        ## Now we should use indices to 1) interpolate the baseline trajectory, 2) evaluate the fit of the test to the
        ## interpolation
        traj_ref = reference
        traj_test = target

        traj_ref_sample = traj_ref[[start_rel_ref-1,start_rel_ref,end_rel_ref,end_rel_ref+1],:]

        ## Create interpolation function:
        f = interp1d(sample_indices_rel,traj_ref_sample,axis = 0,kind = 'cubic')

        ## Now evaluate this at the relevant points on the test function!

        interped_points = f(comp_indices_rel_test)

        sampled_points = traj_test[sample_indices_absolute[0]:sample_indices_absolute[-1],:]
        return interped_points,sampled_points,comp_indices_rel_test

    def deviance_final_p(self,i,windowlength,reference,ref_pindex,target,target_pindex):
        ## We have to define all relevant index sets first. This is actually where most of the trickiness happens. We do the
        ## following: 1) define a starting point in the TARGET trajectory, by giving an index into the set of allowed axes.
        # First define the relevant indices for interpolation: return the i+1th and the windowlength+i+1th index in the test set:
        sample_indices_absolute = [i+1,windowlength+i+1]
        ref_indices = ref_pindex

        target_indices = target_pindex

        test_indices_sample_start = target_indices[sample_indices_absolute[0]]
        test_indices_sample_end = target_indices[sample_indices_absolute[-1]]

        ## We have to find the appropriate indices in the reference trajectory: those equal to or just outside the test indices
        start_rel_ref,end_rel_ref = np.where(ref_indices <= test_indices_sample_start)[0][-1],np.where(ref_indices >= test_indices_sample_end)[0][0]

        sample_indices_rel = ref_indices[[start_rel_ref-1,start_rel_ref,end_rel_ref-1,end_rel_ref]]

        ## Define the relevant indices for comparison in the test trajectory space:
        comp_indices_rel_test = target_indices[sample_indices_absolute[0]:sample_indices_absolute[-1]]



        ## Now we should use indices to 1) interpolate the baseline trajectory, 2) evaluate the fit of the test to the
        ## interpolation
        traj_ref = reference
        traj_test = target

        traj_ref_sample = traj_ref[[start_rel_ref-1,start_rel_ref,end_rel_ref,end_rel_ref+1],:]

        ## Create interpolation function:
        f = interp1d(sample_indices_rel,traj_ref_sample,axis = 0,kind = 'cubic')

        ## Now evaluate this at the relevant points on the test function!

        interped_points = f(comp_indices_rel_test)

        sampled_points = traj_test[sample_indices_absolute[0]:sample_indices_absolute[-1],:]
        return interped_points,sampled_points,comp_indices_rel_test


    def adjacency_matrix(self,frames,vstats,mstats,thresh = [2,7]):
        ## Iterate through the parts of each mouse pairwise, and
        ## fetch the appropriate matrix with distances:
        stats = [vstats,mstats]
        matrix = np.zeros((len(frames),10,10))
        for refmouse in range(2):
            for j in range(5):
                partref = refmouse*5+j
                for targmouse in range(2):
                    threshindex = targmouse == refmouse
                    if threshindex == 0:
                        idxvec = j+1
                    else:
                        idxvec = j
                    for i in range(idxvec):
                        parttarg = targmouse*5+i
                        dist = self.part_dist(partref,parttarg)[frames]
                        ## We want to ask if this is typical for what we would expect:
                        stats_touse = stats[refmouse]
                        if i == j:
                            mean,std = 0,15
                        else:
                            mean,std = stats_touse[(partref,refmouse*5+i)]
                        ## Logical entries:
                        #threshold is generally stricter for your own animal:
                        ## Query: is the target mouse part in line with what would be expected
                        ## given the reference mouse's skeleton? I.e. asking if a body part
                        ## fits better with the other skeleton is comparing columns of this
                        ## block diagonal matrix
                        matrix[:,partref,parttarg] = abs(dist-mean)< thresh[threshindex]*std
                        matrix[:,partref-j+i,parttarg-i+j] = matrix[:,partref,parttarg]

        return matrix

    ## The input consists of two segment sets and the trajectory data:
    ## Note that this will process the part in both mice symmetrically
    ## (self.classify_v3(pindex) == self.classify_v3(other_pindex))
    def classify_v3(self,pindex):
        mouse_nb = pindex/5
        other_mouse = abs(mouse_nb-1)
        other_pindex = int((other_mouse-mouse_nb)*5+pindex)
        pindices = [pindex,other_pindex]
        # Find trajectories:
        trajectories = self.render_trajectories(pindices)
        length = len(trajectories[0])
        # Find segments:
        self_good_indices = self.allowed_index_full[pindex][:,0]
        other_good_indices = self.allowed_index_full[other_pindex][:,0]

        ssegs = find_segments(self_good_indices)
        osegs = find_segments(other_good_indices)

        ## Have a state variable for both animals that says if we are in a trajectory or not, and which we are in
        segs = [ssegs,osegs]

        ssegind = 0
        osegind = 0
        seginds = np.array([ssegind,osegind])
        sstart,send = ssegs[ssegind][0],ssegs[ssegind][-1]
        ostart,oend = osegs[osegind][0],osegs[osegind][-1]

        starts = np.array([sstart,ostart])
        ends = np.array([send,oend])
        labels = ['virgin','dam']

        ## Initialize a shell array of nans:
        shell_array = np.empty((self.dataset.shape[0],2))
        shell_array[:] = np.nan

        ## Initialize an array to keep track of when we're in the NEST:
        nest_array = np.zeros(self.dataset.shape[0],)

        ## Keep track of last entry for both animals:
        lastentries_hist = [[],[]]
        for time in tqdm(range(length)):

            ## until we exit the current interval, we track the start and end of the current interval:
            timevec = np.repeat(time,2)
            ## If we exit the segment:
            if np.any(timevec == ends):

                exits = np.where(timevec == ends)[0]

                for exit_ind in exits:

                    starts[exit_ind] = segs[exit_ind][seginds[exit_ind]][0]
                    ends[exit_ind] = segs[exit_ind][seginds[exit_ind]][-1]

            ## If we enter the next registered segment:
            if np.any(timevec == starts):

                enters = np.where(timevec == starts)[0]

                for enter_ind in enters:
                    ## Define a start and end for this trajectory:
                    selfstart,selfend = segs[enter_ind][seginds[enter_ind]][0],segs[enter_ind][seginds[enter_ind]][-1]
                    otherstart,otherend = segs[1-enter_ind][seginds[1-enter_ind]][0],segs[1-enter_ind][seginds[1-enter_ind]][-1]

                    ## Grab the trajectory:
                    segment = segment_getter(trajectories,segs,seginds[enter_ind],enter_ind)

                    ## First compare to your own past:

                    # Find difference with past trajectory:
                    # Find last end:
                    shell_column = shell_array[:,0+enter_ind:1+enter_ind]
                    if np.all(np.isnan(shell_column)):
                        last_entry = segment[0:1,:]
                    else:
                        last_entry_inds = np.where(~np.isnan(shell_column[:,0]))[0][-1:]
                        last_entry_val = shell_column[np.where(~np.isnan(shell_column[:,0]))[0][-1:]]
                        ## Indexing trickery:
                        ind_choice = [1-enter_ind,enter_ind]
                        ## Current
                        curr_pindex = pindices[enter_ind]

                        last_entry = trajectories[ind_choice[curr_pindex == last_entry_val[0][0]]][last_entry_inds,:]
                    lastentries_hist[enter_ind] = last_entry

                    histselfdifference = np.linalg.norm(last_entry-segment[0,:])

                    threshbound = 100


                    if histselfdifference < threshbound:
                        print('accept',time)
                        ###################################################################

                        shell_array[selfstart:selfend,0+enter_ind:1+enter_ind] = pindices[enter_ind]

                        ###################################################################
                    # If it doesnt fit with the past, compare to mirroring trajectory:
                    else:
                        ## find indices that overlap with mirroring trajectory:
                        maxstart,minend = np.max(starts),np.min(ends)

                        if minend-maxstart <=0:
                            pass ## If other trajectory not valid at the time
                            print('passed',time)
                        else:
                            self = trajectories[enter_ind][maxstart:minend,:]
                            other = trajectories[1-enter_ind][maxstart:minend,:]

                            # Find difference with other trajectory
                            maxdifference = np.max(np.linalg.norm(self-other,axis = 1))
                            histotherdifference = np.linalg.norm(last_entry-other[0,:])

                            ## Three cases here:
                            ## 1) Trajectory duplicates an already existing other trajectory:
                            if histselfdifference > 3*maxdifference:
                                print('ignore',time)
                                pass ## We do not assign this segment to anyone, as it is already accounted for.
                            ## 2) Trajectory has switched with the other trajectory:
                            elif histselfdifference > 3*histotherdifference:
                                print('switch',time)

                                ###################################################################
                                shell_array[maxstart:minend,0+enter_ind:1+enter_ind] = pindices[1-enter_ind]
                                ## Additionally, if this trajectory is close to the other's ending point:
                                shell_array[maxstart:minend,0+1-enter_ind:1+1-enter_ind] = pindices[enter_ind]
                                ###################################################################
                            ## 3) Everything is fine.
                            else:
                                print('keep',time)
                                ## Measure backwards from the current segment:
                                delay_interval = selfstart-last_entry_inds

                                if delay_interval > 30:
                                    ###################################################################
                                    shell_array[selfstart:selfend,0+enter_ind:1+enter_ind] = pindices[enter_ind]
                                    ###################################################################
                                else:
                                    pass
                                    # plt.plot(np.arange(minend-maxstart)+maxstart,self,'ro',label = labels[enter_ind])
                                    # plt.plot(np.arange(minend-maxstart)+maxstart,other,'bo',label = labels[1-enter_ind])
                                    # plt.plot(np.ones((1,2))*maxstart,last_entry,'g*',markersize = 5)
                                    # plt.legend()
                                    # plt.show()

                ## We only want this to update if we're not at the last trajectory:
                    if seginds[enter_ind]!= len(segs[enter_ind])-1:
                        seginds[enter_ind] += 1
    #         if time == length -1:


    #             fig,ax = plt.subplots(2,1)
    #             ax[0].plot(shell_array[:,0:1],label = 'virginx')
    #             ax[0].plot(shell_array[:,2:3],label = 'damx')
    #             ax[1].plot(shell_array[:,1:2],label = 'virginy')
    #             ax[1].plot(shell_array[:,3:4],label = 'damy')
    #             ax[0].legend()
    #             ax[1].legend()
    #             plt.show()

        return shell_array

    def classify_v4(self,pindex):
        threshbound = 100
        mouse_nb = pindex/5
        other_mouse = abs(mouse_nb-1)
        other_pindex = int((other_mouse-mouse_nb)*5+pindex)
        pindices = [pindex,other_pindex]
        # Find trajectories:
        trajectories = self.render_trajectories(pindices)
        length = len(trajectories[0])
        # Find segments:
        self_good_indices = self.allowed_index_full[pindex][:,0]
        other_good_indices = self.allowed_index_full[other_pindex][:,0]

        ssegs = find_segments(self_good_indices)
        osegs = find_segments(other_good_indices)

        ## Have a state variable for both animals that says if we are in a trajectory or not, and which we are in
        segs = [ssegs,osegs]

        ssegind = 0
        osegind = 0
        seginds = np.array([ssegind,osegind])
        sstart,send = ssegs[ssegind][0],ssegs[ssegind][-1]
        ostart,oend = osegs[osegind][0],osegs[osegind][-1]

        starts = np.array([sstart,ostart])
        ends = np.array([send,oend])
        labels = ['virgin','dam']

        ## Initialize a shell array of nans:
        shell_array = np.empty((self.dataset.shape[0],2))
        shell_array[:] = np.nan

        ## Initialize an array to keep track of when we're in the NEST:
        nest_array = np.zeros(self.dataset.shape[0],)

        ## Keep track of last entry for both animals:
        lastentries_hist = [[],[]]
        for time in tqdm(range(length)):

            ## until we exit the current interval, we track the start and end of the current interval:
            timevec = np.repeat(time,2)
            ## If we exit the segment:
            if np.any(timevec == ends):

                exits = np.where(timevec == ends)[0]

                for exit_ind in exits:

                    starts[exit_ind] = segs[exit_ind][seginds[exit_ind]][0]
                    ends[exit_ind] = segs[exit_ind][seginds[exit_ind]][-1]

            ## If we enter the next registered segment:
            if np.any(timevec == starts):

                enters = np.where(timevec == starts)[0]

                for enter_ind in enters:
                    ## Define a start and end for this trajectory:
                    selfstart,selfend = segs[enter_ind][seginds[enter_ind]][0],segs[enter_ind][seginds[enter_ind]][-1]
                    otherstart,otherend = segs[1-enter_ind][seginds[1-enter_ind]][0],segs[1-enter_ind][seginds[1-enter_ind]][-1]

                    ## Grab the trajectory:
                    segment = segment_getter(trajectories,segs,seginds[enter_ind],enter_ind)

                    ## First compare to your own past:

                    # Find difference with past trajectory:
                    # Find last end:
                    shell_column = shell_array[:,0+enter_ind:1+enter_ind]
                    if np.all(np.isnan(shell_column)):
                        last_entry = segment[0:1,:]
                    else:
                        last_entry_inds = np.where(~np.isnan(shell_column[:,0]))[0][-1:]
                        last_entry_val = shell_column[np.where(~np.isnan(shell_column[:,0]))[0][-1:]]
                        ## Indexing trickery:
                        ind_choice = [1-enter_ind,enter_ind]
                        ## Current
                        curr_pindex = pindices[enter_ind]

                        last_entry = trajectories[ind_choice[curr_pindex == last_entry_val[0][0]]][last_entry_inds,:]
                    lastentries_hist[enter_ind] = last_entry

                    histselfdifference = np.linalg.norm(last_entry-segment[0,:])

                    if histselfdifference < threshbound:

                        ###################################################################

                        shell_array[selfstart:selfend,0+enter_ind:1+enter_ind] = pindices[enter_ind]

                        ###################################################################
                    # If it doesnt fit with the past, compare to mirroring trajectory:
                    else:

                        ## find indices that overlap with mirroring trajectory:
                        maxstart,minend = np.max(starts),np.min(ends)

                        if minend-maxstart <=0:
                            pass ## If other trajectory not valid at the time

                        else:
                            self = trajectories[enter_ind][maxstart:minend,:]
                            other = trajectories[1-enter_ind][maxstart:minend,:]

                            # Find difference with other trajectory
                            maxdifference = np.max(np.linalg.norm(self-other,axis = 1))
                            histotherdifference = np.linalg.norm(last_entry-other[0,:])
                            histcrossdifference = np.linalg.norm(lastentries_hist[1-enter_ind] - other[0,:])
                            ## Three cases here:
                            ## 1) Trajectory duplicates an already existing other trajectory:
                            if histselfdifference > 3*maxdifference:

                                pass ## We do not assign this segment to anyone, as it is already accounted for.
                            ## 2) Trajectory has switched with the other trajectory:
                            elif histselfdifference > 3*histotherdifference:

                                if histcrossdifference > threshbound:

                                    ###################################################################
                                    shell_array[maxstart:minend,0+enter_ind:1+enter_ind] = pindices[1-enter_ind]
                                    ## Additionally, if this trajectory is close to the other's ending point:
                                    shell_array[maxstart:minend,0+1-enter_ind:1+1-enter_ind] = pindices[enter_ind]
                                    ###################################################################
                                else:
                                    delay_interval = selfstart-last_entry_inds
                                    if delay_interval > 20:
                                        ###################################################################
                                        shell_array[selfstart:selfend,0+enter_ind:1+enter_ind] = pindices[enter_ind]
                                        ###################################################################
                                    else:
                                        pass
                            ## 3) Everything is fine.
                            else:

                                ## Measure backwards from the current segment:
                                delay_interval = selfstart-last_entry_inds

                                if delay_interval > 30:
                                    ###################################################################
                                    shell_array[selfstart:selfend,0+enter_ind:1+enter_ind] = pindices[enter_ind]
                                    ###################################################################
                                else:
                                    pass
                                    # plt.plot(np.arange(minend-maxstart)+maxstart,self,'ro',label = labels[enter_ind])
                                    # plt.plot(np.arange(minend-maxstart)+maxstart,other,'bo',label = labels[1-enter_ind])
                                    # plt.plot(np.ones((1,2))*maxstart,last_entry,'g*',markersize = 5)
                                    # plt.legend()
                                    # plt.show()

                ## We only want this to update if we're not at the last trajectory:
                    if seginds[enter_ind]!= len(segs[enter_ind])-1:
                        seginds[enter_ind] += 1
    #         if time == length -1:


    #             fig,ax = plt.subplots(2,1)
    #             ax[0].plot(shell_array[:,0:1],label = 'virginx')
    #             ax[0].plot(shell_array[:,2:3],label = 'damx')
    #             ax[1].plot(shell_array[:,1:2],label = 'virginy')
    #             ax[1].plot(shell_array[:,3:4],label = 'damy')
    #             ax[0].legend()
    #             ax[1].legend()
    #             plt.show()

        return shell_array

    def refine(self,out_array,pindex):

        mouse_nb = int(pindex/5)
        other_mouse = abs(mouse_nb-1)
        other_pindex = int((other_mouse-mouse_nb)*5+pindex)
        pindices = [pindex,other_pindex]
        # Find trajectories:
        trajectories = [self.select_trajectory(pind) for pind in pindices]
        v_conf = trajectories[0][np.isnan(out_array[:,0:1])[:,0],:]
        m_conf = trajectories[1][np.isnan(out_array[:,1:2])[:,0],:]
        # Make interpolating functions from them:
        vinterp,minterp = interpolate_isnans(trajectories,out_array)

        # Initialize a new output array for just the residual values with nans
        shell_array_2 = np.empty((self.dataset.shape[0],4))
        shell_array_2[:] = np.nan

        shell_array_2[np.isnan(out_array[:,0:1])[:,0],:2] = v_conf
        shell_array_2[np.isnan(out_array[:,1:2])[:,0],2:] = m_conf

        residualsmm = minterp(np.arange(self.dataset.shape[0]))-shell_array_2[:,2:]
        residualsvv = vinterp(np.arange(self.dataset.shape[0]))-shell_array_2[:,:2]
        residualsmv = minterp(np.arange(self.dataset.shape[0]))-shell_array_2[:,:2]
        residualsvm = vinterp(np.arange(self.dataset.shape[0]))-shell_array_2[:,2:]

        mm_redeemed = np.where(np.linalg.norm(residualsmm,axis = 1)<3)[0]
        mv_redeemed = np.where(np.linalg.norm(residualsmv,axis = 1)<3)[0]
        vv_redeemed = np.where(np.linalg.norm(residualsvv,axis = 1)<3)[0]
        vm_redeemed = np.where(np.linalg.norm(residualsvm,axis = 1)<3)[0]

        # out_array[vm_redeemed,:1] = pindices[-1]
        # out_array[vv_redeemed,:1] = pindices[0]
        # out_array[mv_redeemed,1:] = pindices[0]
        # out_array[mm_redeemed,1:] = pindices[-1]
        # plt.plot(residualsmm)
        # plt.show()
        # plt.plot(residualsvv)
        # plt.show()
        return out_array
#############################################################

## Define a function to analyze the adjacency matrix by its individual blocks, and decide if
## an entry is misassigned:
    def filter_crosscheck_v2(self,vstats,mstats,thresh = [2,2],indices = None):
        if indices is None:
            indices = self.time_index
        ## We analyze all entries in parallel:
        all_matrices = self.adjacency_matrix(indices,vstats,mstats,thresh)
        vself = all_matrices[:,:5,:5]
        mconf = all_matrices[:,:5,5:]
        mself = all_matrices[:,5:,5:]
        vconf = all_matrices[:,5:,:5]

        v_confidence = np.sum(vself,axis = 1)
        v_cross = np.sum(vconf,axis = 1)
        m_confidence = np.sum(mself,axis = 1)
        m_cross = np.sum(mconf,axis = 1)

        vconfusion = v_confidence-v_cross
        mconfusion = m_confidence-m_cross

        confusion = np.concatenate((vconfusion,mconfusion),axis = -1)

        return confusion

## Less harsh of a threshold: If you are close to the body part of the other animal,
## and your body parts cant intercede for you, you are removed.
    def filter_crosscheck_v3(self,vstats,mstats,thresh = [2,2],indices = None):
        if indices is None:
            indices = self.time_index
        ## We analyze all entries in parallel:
        all_matrices = self.adjacency_matrix(indices,vstats,mstats,thresh)
        ## Look for diagonals of the off-diagonal blocks
        vself = all_matrices[:,:5,:5]
        mconf = all_matrices[:,:5,5:]
        mself = all_matrices[:,5:,5:]
        vconf = all_matrices[:,5:,:5]

        v_confidence = np.sum(vself,axis = 1)
        v_cross = np.diagonal(vconf,axis1 = 1,axis2 = 2)
        m_confidence = np.sum(mself,axis = 1)
        m_cross = np.diagonal(mconf,axis1 = 1,axis2 = 2)

        # Normalize to how tight we want this threshold to be:
        vconfusion = v_confidence-2*v_cross
        mconfusion = m_confidence-2*m_cross



        confusion = np.concatenate((vconfusion,mconfusion),axis = -1)

        return confusion


    def filter_crosscheck_replaces_v2(self,parts,vstats,mstats,thresh = [2,7],indices = None):
        if indices is None:
            indices = self.time_index
        confusion = self.filter_crosscheck_v2(vstats,mstats,thresh,indices)

        for p,part in enumerate(parts):
            pconfusion = indices[np.where(confusion[:,part] <0)[0]]

            ind_before = self.allowed_index_full[part]

            okay_indices = np.array([index for index in self.allowed_index_full[part] if index[0] not in pconfusion])
            self.allowed_index_full[part] = okay_indices

    def filter_crosscheck_replaces_v3(self,parts,vstats,mstats,thresh = [2,2],indices = None):
        if indices is None:
            indices = self.time_index
        confusion = self.filter_crosscheck_v3(vstats,mstats,thresh,indices)

        for p,part in enumerate(parts):
            print(np.where(confusion[:,part]<0)[0])
            pconfusion = indices[np.where(confusion[:,part] <= 0)[0]]

            ind_before = self.allowed_index_full[part]

            okay_indices = np.array([index for index in self.allowed_index_full[part] if index[0] not in pconfusion])
            self.allowed_index_full[part] = okay_indices

    def filter_check_v2(self,pindex,windowlength,varthresh,skip):

        sample_indices = lambda i: [i,i+1,windowlength+i+1,windowlength+i+2]
        # Define the indices you want to check: acceptable indices in both the part trajectory for this animal and
        # its reference counterpoint
        mouse_nb = pindex/5

        all_vars = []
        all_outs = []

        target = self.render_trajectory_full(pindex)
        target_indices = self.allowed_index_full[pindex]

        scores = np.zeros(len(target))

        compare_max = len(target_indices)-windowlength-2
        for i in tqdm(range(compare_max)[::skip]):

            current_vars = []
            current_outs = []

            interped,sampled,indices = self.deviance_final(i,windowlength,target,target_indices,target,target_indices)
            linewise_var = np.max(np.max(abs(interped-sampled),axis = 0))
            current_vars.append(linewise_var)
            if linewise_var > varthresh:
                mis = -1*(np.max((abs(interped-sampled)>varthresh),axis = 1)*2-1)
            else:
                mis = np.ones(np.shape(interped)[0])
            scores[i+1:i+windowlength+1] += mis
        return scores
## Parallelized version of the above. Could be up to 10x faster.
    def filter_check_p(self,pindices,windowlength,varthresh,skip):

        sample_indices = lambda i: [i,i+1,windowlength+i+1,windowlength+i+2]
        # Define the indices you want to check: acceptable indices in both the part trajectory for this animal and
        # its reference counterpoint

        all_vars = []
        all_outs = []

        target = np.concatenate(self.render_trajectories(pindices),axis = 1)
        ## Make sure that no other processing has been done:
        for pindex in pindices:
            assert len(self.allowed_index_full[pindex]) == len(self.dataset.values), 'Must be done first'
        target_indices = self.time_index

        scores = np.zeros((len(target),len(pindices)))

        compare_max = len(target_indices)-windowlength-2
        for i in tqdm(range(compare_max)[::skip]):

            current_vars = []
            current_outs = []

            interped,sampled,indices = self.deviance_final_p(i,windowlength,target,target_indices,target,target_indices)
            linewise_vars = np.array([np.max(np.max(abs(interped-sampled),axis = 0)[2*column:2*(column+1)]) for column in range(len(pindices))])

            ## Where do we violate that constraint?

            for column,linewise_var in enumerate(linewise_vars):
                if linewise_var > varthresh:
                    mis = -1*(np.max((abs(interped[:,2*column:2*(column+1)]-sampled[:,2*column:2*(column+1)])>varthresh),axis = 1)*2-1)
                else:
                    mis = np.ones(np.shape(interped)[0])
                scores[i+1:i+windowlength+1,column] += mis
        return scores


    def filter_check_replace_p(self,pindices,windowlength = 45,varthresh = 40,skip = 1):
        preouts = self.filter_check_p(pindices,windowlength,varthresh,skip)
        for pindind,pindex in enumerate(pindices):
            if not np.all(self.filter_check_counts[pindex] == preouts[:,pindind]):
                self.filter_check_counts[pindex] = preouts[:,pindind]
            outs = np.where(preouts[:,pindind]<0)[0]
            if not len(outs):
                pass
            else:
                outs = np.unique(outs)
            okay_indices = np.array([index for index in self.allowed_index_full[pindex] if index[0] not in outs])
            self.allowed_index_full[pindex] = self.simple_index_maker(pindex,okay_indices[:,0:1])


    def filter_check_replace_v2(self,pindex,windowlength= 45,varthresh=40,skip=1):
        mouse_nb = pindex/5
        preouts = self.filter_check_v2(pindex,windowlength,varthresh,skip)
        ## Update in memory if not the same
        if not np.all(self.filter_check_counts[pindex] == preouts):
            self.filter_check_counts[pindex] = preouts
        outs = np.where(preouts<0)[0]
        if not len(outs):
            pass
        else:
            outs = np.unique(outs)
        okay_indices = np.array([index for index in self.allowed_index_full[pindex] if index[0] not in outs])
        self.allowed_index_full[pindex] = self.simple_index_maker(pindex,okay_indices[:,0:1])

    def filter_check_replaces_v2(self,indices,windowlength= 45,varthresh=40,skip = 1):
        for pindex in indices:
            print('checking '+ str(self.part_dict[pindex]))
            self.filter_check_replace_v2(pindex,windowlength= windowlength,varthresh=varthresh,skip=skip)

    def filter_segment_noresids(self,pindex):
        mouse_nb = pindex/5
        other_mouse = abs(mouse_nb-1)
        other_pindex = int((other_mouse-mouse_nb)*5+pindex)
        indices = [pindex,other_pindex]
        out = self.classify_v3(pindex)
        # print(sad)
        for i in range(2):
            col = out[:,i]
            good = np.where(~np.isnan(col))[0][:,None]
            col[good]
            good_inds = np.concatenate((good,col[good]),axis = 1).astype(int)
            self.allowed_index_full[indices[i]] = good_inds

    def filter_segment(self,pindex):
        mouse_nb = pindex/5
        other_mouse = abs(mouse_nb-1)
        other_pindex = int((other_mouse-mouse_nb)*5+pindex)
        indices = [pindex,other_pindex]
        preout = self.classify_v3(pindex)
        out = self.refine(preout,pindex)
        for i in range(2):
            col = out[:,i]
            good = np.where(~np.isnan(col))[0][:,None]
            col[good]
            good_inds = np.concatenate((good,col[good]),axis = 1).astype(int)
            self.allowed_index_full[indices[i]] = good_inds


    def filter_segment_v2(self,pindex):
        mouse_nb = pindex/5
        other_mouse = abs(mouse_nb-1)
        other_pindex = int((other_mouse-mouse_nb)*5+pindex)
        indices = [pindex,other_pindex]
        preout = self.classify_v4(pindex)
        out = self.refine(preout,pindex)
        for i in range(2):
            col = out[:,i]
            good = np.where(~np.isnan(col))[0][:,None]
            col[good]
            good_inds = np.concatenate((good,col[good]),axis = 1).astype(int)
            self.allowed_index_full[indices[i]] = good_inds

    def filter_segment_replaces(self,indices):
        assert len(indices)<= 5, 'symmetric in mice, dont rerun'
        for index in indices:
            self.filter_segment(index)

    def filter_segment_replaces_v2(self,indices):
        assert len(indices)<= 5, 'symmetric in mice, dont rerun'
        for index in indices:
            self.filter_segment_v2(index)

    def filter_classify(self,pindex):
        mouse_nb = pindex/5
        other_mouse = abs(mouse_nb-1)
        other_pindex = int((other_mouse-mouse_nb)*5+pindex)
        indices = [pindex,other_pindex]
        preout = self.classify_v4(pindex)
        out = self.refine(preout,pindex)
        for i in range(2):
            col = out[:,i]
            good = np.where(~np.isnan(col))[0][:,None]
            col[good]
            good_inds = np.concatenate((good,col[good]),axis = 1).astype(int)
            self.allowed_index_full[indices[i]] = good_inds

    def filter_classify_replaces(self,indices):
        assert len(indices)<= 5, 'symmetric in mice, dont rerun'
        for index in indices:
            self.filter_classify(index)


### FUNCTIONS FOR PARAMETER SEARCH ON CLASSIFY:
    def filter_classify_replaces_ps(self,indices,params):
        assert len(indices)<= 5, 'symmetric in mice, dont rerun'
        for index in indices:
            self.filter_classify_ps(index,params)

    def filter_classify_ps(self,pindex,params):
        mouse_nb = pindex/5
        other_mouse = abs(mouse_nb-1)
        other_pindex = int((other_mouse-mouse_nb)*5+pindex)
        indices = [pindex,other_pindex]
        preout = self.classify_v4_ps(pindex,params)
        out = self.refine(preout,pindex)
        for i in range(2):
            col = out[:,i]
            good = np.where(~np.isnan(col))[0][:,None]
            col[good]
            good_inds = np.concatenate((good,col[good]),axis = 1).astype(int)
            self.allowed_index_full[indices[i]] = good_inds

    def classify_v4_ps(self,pindex,params):

        threshbound = 100+np.exp(params[0])
        T0 = params[1]
        T1 = params[2]
        T2 = params[3]
        T3 = params[4]


        mouse_nb = pindex/5
        other_mouse = abs(mouse_nb-1)
        other_pindex = int((other_mouse-mouse_nb)*5+pindex)
        pindices = [pindex,other_pindex]
        # Find trajectories:
        trajectories = self.render_trajectories(pindices)
        length = len(trajectories[0])
        # Find segments:
        self_good_indices = self.allowed_index_full[pindex][:,0]
        other_good_indices = self.allowed_index_full[other_pindex][:,0]

        ssegs = find_segments(self_good_indices)
        osegs = find_segments(other_good_indices)

        ## Have a state variable for both animals that says if we are in a trajectory or not, and which we are in
        segs = [ssegs,osegs]

        ssegind = 0
        osegind = 0
        seginds = np.array([ssegind,osegind])
        sstart,send = ssegs[ssegind][0],ssegs[ssegind][-1]
        ostart,oend = osegs[osegind][0],osegs[osegind][-1]

        starts = np.array([sstart,ostart])
        ends = np.array([send,oend])
        labels = ['virgin','dam']

        ## Initialize a shell array of nans:
        shell_array = np.empty((self.dataset.shape[0],2))
        shell_array[:] = np.nan

        ## Initialize an array to keep track of when we're in the NEST:
        nest_array = np.zeros(self.dataset.shape[0],)

        ## Keep track of last entry for both animals:
        lastentries_hist = [[],[]]
        for time in tqdm(range(length)):

            ## until we exit the current interval, we track the start and end of the current interval:
            timevec = np.repeat(time,2)
            ## If we exit the segment:
            if np.any(timevec == ends):

                exits = np.where(timevec == ends)[0]

                for exit_ind in exits:

                    starts[exit_ind] = segs[exit_ind][seginds[exit_ind]][0]
                    ends[exit_ind] = segs[exit_ind][seginds[exit_ind]][-1]

            ## If we enter the next registered segment:
            if np.any(timevec == starts):

                enters = np.where(timevec == starts)[0]

                for enter_ind in enters:
                    ## Define a start and end for this trajectory:
                    selfstart,selfend = segs[enter_ind][seginds[enter_ind]][0],segs[enter_ind][seginds[enter_ind]][-1]
                    otherstart,otherend = segs[1-enter_ind][seginds[1-enter_ind]][0],segs[1-enter_ind][seginds[1-enter_ind]][-1]

                    ## Grab the trajectory:
                    segment = segment_getter(trajectories,segs,seginds[enter_ind],enter_ind)

                    ## First compare to your own past:

                    # Find difference with past trajectory:
                    # Find last end:
                    shell_column = shell_array[:,0+enter_ind:1+enter_ind]
                    if np.all(np.isnan(shell_column)):
                        last_entry = segment[0:1,:]
                    else:
                        last_entry_inds = np.where(~np.isnan(shell_column[:,0]))[0][-1:]
                        last_entry_val = shell_column[np.where(~np.isnan(shell_column[:,0]))[0][-1:]]
                        ## Indexing trickery:
                        ind_choice = [1-enter_ind,enter_ind]
                        ## Current
                        curr_pindex = pindices[enter_ind]

                        last_entry = trajectories[ind_choice[curr_pindex == last_entry_val[0][0]]][last_entry_inds,:]
                    lastentries_hist[enter_ind] = last_entry

                    histselfdifference = np.linalg.norm(last_entry-segment[0,:])

                    if histselfdifference < threshbound:

                        ###################################################################

                        shell_array[selfstart:selfend,0+enter_ind:1+enter_ind] = pindices[enter_ind]

                        ###################################################################
                    # If it doesnt fit with the past, compare to mirroring trajectory:
                    else:

                        ## find indices that overlap with mirroring trajectory:
                        maxstart,minend = np.max(starts),np.min(ends)

                        if minend-maxstart <=0:
                            pass ## If other trajectory not valid at the time

                        else:
                            self = trajectories[enter_ind][maxstart:minend,:]
                            other = trajectories[1-enter_ind][maxstart:minend,:]

                            # Find difference with other trajectory
                            maxdifference = np.max(np.linalg.norm(self-other,axis = 1))
                            histotherdifference = np.linalg.norm(last_entry-other[0,:])
                            histcrossdifference = np.linalg.norm(lastentries_hist[1-enter_ind] - other[0,:])
                            ## Three cases here:
                            ## 1) Trajectory duplicates an already existing other trajectory:
                            if histselfdifference > T0*maxdifference:

                                pass ## We do not assign this segment to anyone, as it is already accounted for.
                            ## 2) Trajectory has switched with the other trajectory:
                            elif histselfdifference > T1*histotherdifference:

                                if histcrossdifference > threshbound:

                                    ###################################################################
                                    shell_array[maxstart:minend,0+enter_ind:1+enter_ind] = pindices[1-enter_ind]
                                    ## Additionally, if this trajectory is close to the other's ending point:
                                    shell_array[maxstart:minend,0+1-enter_ind:1+1-enter_ind] = pindices[enter_ind]
                                    ###################################################################
                                else:
                                    delay_interval = selfstart-last_entry_inds
                                    if delay_interval > T2:
                                        ###################################################################
                                        shell_array[selfstart:selfend,0+enter_ind:1+enter_ind] = pindices[enter_ind]
                                        ###################################################################
                                    else:
                                        pass
                            ## 3) Everything is fine.
                            else:

                                ## Measure backwards from the current segment:
                                delay_interval = selfstart-last_entry_inds

                                if delay_interval > T3:
                                    ###################################################################
                                    shell_array[selfstart:selfend,0+enter_ind:1+enter_ind] = pindices[enter_ind]
                                    ###################################################################
                                else:
                                    pass
                                    # plt.plot(np.arange(minend-maxstart)+maxstart,self,'ro',label = labels[enter_ind])
                                    # plt.plot(np.arange(minend-maxstart)+maxstart,other,'bo',label = labels[1-enter_ind])
                                    # plt.plot(np.ones((1,2))*maxstart,last_entry,'g*',markersize = 5)
                                    # plt.legend()
                                    # plt.show()

                ## We only want this to update if we're not at the last trajectory:
                    if seginds[enter_ind]!= len(segs[enter_ind])-1:
                        seginds[enter_ind] += 1
    #         if time == length -1:


    #             fig,ax = plt.subplots(2,1)
    #             ax[0].plot(shell_array[:,0:1],label = 'virginx')
    #             ax[0].plot(shell_array[:,2:3],label = 'damx')
    #             ax[1].plot(shell_array[:,1:2],label = 'virginy')
    #             ax[1].plot(shell_array[:,3:4],label = 'damy')
    #             ax[0].legend()
    #             ax[1].legend()
    #             plt.show()

        return shell_array

## Now make an end-user version that packages the above together. The only input
## we take will be the mouse statistics to compare against:
    def filter_full(self,vstats,mstats):
        partinds = [i for i in range(10)]
        ## First reset everything for consistent behavior:
        self.reset_filters(partinds)
        ## Now apply velocity segmentation:
        print('applying temporal segmentation')
        self.filter_check_replace_p(partinds,
                                    windowlength = 45,
                                    varthresh = 38,
                                    skip = 1)
        ## Now apply part filters:
        print('applying spatial segmentation')
        self.filter_crosscheck_replaces_v2(partinds,vstats,mstats,thresh = [2,2])

        ## Use parameters found through optimization:
        print('applying matching and interpolation')
        new_params = np.array([np.log(50),3.09,3.09,40.9,40.9]) ## Found by nelder-mead

        ## Now apply classification, interpolation, etc to the result:
        self.filter_classify_replaces_ps([i for i in range(5)],new_params)
        print('done.')

################################ Filter visualization functions:
    ## Show what the filters are working with:
    def filter_check_parttrace(self,pindex,start,end,windowlength=45,varthresh=40):
        mouse_nb = pindex/5
        other_mouse = abs(mouse_nb-1)
        other_pindex = int((other_mouse-mouse_nb)*5+pindex)
        t,r = self.render_trajectories([pindex,other_pindex])
        # Save some time by loading this quantity in from memory:
        if not len(self.filter_check_counts[pindex]):
            trace = self.filter_check_v2(pindex,windowlength,varthresh)
            self.filter_check_counts[pindex] = trace
        else:
            trace = self.filter_check_counts[pindex]
        t_cropped = t[start:end,:]
        r_cropped = r[start:end,:]
        trace_cropped = trace[start:end]
        fig,ax = plt.subplots(3,1,sharex = True)
        ax[0].plot(t_cropped)
        ax[0].set_title('Target')
        ax[1].plot(trace_cropped)
        ax[1].set_title('Score')
        ax[2].plot(r_cropped)
        ax[2].set_title('Cross')
        plt.show()

    def filter_crosscheck_parttrace(self,pindex,start,end,vstats,mstats,thresh=[5,2]):
        mouse_nb = pindex/5
        other_mouse = abs(mouse_nb-1)
        other_pindex = int((other_mouse-mouse_nb)*5+pindex)
        t,r = self.render_trajectories([pindex,other_pindex])
        fulltrace = self.filter_crosscheck_v2(vstats,mstats,thresh)
        trace = fulltrace[:,pindex]
        t_cropped = t[start:end,:]
        r_cropped = r[start:end,:]
        trace_cropped = trace[start:end]
        fig,ax = plt.subplots(3,1,sharex = True)
        ax[0].plot(t_cropped)
        ax[0].set_title('Target')
        ax[1].plot(trace_cropped)
        ax[1].set_title('Score')
        ax[2].plot(r_cropped)
        ax[2].set_title('Cross')
        plt.show()

    def filter_check_image(self,frame,pindex,windowlength,varthresh):
        all_parts = [i for i in range(10)]
        fig = plt.figure(figsize=(10,10))

        ax0 = plt.subplot2grid((3, 3), (0, 0))
        ax1 = plt.subplot2grid((3, 3), (0, 1))
        ax2 = plt.subplot2grid((3, 3), (0, 2))
        ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=3)

        img_axes = [ax0,ax1,ax2]
        frameoffset = [-1,0,1]
        for i,imax in enumerate(img_axes):
            self.plot_image_compare([pindex],frame+frameoffset[i],figureparams = (fig,imax))
        if not len(self.filter_check_counts[pindex]):
            trace = self.filter_check_v2(pindex,windowlength,varthresh)
            self.filter_check_counts[pindex] = trace
        else:
            trace = self.filter_check_counts[pindex]

        ## derive the point the window should start at:
        fstart = frame - windowlength/2
        target = self.render_trajectory_full(pindex)
        target_indices = self.allowed_index_full[pindex]
        fpoints,spoints,indices = self.deviance_final(fstart,windowlength,target,target_indices,target,target_indices)

        ax1.plot(spoints[:,0],spoints[:,1],'bo',markersize = 1)
        ax1.plot(fpoints[:,0],fpoints[:,1],color = 'blue')
        ax3.plot(trace[frame-50:frame+50])
        ax3.axvline(51,color = 'black')
        ax1.set_title('Local Parameters')
        plt.show()

    def filter_crosscheck_image(self,frame,part,vstats,mstats,thresh):
        all_parts = [i for i in range(10)]
        fig = plt.figure(figsize=(10,10))

        ax0 = plt.subplot2grid((3, 3), (0, 0))
        ax1 = plt.subplot2grid((3, 3), (0, 1))
        ax2 = plt.subplot2grid((3, 3), (0, 2))
        ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=3)

        img_axes = [ax0,ax1,ax2]
        frameoffset = [-1,0,1]
        for i,imax in enumerate(img_axes):
            self.plot_image_compare(all_parts,frame+frameoffset[i],figureparams = (fig,imax))
        all_traces = self.filter_crosscheck_v2(vstats,mstats,thresh = thresh)
        ax3.plot(all_traces[frame-50:frame+50,part])
        ax3.axvline(51,color = 'black')
        ax1.set_title('Local Parameters')
        plt.show()

################################
    def robust_intervals(self,indices,structure):
        buffered = binary_dilation(indices,structure)

        ## Now find intervals of uninterestingness:
        diff = np.diff(buffered.astype(int))
        intervals = []
        intend = None
        intstart = None
        for i in range(len(diff))[::-1]:
            value = diff[i]

            if value == -1:
                intend = i
            elif value == 1:
                ## Account for corner case
                if intend is not None:
                    intstart = i
                else:
                    intstart = i
                    intend = len(diff)
            if i == 0 and intend is not None:
                intstart = 0
            if intend is not None and intstart is not None:
                interval = [intstart,intend]
                intervals.append(interval)
                intend = None
                intstart = None
        return intervals

### Visualization functions (I know there are more up there that I should move down.)
## This function takes as input a one-hot vector that determines points of
## interest in the video. a buffer on each end can also help make this more visually
## pleasing.
    def highlights_reel(self,name,indices,buffer = 5):
        ## First buffer the index vector:
        structure = np.ones(buffer*2+1,)
        buffered = binary_dilation(indices,structure)
        uninteresting = ~buffered
        ## Now find intervals of uninterestingness:
        diff = np.diff(uninteresting.astype(int))
        intervals = []
        intend = None
        intstart = None
        for i in range(len(diff))[::-1]:
            value = diff[i]

            if value == -1:
                intend = i
            elif value == 1:
                ## Account for corner case
                if intend is not None:
                    intstart = i
                else:
                    intstart = i
                    intend = len(diff)
            if i == 0 and intend is not None:
                intstart = 0
            if intend is not None and intstart is not None:
                interval = [intstart,intend]
                intervals.append(interval)
                intend = None
                intstart = None

        print(intervals)
        ## Now cut these parts out of the clip:
        cutclip = self.movie
        for interval in intervals:
            cutclip = cutclip.cutout(interval[0]/cutclip.fps,interval[1]/cutclip.fps)
        cutclip.write_videofile(name+'.mp4',codec= 'mpeg4',bitrate = '1000k')


    def return_cropped_view(self,mice,frame,radius = 64):
        mouse_views = []
        image = img_as_ubyte(self.movie.get_frame((frame)/self.movie.fps))
        for mouse in mice:
            ## Image Plots:
            all_cents = self.render_trajectory_full(mouse*5+3)
            xcent,ycent = all_cents[frame,0],all_cents[frame,1]
    #         if (xcent < 550) and (ycent<400):

            xsize,ysize = self.movie.size
            xmin,xmax,ymin,ymax = int(xcent-radius),int(xcent+radius),int(ycent-radius),int(ycent+radius)
            ## do edge detection:
            pads  = np.array([[ymin - 0,ysize - ymax],[xmin - 0,xsize - xmax],[0,0]])


            clip = image[ymin:ymax,xmin:xmax]
            if np.any(pads < 0):
        #         print('ehre')
                topad = pads<0
                padding = -1*pads*topad
                clip = np.pad(clip,padding,'edge')
            mouse_views.append(clip)
        return mouse_views

    def return_cropped_view_rot(self,mice,frame,angle,radius = 64):
        mouse_views = []
        image = img_as_ubyte(self.movie.get_frame((frame)/self.movie.fps))
        for mouse in mice:
            ## Image Plots:
            all_cents = self.render_trajectory_full(mouse*5+3)
            xcent,ycent = all_cents[frame,0],all_cents[frame,1]
    #         if (xcent < 550) and (ycent<400):

            xsize,ysize = self.movie.size
            buffer = np.ceil(np.sqrt(2))
            xmin,xmax,ymin,ymax = int(xcent-buffer*radius),int(xcent+buffer*radius),int(ycent-buffer*radius),int(ycent+buffer*radius)
            ## do edge detection:
            pads  = np.array([[ymin - 0,ysize - ymax],[xmin - 0,xsize - xmax],[0,0]])


            clip = image[ymin:ymax,xmin:xmax]
            if np.any(pads < 0):
        #         print('ehre')
                topad = pads<0
                padding = -1*pads*topad
                clip = np.pad(clip,padding,'edge')

            clip_rot = rotate(clip,angle,reshape = False)
            print(clip_rot.shape)
            xminf,xmaxf,yminf,ymaxf = int(buffer*radius-radius),int(buffer*radius+radius),int(buffer*radius-radius),int(buffer*radius+radius)
            clip_final = clip_rot[yminf:ymaxf,xminf:xmaxf]
            mouse_views.append(clip_final)
        return mouse_views

    # Return video of trajectories in polar coordinates relative to center:
    def render_trajectories_polar(self,start=0,stop=-1,to_render= None,save = False):
        ## Gaze angles:
        angles0 = self.gaze_relative(0)[1305:2820]
        angles1 = self.gaze_relative(1)[1305:2820]
        angles_together = [angles0,angles1]
        gaze0 = np.where(angles0< 0.1)[0]
        gaze1 = np.where(angles1< 0.1)[0]
        gaze0_ints = [[gaze0_val-1,gaze0_val+1] for gaze0_val in gaze0]
        gaze1_ints = [[gaze1_val-1,gaze1_val+1] for gaze1_val in gaze1]
        gaze_ints = [gaze0_ints,gaze1_ints]
        pindices = [3,8]
        rmax = 70
        all_thetas = []
        all_rs = []
        all_cents = []
        for mouse_number,pindex in enumerate(pindices):
            mouse_thetas = []
            mouse_rs = []
            if to_render == None:
                to_render = [index for index in self.part_index if index != pindex]

            reference_traj = self.render_trajectory_full(pindex)

            for other in to_render:
                part_traj = self.render_trajectory_full(other)
                relative = part_traj-reference_traj
                # Calculate r:
                rs = np.linalg.norm(relative,axis = 1)
                rs[rs>rmax] = rmax
                north = np.concatenate((np.ones((len(self.time_index),1)),np.zeros((len(self.time_index),1))),axis = 1)
                angles = angle_between_vec(north,relative)
                sign = np.sign(north[:,1]-relative[:,1])
                mouse_thetas.append(angles*sign)
                mouse_rs.append(rs)

            all_thetas.append(mouse_thetas)
            all_rs.append(mouse_rs)
        names = ['Virgin','Dam']
        colors = ['red','blue']
        for frame in range(1500):
            f = plt.figure(figsize = (20,20))
            ax0 = f.add_axes([0.35, 0.65, 0.25, 0.25], polar=True)
            ax1 = f.add_axes([0.7, 0.65, 0.25, 0.25], polar=True)
            ax = [ax0,ax1]
            ax_im0 = f.add_axes([0.35,0.35,0.25,0.25])
            ax_im1 = f.add_axes([0.7,0.35,0.25,0.25])
            ax_im = [ax_im0,ax_im1]
            ax_gaze = f.add_axes([0.35,0.1,0.6,0.20])
#             ax_full0 = f.add_axes([0.15,0.2,0.30,0.30])
            ax_full = f.add_axes([0.05,0.6,0.20,0.25])
            ax_enclosure = f.add_axes([0.05,0.2,0.20,0.30])
#             ax_full.axis('off')
#             ax_full = [ax_full0,ax_full1]
#             fig,ax = plt.subplots(2,2,subplot_kw=dict(projection='polar'))
#             slices = [np.range(9)[0:4],np.range(9)[5:9]]
            for mouse in range(2):
            ## Polar Plots:
                eff_mouse = [0]*(5-mouse)+[1]*(5-(1-mouse))
                for part in range(9):

                    ax[mouse].plot(all_thetas[mouse][part][start+frame:stop+frame],all_rs[mouse][part][start+frame:stop+frame],color = colors[eff_mouse[part]],linewidth = 2.,marker = 'o',markersize = 0.1)
                    ax[mouse].plot(all_thetas[mouse][part][stop+frame],all_rs[mouse][part][stop+frame],'o',color = colors[eff_mouse[part]])
                face_indices = mouse*4+np.array([0,1,2,0])

                ax[mouse].plot([all_thetas[mouse][face][stop+frame] for face in face_indices],[all_rs[mouse][face][stop+frame] for face in face_indices],color = colors[mouse],linewidth = 3.)
                body_index = mouse*4

                ax[mouse].plot([all_thetas[mouse][body_index][stop+frame],0],[all_rs[mouse][body_index][stop+frame],0],color = colors[mouse],linewidth = 3.)
                tail_index = mouse*5+3
                ax[mouse].plot([all_thetas[mouse][tail_index][stop+frame],0],[all_rs[mouse][tail_index][stop+frame],0],color = colors[mouse],linewidth = 3.)
                ax[mouse].set_rmax(rmax)
                ax[mouse].set_yticklabels([])
                ax[mouse].set_title(names[mouse]+' Centered Behavior',fontsize = 18)

                ## Image Plots:
                all_cents = self.render_trajectory_full(mouse*5+3)
                xcent,ycent = all_cents[stop+frame,0],all_cents[stop+frame,1]
                print(all_cents[mouse].shape)
        #         if (xcent < 550) and (ycent<400):
                print(xcent,ycent)
                radius = 50
                xsize,ysize = self.movie.size
                xmin,xmax,ymin,ymax = int(xcent-radius),int(xcent+radius),int(ycent-radius),int(ycent+radius)
                ## do edge detection:
                pads  = np.array([[ymin - 0,ysize - ymax],[xmin - 0,xsize - xmax],[0,0]])
                print(xmin,xmax,ymin,ymax)
                image = img_as_ubyte(self.movie.get_frame((stop+frame)/self.movie.fps))
                clip = image[ymin:ymax,xmin:xmax]
                if np.any(pads < 0):
            #         print('ehre')
                    topad = pads<0
                    padding = -1*pads*topad
                    clip = np.pad(clip,padding,'edge')

                ax_im[mouse].imshow(clip)
                ax_im[mouse].axis('off')
                ax_im[mouse].set_title(names[mouse]+' Zoomed View',fontsize = 18)
                ## Now do gaze detection:
                ax_gaze.plot(angles_together[mouse],color = colors[mouse],label = names[mouse])
#                 ax_gaze.plot(angles1,color = colors[mouse])
                ax_gaze.set_title('Relative Angle of Posture',fontsize = 18)
                ax_gaze.set_ylabel('Angle from Mouse Centroid (Absolute Radians)')
                ax_gaze.set_xlabel('Frame Number')
                ax_gaze.axvline(x = frame,color = 'black')
                ax_gaze.legend()
                for interval in gaze_ints[mouse]:
                    ax_gaze.axvspan(interval[0],interval[1],color =colors[mouse],alpha = 0.2)

                ax_full.plot(xcent,ycent,'o',color = colors[mouse])
                ax_full.set_yticklabels([])
                ax_full.set_xticklabels([])
                ax_full.plot(all_cents[start+frame-100:stop+frame,0],-all_cents[start+frame-100:stop+frame,1],color = colors[mouse],linewidth = 2.)
            ax_full.set_xlim(300,600)
            ax_full.set_ylim(-470,-70)
            ax_full.set_title('Physical Position',fontsize = 18)
            ax_enclosure.imshow(image[70:470,300:600])
            ax_enclosure.axis('off')
            ax_enclosure.set_title('Raw Video',fontsize = 18)
            plt.tight_layout()
            if save != False:
                plt.savefig('testframe%s.png'%frame)
                plt.close()
            else:
                plt.show()
                plt.close()
        if save != False:
            subprocess.call(['ffmpeg', '-framerate', str(10), '-i', 'testframe%s.png', '-r', '30','Angle_simmotion_'+'.mp4'])
#             for filename in glob.glob('testframe*.png'):
#                 os.remove(filename)
