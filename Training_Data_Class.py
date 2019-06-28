'''
A dataset class to hold training data, specifically. Useful to calculate
statistics on the training data to consider as ground truth going forwards.
'''

import numpy as np
import scipy
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os,sys
import tqdm


class training_dataset(object):
    def __init__(self,datapath,additionalpath):
        self.data = pd.read_hdf(datapath)
        self.dataname = datapath.split('.h5')[0]
        self.scorer = 'Taiga'
        self.part_list = ['vtip','vlear','vrear','vcent','vtail','mtip','mlear','mrear','mcent','mtail']
        self.part_index = np.arange(len(self.part_list))
        self.part_dict = {index:self.part_list[index] for index in range(len(self.part_list))}
        self.size = self.data.shape[0]
        self.datamapping = self.datasets_indices()
        self.xmin = 0
        self.xmax = 0
        self.ymin = 0
        self.ymax = 0
        self.additionalpath = additionalpath

## Correctly crop the window to match up detected points with cropped frames:
    def set_cropping(self,xmin,xmax,ymin,ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

## Gives indices for different datasets
    def datasets_indices(self):
        allpaths = self.get_imagenames(range(self.size))
        ## Find unique folder names:
        datafolders = [datafolder.split('/')[-2] for datafolder in allpaths]
        unique = list(set(datafolders))

        ## Return separate index sets for each folder:
        datamapping = {}
        for folder in unique:
            folderdata = np.array([i for i in range(self.size) if datafolders[i] in folder])
            datamapping[folder] = folderdata
        return datamapping

## Atomic actions: selecting entries of training data.
    def get_positions(self,indices,part):
        part_id = self.part_dict[part]
        position = self.data[self.scorer][part_id].values[indices,:]
        return position

    def get_imagenames(self,indices):
        ids = self.data.index.tolist()
        relevant = [id for i,id in enumerate(ids) if i in indices]
        return relevant

## Define skeleton statistic functions:
    def distances(self,indices,part0,part1):
        positions0 = self.get_positions(indices,part0)
        positions1 = self.get_positions(indices,part1)
        dists = np.linalg.norm(positions0-positions1,axis = 1)
        return dists

    def distances_mean(self,indices,part0,part1):
        dists = self.distances(indices,part0,part1)
        mean = np.nanmean(dists)
        return mean

    def distances_std(self,indices,part0,part1):
        dists = self.distances(indices,part0,part1)
        mean = np.nanstd(dists)
        return mean

    def distances_hist(self,indices,part0,part1,bins=None):
        dists = self.distances(indices,part0,part1)
        dists = dists[~np.isnan(dists)]
        if bins is not None:
            hist,edges = np.histogram(dists,bins)
        else:
            hist,edges = np.histogram(dists)
        return hist,edges

## Define iteration over all pairwise for a mouse:
    def distances_wholemouse(self,indices,mouse):
        id_0 = np.arange(5)+mouse
        pairwise_dists = {}
        for p,j in enumerate(id_0):
            for i in id_0[:p]:
                dist = self.distances(indices,j,i)
                pairwise_dists[(j,i)] = dist
        return pairwise_dists

## Define iteration over all pairwise for a mouse:
    def stats_wholemouse(self,indices,mouse):
        id_0 = np.arange(5)+5*mouse
        pairwise_dists = {}
        for p,j in enumerate(id_0):
            for i in id_0[:p]:
                mean = self.distances_mean(indices,j,i)
                std = self.distances_std(indices,j,i)
                pairwise_dists[(j,i)] = (mean,std)
        return pairwise_dists

    def hists_wholemouse(self,indices,mouse,bins = None):
        id_0 = np.arange(5)+5*mouse
        pairwise_hists = {}
        for p,j in enumerate(id_0):
            for i in id_0[:p]:
                hist = self.distances_hist(indices,j,i,bins)
                pairwise_hists[(j,i)] = hist
        return pairwise_hists

## Likewise for both mice, for a single dataset:
    def distances_dataset(self,dataset):
        indices = self.datamapping[dataset]
        outmice = []
        for mouse in range(2):
            out = self.distances_wholemouse(indices,mouse)
            outmice.append(out)
        return outmice

    def stats_dataset(self,dataset):
        indices = self.datamapping[dataset]
        outmice = []
        for mouse in range(2):
            out = self.stats_wholemouse(indices,mouse)
            outmice.append(out)
        return outmice

    def hists_dataset(self,dataset,bins = None):
        indices = self.datamapping[dataset]
        outmice = []
        for mouse in range(2):
            out = self.hists_wholemouse(indices,mouse,bins)
            outmice.append(out)
        return outmice


## Likewise for both mice, for all datapoints:
    def distances_all(self):
        indices = np.arange(self.size)
        outmice = []
        for mouse in range(2):
            out = self.distances_wholemouse(indices,mouse)
            outmice.append(out)
        return outmice

    def stats_all(self):
        indices = np.arange(self.size)
        outmice = []
        for mouse in range(2):
            out = self.stats_wholemouse(indices,mouse)
            outmice.append(out)
        return outmice

    def hists_all(self,bins = None):
        indices = np.arange(self.size)
        outmice = []
        for mouse in range(2):
            out = self.hists_wholemouse(indices,mouse,bins)
            outmice.append(out)
        return outmice

## Done with training data statistics. Now consider image statistic functions:
## We assume that if all of the data folders are not subfolders in the current
## directory, that they are at least packaged together.

    def get_images(self,indices):
        imagenames = self.get_imagenames(indices)
        ## Check if the images are somewhere else:
        if self.additionalpath is None:
            pass
        else:
            imagenames = [self.additionalpath+img for img in imagenames]

        ## Now we will load the images:
        images = [plt.imread(image) for image in imagenames]

        return images

    def make_patches(self,indices,part,radius,):
        points = self.get_positions(indices,part)
        xcents,ycents = points[:,0],points[:,1]
        images = self.get_images(indices)
        all_clipped = np.zeros((len(indices),2*radius,2*radius,3)).astype(np.uint8)
        for i,image in enumerate(images):
            ysize,xsize = image.shape[:2]

            xcent,ycent = xcents[i]-self.xmin,ycents[i]-self.ymin

            xmin,xmax,ymin,ymax = int(xcent-radius),int(xcent+radius),int(ycent-radius),int(ycent+radius)
            ## do edge detection:
            pads  = np.array([[ymin - 0,ysize - ymax],[xmin - 0,xsize - xmax],[0,0]])


            clip = image[ymin:ymax,xmin:xmax]

            # print(clip,'makedocip')
            if np.any(pads < 0):
                topad = pads<0
                padding = -1*pads*topad
                clip = np.pad(clip,padding,'edge')

            all_clipped[i,:,:,:] = (np.round(255*clip)).astype(np.uint8)
        return all_clipped
    ## Calculate image histograms over all frames
    def patch_grandhist(self,frames,part,radius):
        dataarray = self.make_patches(frames,part,radius)
        hists = [np.histogram(dataarray[:,:,:,i],bins = np.linspace(0,255,256)) for i in range(3)]
        return hists

    ## Calculate image histograms over each frame
    def patch_hist(self,frames,part,radius):
        dataarray = self.make_patches(frames,part,radius)
        hists = [[np.histogram(dataarray[f,:,:,i],bins = np.linspace(0,255,256)) for i in range(3)]for f in range(len(frames))]
        return hists
