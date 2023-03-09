from minisom import MiniSom, asymptotic_decay
import xarray as xr
import cftime
import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pylab as plt
from time import time
import os
import glob
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
import subprocess
from pathlib import Path
import csv
import datetime
import pickle
import sys
import json5


"""
The blosSOM class trains a SOM and creates several useful visualizations.
"""
class blossom:
    """
    """
    winmap = [[],[]]
    
    """
    An array of composite data for each SOM node.
    """
    comp_data =[[],[]]
   
    
    """
    Latitude domain of the SOM.
    """
    lats = []
    
    """
    Longitude domain of the SOM.
    """
    lons = []
    
    """
    Name of the variable we're dealing with. 
    """
    varName = ""
    
    """
    Topographic error value
    """
    t_err = 0
    
    """
    Quantization error value
    """
    q_err = 0
    
    """
    Initializes the class.
    param da: the data to train the SOM with
    param rows: the number of rows in the SOM
    param cols: the number of columns in the SOM
    param save_som: true if you want to save SOM composites as .nc
    """
    def __init__(self, da, rows, cols):
        self._da = da
        self._lats = da['lat'].values
        self._lons = da['lon'].values
        self._rows = rows
        self._cols = cols
        self._varName = ""
        comp_data = np.empty((rows, cols),dtype=object)
        self._comp_data = comp_data
        
        
    @property
    def comp_data(self):
        return self._comp_data
    
    """
    Gets the number of rows
    return rows: the number of rows
    """    
    @property
    def rows(self):
        return self._rows
    
    """
    Gets the number of columns
    return cols: the number of rows
    """    
    @property
    def cols(self):
        return self._cols
    
    """
    Gets the data array used to train the SOM
    return da: the data 
    """ 
    @property
    def da(self):
        return self._da
    
    """
    Sets the data to train the SOM
    param da: the data
    """
    @da.setter
    def da(self, da):
        self.da = da
        
    """
    Sets the number of SOM rows.
    param rows: the number of rows in the SOM
    """    
    @rows.setter
    def rows(self, som_grid_rows):
        self.som_grid_rows = som_grid_rows
    
    """
    Sets the number of SOM columns.
    param cols: the number of columns in the SOM
    """
    @cols.setter
    def cols(self, som_grid_columns):
        self.som_grid_columns = som_grid_columns
        
    """
    Saves the composites created to netCDF format.
    """
    def save_SOM(self):
        mapnum = 0
        for x in range(0,self.rows):
            for y in range(0,self.cols):
                #units = self.da.attrs["units"]
                import xarray as xr
                x_data = xr.DataArray(self.comp_data[x][y], coords=[self.da['lat'].values, self.da['lon'].values], dims=['lat', 'lon'])
                x_data.rename(self.varName)
                x_data.to_netcdf(path = os.getcwd() + "/composite_" + str(mapnum) + ".nc", mode = 'w',format = "NETCDF4")
                mapnum += 1
        
    """
    Gets an array containing the indices of each case based on how they were
    sorted in the original data array. So, if you want to get a case, get values
    from your data array at one of the indices stored in winmap.
    param somRow: the row in the SOM lattice
    param somCol: the column in the SOM lattice
    return cases: an array containing case indices for the given SOM node
    """
    def get_cases(self, somRow, somCol):
        return self.winmap[somRow,somCol]
        
        
    """
    Finds which SOM node the case index is associated with.
    param case: the index of a case
    return data: the composite data for the case's node
    """
    def find_case_node(self,case):
        winmap = self.winmap
        for i in range(self.rows):
            for j in range(self.cols):
                for k in range(len(winmap[i,j])):
                    if winmap[i,j][k] == case:
                        temprow = i
                        tempcol = j
                        break
        
        data = self.da[temprow, tempcol]
        return [temprow, tempcol]        
              
    
    """
    Trains a SOM with the data given and generates the following visualizations:
        1. A distance map of weights. Each cell is the normalised sum of the distances between
        a neuron and its neighbours.
        2. Distribution of data across the SOM lattice. The markers are randomly plotted across each node.
        3. Data frequency across the SOM lattice. 
    return som: the SOM generated. It can be saved using pickle.
    """
    def make_SOM(self):

        self.varName = self.da.attrs['standard_name'] 
        """
        Function for normalizing data prior to training using z-score (MM)
        param data: the data
        returns: the normalized data.
        """
        def normalize_data(data):
            return (data - np.nanmean(data)) / np.nanstd(data)

        """
        SOM training user settings
        """
        decay_function = ""
        myVars = vars()
        #Open the blossom_settings configuration file and read from it
        f = open(file = os.getcwd() + "/blossom_settings.json5", mode = "r")
        settings_str = f.read()
        f.close()
        settings = json5.loads(settings_str)
        
        learning_rate = settings["learning_rate"]
        
        decay_function = asymptotic_decay
        
        neighborhood_function = settings["neighborhood_function"]
        
        topology = settings["topology"]
        
        activation_distance = settings["activation_distance"]
        
        random_seed = settings["random_seed"]
        
        num_iteration = settings["num_iteration"]
        
        random_order = settings["random_order"]
        
        verbose = settings["verbose"]
        
        if (settings["sigma"] == 0):
            sigma = (self.rows + self.cols)/2. - .1  
        else:
            sigma = settings["sigma"]
            
            
        """
        End user settings.
        """

        if (len(self.da.shape) == 3): #time/lat/lon
            """
            Data formatting
            """
            lats = self.da['lat'].values
            lons = self.da['lon'].values
            subset = self.da
            
            # we need to reshape the array to have two dimensions (time, lat x lon)
            # reshape this using xarray instead of numpy to be safe (using dimension names)
            data = self.da.values
            nt,ny,nx = data.shape
            subset=subset.stack(new=("lat","lon"))

            # if array looks good, assign to numpy array object
            subsetarray = subset.values
            input_length = subsetarray.shape[1] 
        else:
            sys.exit("Unexpected data format. Double check your input data is formatted correctly!")
        
        """
        SOM initialization
        """

        # Initialize the SOM
        som = MiniSom(
                    self.rows,
                    self.cols,
                    input_length,
                    sigma,
                    learning_rate,
                    decay_function,
                    neighborhood_function,
                    topology,
                    activation_distance,
                    random_seed) 

        
        #Normalize the data before training
        if (len(self.da.shape) == 4):
            data[:,0,:] = normalize_data(subsetarray[:,0,:])
            data[:,1,:] = normalize_data(subsetarray[:,1,:])

            data=xr.DataArray(data,dims=['time', 'stackxy','variables'])
            data=data.stack(new=('stackxy','variables')).values
        elif (len(self.da.shape) == 3):
            data = normalize_data(subsetarray)
        else:
            sys.exit("Unexpected data format.")
            
        # before training, initialize the weights, here using random weights to initialize
        som.random_weights_init(data)    # random method

        #som.pca_weights_init(data)   # or use this: Initializes the weights to span the first two principal components
        
        """
        SOM Training
        """

        som.train(
                data,
                num_iteration,
                random_order,
                verbose)

        print("Topographic error: " + str(som.topographic_error(data)))
        
        self.t_err = som.topographic_error(data)
        self.q_err = som.quantization_error(data)
        """
        This plot shows the SOM lattice's distance map of weights.
        Each cell is the normalised sum of the distances between
        a neuron and its neighbours. Note that this method uses
        the euclidean distance.
        """

        plt.figure(figsize=(10, 9))
        cs = plt.pcolormesh(som.distance_map(), cmap='bone_r')
        plt.title("distance map of the weights", fontsize=12)
        plt.colorbar(cs)
        plt.ylim(self.rows, 0)  # flip the y axis to be the same as composite map axes later
        plt.show()


        """
        This plot shows how cases are distributed across the SOM nodes. 
        The markers are randomly plotted across each node.
        """

        # grab the x and y coords across the lattice for the winner node for each data point
        w_x, w_y = zip(*[som.winner(d) for d in data])
        w_x = np.array(w_x)
        w_y = np.array(w_y)

        # visualize where data falls in lattice (nearest winning neuron/node)
        plt.figure(figsize=(10, 9))
        plt.pcolormesh(som.distance_map(), cmap='Blues', alpha=.2)
        plt.colorbar()

        for num, c in enumerate(w_x[::1]):     # every 20th data point

            plt.scatter(w_y[num] + np.random.rand(1),   # add a random decimal to prevent overlapping of markers
                        w_x[num] + np.random.rand(1),
                        s=50, c='k', marker='x')

        plt.title("Data across SOM lattice", fontsize=12)
        plt.margins(x=0,y=0)
        plt.grid()
        plt.ylim(self.rows, 0)  # flip the y axis to be the same as composite map axes later
        plt.show()


        """
        This plot shows distribution across the lattice.
        """

        plt.figure(figsize=(10, 9))
        frequencies = som.activation_response(data)
        plt.pcolormesh(frequencies, cmap='Blues') 
        plt.colorbar()
        plt.title("data frequency (2d histogram) across SOM lattice", fontsize=12)
        plt.ylim(self.rows, 0)  # flip the y axis to be the same as composite map axes later
        plt.show()



        """
        After the SOM is trained, we want to know which cases are associated which each SOM node.
        These values are stored in winmap[x,y], where x is the SOM row and y is the SOM column.
        """

        # create an empty dictionary using the rows and columns of SOM
        keys = [i for i in product(range(self.rows),range(self.cols))]
        winmap = {key: [] for key in keys}
        self.winmap = winmap

        # grab the indices for the data within the SOM lattice
        for i, x in enumerate(data):
            winmap[som.winner(x)].append(i)

        # create list of the dictionary keys
        som_keys = list(winmap.keys())

        """
        Now, we can get the composite data for each SOM node and save it in a list.
        """
        mapnum = 0  #iterator
        for x in range(0,self.rows):
            for y in range(0,self.cols):
                self.comp_data[x][y] = subset[np.array(winmap[som_keys[mapnum]])].unstack().mean(dim="time",skipna=True).values
                mapnum = mapnum + 1;
        return som
    
    
    """
    Trains an event relative SOM and creates composites around the central location.
        1. A distance map of weights. Each cell is the normalised sum of the distances between
        a neuron and its neighbours.
        2. Distribution of data across the SOM lattice. The markers are randomly plotted across each node.
        3. Data frequency across the SOM lattice. 
    return som: the SOM generated. It can be saved using pickle.
    """
    def make_ER_SOM(self):
        
        self.varName = self.da.attrs['standard_name'] 
        """
        Function for normalizing data prior to training using z-score (MM)
        param data: the data
        returns: the normalized data.
        """
        def normalize_data(data):
            return (data - np.nanmean(data)) / np.nanstd(data)

        """
        SOM training user settings
        """
        decay_function = ""
        myVars = vars()
        #Open the blossom_settings configuration file and read from it
        f = open(file = os.getcwd() + "/blossom_settings.json5", mode = "r")
        settings_str = f.read()
        f.close()
        settings = json5.loads(settings_str)
        
        learning_rate = settings["learning_rate"]
        
        decay_function = asymptotic_decay
        
        neighborhood_function = settings["neighborhood_function"]
        
        topology = settings["topology"]
        
        activation_distance = settings["activation_distance"]
        
        random_seed = settings["random_seed"]
        
        num_iteration = settings["num_iteration"]
        
        random_order = settings["random_order"]
        
        verbose = settings["verbose"]
        
        if (settings["sigma"] == 0):
            sigma = (self.rows + self.cols)/2. - .1  
        else:
            sigma = settings["sigma"]
          
        """
        End user settings.
        """
        
        subset = self.da

        # we need to reshape the array to have two dimensions (time, lat x lon)
        # reshape this using xarray instead of numpy to be safe (using dimension names)
        data = self.da.values
        nt,ny,nx = data.shape
        subset=subset.stack(new=("lat","lon"))

        # if array looks good, assign to numpy array object
        subsetarray = subset.values

        input_length = subsetarray.shape[1] 
        # Initialize the SOM
        som = MiniSom(
                    self.rows,
                    self.cols,
                    input_length,
                    sigma,
                    learning_rate,
                    decay_function,
                    neighborhood_function,
                    topology,
                    activation_distance,
                    random_seed)

        #Normalize the data before training
        data = normalize_data(subsetarray)

        # before training, initialize the weights, here using random weights to initialize
        som.random_weights_init(data)    # random method

        #som.pca_weights_init(data)   # or use this: Initializes the weights to span the first two principal components

        # train the SOM 

        som.train(
                data,
                num_iteration,
                random_order,
                verbose)

        """
        This plot shows the SOM lattice's distance map of weights.
        Each cell is the normalised sum of the distances between
        a neuron and its neighbours. Note that this method uses
        the euclidean distance.
        """

        plt.figure(figsize=(10, 9))
        cs = plt.pcolormesh(som.distance_map(), cmap='bone_r')
        plt.title("distance map of the weights", fontsize=12)
        plt.colorbar(cs)
        plt.ylim(self.rows, 0)  # flip the y axis to be the same as composite map axes later
        plt.show()


        """
        This cell shows how cases are distributed across the SOM nodes. 
        The markers are randomly plotted across each node.
        """

        # grab the x and y coords across the lattice for the winner node for each data point
        w_x, w_y = zip(*[som.winner(d) for d in data])
        w_x = np.array(w_x)
        w_y = np.array(w_y)

        # visualize where data falls in lattice (nearest winning neuron/node)
        plt.figure(figsize=(10, 9))
        plt.pcolormesh(som.distance_map(), cmap='Blues', alpha=.2)
        plt.colorbar()

        for num, c in enumerate(w_x[::1]):     # every 20th data point

            plt.scatter(w_y[num] + np.random.rand(1),   # add a random decimal to prevent overlapping of markers
                        w_x[num] + np.random.rand(1),
                        s=50, c='k', marker='x')

        plt.title("Data across SOM lattice", fontsize=12)
        plt.margins(x=0,y=0)
        plt.grid()
        plt.ylim(self.rows, 0)  # flip the y axis to be the same as composite map axes later
        plt.show()

        # Shows frequencies across lattice

        plt.figure(figsize=(10, 9))
        frequencies = som.activation_response(data)
        plt.pcolormesh(frequencies, cmap='Blues') 
        plt.colorbar()
        plt.title("data frequency (2d histogram) across SOM lattice", fontsize=12)
        plt.ylim(self.rows, 0)  # flip the y axis to be the same as composite map axes later
        plt.show()

        """
        After the SOM is trained, we want to know which cases are associated which each SOM node.
        These values are stored in winmap[x,y], where x is the SOM row and y is the SOM column.
        """

        # create an empty dictionary using the rows and columns of SOM
        keys = [i for i in product(range(self.rows),range(self.cols))]
        winmap = {key: [] for key in keys}
        self.winmap = winmap

        # grab the indices for the data within the SOM lattice
        for i, x in enumerate(data):
            winmap[som.winner(x)].append(i)

        # create list of the dictionary keys
        som_keys = list(winmap.keys())
        
        mapnum = 0  #iterator
        for x in range(0,self.rows):
            for y in range(0,self.cols):
                self.comp_data[x][y] = subset[np.array(winmap[som_keys[mapnum]])].unstack().mean(dim="time",skipna=True).values
                mapnum = mapnum + 1;
                
        return som