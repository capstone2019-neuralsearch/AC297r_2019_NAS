# Reference implementation of table -> lattice conversion
# Copied from
# - https://github.com/phanakata/ML_for_kirigami_design/blob/master/tools/generate_lattice.py
# - https://github.com/phanakata/ML_for_kirigami_design/blob/master/models/CNN_regression/convert_coarse_to_fine.ipynb

import numpy as np


def generateInnerCell(NCcell_x, NCcell_y, ncell_x, ncell_y):
    """
    Create pristine ribbon of size (ncell_x-4)*ncell_y
    """
    inner = np.ones((ncell_x-4)*ncell_y)

    #for debugging
    #for j in range (ncell_x):
    #   print (kirigami[j*ncell_y:(j+1)*(ncell_y)])

    return inner


def makeCutsonCell(cutConfigurations, inner, NCcell_x, NCcell_y, ncell_x, ncell_y):
    """
    Make cuts in inner region

    Parameters
    --------------------
    cutConfigurations: string
        N-dimensional (binary) array of cuts with size NCcell_x * NCcell_y
    inner: array
        N-dimensional (binary) array with size (ncell_x-4)*ncell_y
    return: inner (array)
    """

    mx = (ncell_x-4)//NCcell_x
    my = ncell_y//NCcell_y
    #debugging: print(mx, my)

    #ONLY MAKE CUTS inside the INNER REGION !!
    for i in range (len(inner)):
        #first find index nx and ny
        nx = (i)//ncell_y
        ny = (i)%ncell_y

        #now convert nx and ny to Nx Ny
        Nx = nx//mx
        Ny = ny//my
        #debugging: print(Nx, Ny)

        #now conver (Nx, Ny) to one dimensional and check whether it's 0 or 1
        index = Nx * NCcell_y + Ny

        if (cutConfigurations[index]==0):
            if nx> Nx * mx +mx/3 and nx < Nx* mx +mx *2/3:
                inner[i] = 0

    return inner


def convert_to_fine_grid(raw_data):
    """
    Convert raw data (including labels) to fine-grid image
    """
    assert raw_data.shape[1] == 18
    alldata_15G = raw_data

    # paramters to make finer grids
    NCcell_x = 3
    NCcell_y = 5
    ncell_x = 34
    ncell_y = 80

    # create fine grids
    listFG=[]
    for i in range (len(alldata_15G)):
        cutConfigurations=alldata_15G[i, 0:-3]
        inner = generateInnerCell(NCcell_x, NCcell_y, ncell_x, ncell_y)
        inner_wCuts = makeCutsonCell(cutConfigurations, inner, NCcell_x, NCcell_y, ncell_x, ncell_y)
        listFG.append(inner_wCuts)

    alldata_FG = np.array(listFG)

    h=30
    w=80
    fine_image = alldata_FG.reshape([-1, h, w])

    return fine_image
