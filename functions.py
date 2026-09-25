"----------------------------------------------------""IMPORTS""-----------------------------------------------------"
import Interp3Dfield as tt

from neuron import h, gui
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import sys
import math
from scipy.interpolate import griddata, interpn
import mat73
import trimesh
import pickle
import copy
import argparse

current_time = datetime.datetime.now()
now = datetime.datetime.strftime(current_time,'%d_%m_%Y_%H_%M_%S')
PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))

h.load_file('init.hoc')
h.load_file('thresh4.hoc') 
"---------------------------------------------------------------------------------------------------------------------"

"-------------------------------------------------""PRINT METADATA""--------------------------------------------------"
# writes all the sections with its segments to a txt_file
def write_model(cell_nr,txt_name=None):
    #h.cell_chooser(cell_nr)
    with open (txt_name+"_"+now+'.txt' if txt_name else 'sec_seg_model_'+str(cell_nr)+'_'+now+'.txt','w') as f:
        for sec in h.allsec():
            f.write("sec: ")
            f.write(str(sec))
            f.write("\n")
            for seg in sec.allseg():
                f.write('\t seg:')
                f.write(str(seg))
                f.write("\n")
            f.write("\n")
            f.write("\n")


# writes for every segment in every sections the extracellular variables (xtra) to a txt_file
def write_xtra(cell_nr,txt_name=None):
    #h.cell_chooser(cell_nr)
    with open (txt_name+"_"+now+'.txt' if txt_name else 'xtra_model_'+str(cell_nr)+'_'+now+'.txt','w') as f:
        for sec in h.allsec():
            f.write("sec: ")
            f.write(str(sec))
            f.write("\n")
            for seg in sec.allseg():
                if "Scale" not in str(seg) and "Elec" not in str(seg):
                    f.write("\tseg: "); f.write(str(seg)); f.write("\n")
                    f.write("\t"); f.write("x = "); f.write(str(seg.x_xtra)); f.write(", y = ")
                    f.write(str(seg.y_xtra)); f.write(", z = "); f.write(str(seg.z_xtra)); 
                    f.write("\n"); f.write("\t")
                    #seg.es_xtra = 5 #you can adapt these values
                    f.write("es = "); f.write(str(seg.es_xtra)); f.write(", type = ")
                    f.write(str(seg.type_xtra)); f.write(", order = "); f.write(str(seg.order_xtra))
                f.write("\n")
                f.write("\n")


# prints the temporal/spatial parameters
def print_temporal(cell_nr): #stimWaveform.hoc
    #h.cell_chooser(cell_nr)
    print("-"*15,'temporal parameters',"-"*15)
    print("Amplitude = ",h.AMP)
    print("Duration= ",h.DUR)
    print("Delay = ",h.DEL)
    print("-"*50)
def print_spatial(cell_nr): #calcVe.hoc
    #h.cell_chooser(cell_nr)
    #h.stim_mode = 2 #to change the stimulation mode
    print("-"*15,'spatial parameters',"-"*15)
    print("Stim mode: ",h.stim_mode, "--> 1: ICMS, 2: uniform E-field")
    print("For ICMS: x_e = ",h.xe,", y_e = ",h.ye,", z_e = ",h.ze,", sig_e = ",h.sigma_e)
    print("For uniform E-field stim: theta = ",h.theta,"phi = ",h.phi)
    print("-"*50)
"---------------------------------------------------------------------------------------------------------------------"

"---------------------------------------------""SET SPATIAL & TEMPORAL""----------------------------------------------"
# sets the spatial (1: ICMS, 2: uniform) and temporal parameters
def set_spatial1(x=None,y=None,z=None):
    h.xe = x if x is not None else h.xe
    h.ye = y if y is not None else h.ye
    h.ze = z if z is not None else h.ze
def set_spatial2(theta=None,phi=None):
    h.theta = theta if theta is not None else h.theta
    h.phi = phi if phi is not None else h.phi
def set_temporal(amp,dur,delay):
    h.AMP = amp if amp is not None else h.AMP
    h.DUR = dur if dur is not None else h.DUR
    h.DEL = delay if delay is not None else h.DEL 


# generates a ICMS(1) or uniform field(2) with the given spatial and temporal parameters
def field_gen(cell_nr,field_type,theta=None,phi=None,x=None,y=None,z=None,sigma=None,amp=None,dur=None,delay=None):
    #h.cell_chooser(cell_nr)
    h.stim_mode = field_type # this is 1 by default (in calcVe.hoc)

    #spatial parameters
    #for ICMS
    set_spatial1(x,y,z)

    #for uniform E-field
    set_spatial2(theta,phi)

    #h.x()
    h.getes() #run this after changing a spatial param

    #temporal parameters
    set_temporal(amp,dur,delay)

    h.setstim(h.DEL,h.DUR,h.AMP) #to set the stimulation, run this after changing a temporal param
    #h.plot_waveform(h.AMP) #optional: to plot the simulation waveform
"---------------------------------------------------------------------------------------------------------------------"

"--------------------------------------------""SPATIAL POTENTIAL MATRIX""---------------------------------------------"
# gives an analysis of the coordinate distribution by plotting the frequency histograms
def coord_distr_analysis():
    x_coord, y_coord, z_coord = [],[],[]
    for sec in h.allsec():
        for seg in sec.allseg():
            if "Scale" not in str(seg) and "Elec" not in str(seg):
                x_coord.append(seg.x_xtra)
                y_coord.append(seg.y_xtra)
                z_coord.append(seg.z_xtra)
    plt.hist(x_coord,bins='auto')
    plt.xlabel("x coord [$\mu m$]")
    plt.ylabel("frequency")
    plt.title("Histogram for %i sections"%(len(x_coord)))
    plt.show()
    plt.hist(y_coord,bins='auto')
    plt.xlabel("y coord [$\mu m$]")
    plt.ylabel("frequency")
    plt.title("Histogram for %i sections"%(len(y_coord)))
    plt.show()
    plt.hist(z_coord,bins='auto')
    plt.xlabel("z coord [$\mu m$]")
    plt.ylabel("frequency")
    plt.title("Histogram for %i sections"%(len(z_coord)))
    plt.show()


# determines the min and max values of the extracellular spatial coords
def xtra_minmax(cell_nr):
    x_min,x_max,y_min,y_max,z_min,z_max  = 0,0,0,0,0,0
    for sec in h.allsec():
        for seg in sec.allseg():
            if "Scale" not in str(seg) and "Elec" not in str(seg):
                if seg.x_xtra < x_min:
                    x_min = seg.x_xtra
                elif seg.x_xtra > x_max:
                    x_max = seg.x_xtra
                if seg.y_xtra < y_min:
                    y_min = seg.y_xtra
                elif seg.y_xtra > y_max:
                    y_max = seg.y_xtra
                if seg.z_xtra < z_min:
                    z_min = seg.z_xtra
                elif seg.z_xtra > z_max:
                    z_max = seg.z_xtra  
    return x_min,x_max,y_min,y_max,z_min,z_max 


# determines the percentile values of the extracellular spatial coords (instead of using the absolute min and max)
def xtra_minmax_percentile(cell_nr,min_perc,max_perc):
    x_coord, y_coord, z_coord = [],[],[]
    for sec in h.allsec():
        for seg in sec.allseg():
            if "Scale" not in str(seg) and "Elec" not in str(seg):
                x_coord.append(seg.x_xtra)
                y_coord.append(seg.y_xtra)
                z_coord.append(seg.z_xtra)
    x_min,x_max = np.percentile(x_coord,min_perc),np.percentile(x_coord,max_perc)
    y_min,y_max = np.percentile(y_coord,min_perc),np.percentile(y_coord,max_perc)
    z_min,z_max = np.percentile(z_coord,min_perc),np.percentile(z_coord,max_perc)
    return x_min,x_max,y_min,y_max,z_min,z_max 


# determines the min and max values in each axis using the discretisation and the actual min and max of the axes
def disc_minmmax(x_min,x_max,dx,y_min,y_max,dy,z_min,z_max,dz):
    x_min,y_min,z_min = np.floor(x_min/dx)*dx, np.floor(y_min/dy)*dy, np.floor(z_min/dz)*dz
    x_max,y_max,z_max = np.ceil(x_max/dx)*dx, np.ceil(y_max/dy)*dy, np.ceil(z_max/dz)*dz
    return x_min,x_max,y_min,y_max,z_min,z_max

# creates a numpy array based on the min. value, max.value and spacing (discretization)
def create_coordvalues(ax_min,ax_max,d_ax):
    return np.arange(ax_min,ax_max+d_ax,d_ax)


# calculates the 3-D es_matrix for a uniform E-field
def es_matrix_uniform(phi,theta,dx,dy,dz,x_min,x_max,y_min,y_max,z_min,z_max,E=1): #E in V/m
    #degree to radian conversion
    theta = theta*np.pi/180
    phi = phi*np.pi/180

    ##dx, dy, dz = (x_max-x_min)/disc_points,(y_max-y_min)/disc_points,(z_max-z_min)/disc_points

    # convert the min and max of x,y and z to the extreme values considering the discretisation
    [x_min,x_max,y_min,y_max,z_min,z_max] = disc_minmmax(x_min,x_max,dx,y_min,y_max,dy,z_min,z_max,dz)

    x_values = create_coordvalues(x_min,x_max,dx)
    y_values = create_coordvalues(y_min,y_max,dy)
    z_values = create_coordvalues(z_min,z_max,dz)
    Ex, Ey, Ez = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
    es_mat = np.array([[[-(Ex*x + Ey*(-z) + Ez*y)*1e-3 for z in z_values] for y in y_values] for x in x_values])
    dimensions = [len(x_values),len(y_values),len(z_values)]
    return es_mat,dimensions #Ve in mV


# calculates the 3-D es_matrix for a point electrode (ICMS)
def es_matrix_ICMS(xe,ye,ze,sigma_e,dx,dy,dz,x_min,x_max,y_min,y_max,z_min,z_max,I=1): #I in μA

    #dx, dy, dz = (x_max-x_min)/disc_points,(y_max-y_min)/disc_points,(z_max-z_min)/disc_points
    [x_min,x_max,y_min,y_max,z_min,z_max] = disc_minmmax(x_min,x_max,dx,y_min,y_max,dy,z_min,z_max,dz)

    x_values = create_coordvalues(x_min,x_max,dx)
    y_values = create_coordvalues(y_min,y_max,dy)
    z_values = create_coordvalues(z_min,z_max,dz)
    es_mat = np.array([[[1e-3/(4*np.pi*sigma_e*(np.sqrt((x-xe)**2+(y-ye)**2+(z-ze)**2))) for z in z_values] for y in y_values] for x in x_values])
    dimensions = [len(x_values),len(y_values),len(z_values)]
    return es_mat,dimensions #Ve in mV


# calculates the tetrahedral mesh of the es for a uniform E-field
def es_tetra_uniform(phi,theta,dx,dy,dz,x_min,x_max,y_min,y_max,z_min,z_max,E=1): #E in V/m
    #degree to radian conversion
    theta = theta*np.pi/180
    phi = phi*np.pi/180

    ##dx, dy, dz = (x_max-x_min)/disc_points,(y_max-y_min)/disc_points,(z_max-z_min)/disc_points

    # convert the min and max of x,y and z to the extreme values considering the discretisation
    [x_min,x_max,y_min,y_max,z_min,z_max] = disc_minmmax(x_min,x_max,dx,y_min,y_max,dy,z_min,z_max,dz)

    x_values = create_coordvalues(x_min,x_max,dx)
    y_values = create_coordvalues(y_min,y_max,dy)
    z_values = create_coordvalues(z_min,z_max,dz)
    Ex, Ey, Ez = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
    es_tetra = np.array([[x, y, z, -(Ex*x + Ey*(-z) + Ez*y)*1e-3] for z in z_values for y in y_values for x in x_values])
    dimensions = [len(x_values),len(y_values),len(z_values)]
    return es_tetra,dimensions #Ve in mV


# calculate the potential matrix using the electric field components
def es_matrix_matf(E_x,E_y,E_z,dx,center,center_coord=None,dy=None,dz=None):
    # E_x,E_y,E_z in V/m
    if not dy:
        dy = dx
    if not dz:
        dz = dx
    # dx in um
    if (E_x.shape != E_y.shape) or (E_x.shape != E_z.shape):
        print("-"*10,"dimensions of Ex,Ey and Ez are not the same!","-"*10)
    es_shape = E_x.shape
    cx, cy, cz = center
    if center_coord: #if the exact coordinates are given instead of the indexes (more neuron placement precision is needed)
        ccx, ccy, ccz = center_coord
    else:
         ccx, ccy, ccz = cx*dx, cy*dy, cz*dz
    x_coord = np.arange(-ccx,es_shape[0]*dx-ccx,dx)
    y_coord = np.arange(-ccy,es_shape[1]*dy-ccy,dy)
    z_coord = np.arange(-ccz,es_shape[2]*dz-ccz,dz)
    xv,yv,zv = np.meshgrid(x_coord,y_coord,z_coord,indexing='ij')
    #print(xv,yv,zv)
    es_mat = -(E_x*xv + E_y*(-zv) + E_z*yv)*1e-3
    return es_mat, x_coord, y_coord, z_coord
"----------------------------------------------------------------------------------------------------------------------"

"--------------------------------------------""CART/TETRA/TXT CONVERSION""---------------------------------------------"
# writes es matrix to a txt_file
def es_mat_to_txt(es_mat,txt_name,dimensions=[0,0,0]):
    dim_str = "dim: "+str(dimensions[0])+"_"+str(dimensions[1])+"_"+str(dimensions[2])
    with open (txt_name+"_"+now+"_"+dim_str+'.txt' if txt_name else 'es_mat_'+str(cell_nr)+'_'+now+"_"+dim_str+'.txt','w') as f:
        # f.write(str(h.AMP));f.write("\t");f.write(str(h.DUR));f.write("\t");f.write(str(h.DEL));f.write("\t")
        # f.write(str(h.stim_mode));f.write("\t");f.write(str(h.xe));f.write("\t");f.write(str(h.ye));f.write("\t");
        # f.write(str(h.ze));f.write("\t");f.write(str(h.theta));f.write("\t");f.write(str(h.phi))
        # f.write("\n")
        #f.write(str(es_mat))
        for line in es_mat:
            np.savetxt(f,line)


# writes es tetrahedral mesh to a txt_file
def tetra_to_txt(es_mesh,txt_name,dimensions=[0,0,0]):
    dim_str = "dim: "+str(dimensions[0])+"_"+str(dimensions[1])+"_"+str(dimensions[2])
    with open (txt_name+"_"+now+"_"+dim_str+'.txt' if txt_name else 'es_mat_'+str(cell_nr)+'_'+now+"_"+dim_str+'.txt','w') as f:
        # f.write(str(h.AMP));f.write("\t");f.write(str(h.DUR));f.write("\t");f.write(str(h.DEL));f.write("\t")
        # f.write(str(h.stim_mode));f.write("\t");f.write(str(h.xe));f.write("\t");f.write(str(h.ye));f.write("\t");
        # f.write(str(h.ze));f.write("\t");f.write(str(h.theta));f.write("\t");f.write(str(h.phi))
        # f.write("\n")
        #f.write(str(es_mat))
        np.savetxt(f,es_mesh)


# converts txt cartesian matrix to numpy array
def txt_to_es_mat(txt_name,dimensions):
    with open (txt_name) as f:
        data = np.genfromtxt(f)
    data = data.reshape((dimensions))
    return data


# converts txt tetrahedral list to numpy array
def txt_to_tetra(txt_name):
    with open (txt_name) as f:
        data = np.genfromtxt(f)
    return data


# converts a cartesian mesh to a tetrahedral mesh
def cart_to_tetra(es_matrix,x_min,x_max,y_min,y_max,z_min,z_max,dx,dy,dz):
    tetra_len = np.prod(es_matrix.shape)
    tetra_mesh = np.empty((tetra_len,4))
    [x_min,x_max,y_min,y_max,z_min,z_max] = disc_minmmax(x_min,x_max,dx,y_min,y_max,dy,z_min,z_max,dz)
    x_values = create_coordvalues(x_min,x_max,dx)
    y_values = create_coordvalues(y_min,y_max,dy)
    z_values = create_coordvalues(z_min,z_max,dz)
    for i,e in enumerate(es_matrix):
        for j,f in enumerate(e):
            for k,g in enumerate(f):
                index = np.ravel_multi_index((i,j,k),(es_matrix.shape))
                tetra_mesh[index] = [x_values[i],y_values[j],z_values[k],g]
    return tetra_mesh


# read matlab EF (electric field) input file
def read_matfEF(name):
    matf = sio.loadmat(name,mat_dtype=True)
    E_x, E_y, E_z = matf['EF'][0][0]
    return E_x,E_y,E_z


# read matlab Model input file
def read_matfMod(name):
    matf = sio.loadmat(name,mat_dtype=True)
    brain = matf['Model'][0][0][0]
    # for i in range(11):
    #     plt.imshow((brain==i)[170,:,:])
    #     plt.show()
    #     print(np.sum(brain==i))
    # print(matf['Model'][0][0][1])
    return brain
"----------------------------------------------------------------------------------------------------------------------"

"------------------------------------------------------""PLOTS""-------------------------------------------------------"
# plots the potential value (es) over the simulation time
def plot_ap(cell):
    timevec = h.Vector()
    timevec.record(h._ref_t)

    esvec = h.Vector()
    esvec.record(cell.soma[0](0)._ref_v)

    h.run()

    tarray = np.array(timevec)
    esarray = np.array(esvec)
    plt.plot(tarray,esarray)
    plt.show()


#plots saved numpy arrays that are loaded in (xfile and yfile contain .npy)
def plot_npy(xfile,yfile,xlabel,ylabel,logtype=None,E_max=None):
    freq_array = np.load(PROJECT_ROOT+xfile)
    thresh_array = np.load(PROJECT_ROOT+yfile)
    if E_max: # if the maximum electric field is given, we can multiply this: titration factor -> actual E-fields
        thresh_array *= E_max
    if logtype == None:
        plt.plot(freq_array,thresh_array)
    if logtype == 'log':
        plt.loglog(freq_array,thresh_array)
    if logtype == 'logx':
        plt.semilogx(freq_array,thresh_array)
    if logtype == 'logy':
        plt.semilogy(freq_array,thresh_array)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show() 


#plots saved numpy arrays that are loaded in (xfile and yfile contain .npy)
def plot_npy_err(xfile,yfile,xlabel,ylabel,gt_index,logtype=None,E_max=None):
    #gt_index = ground truth index
    freq_array = np.load(PROJECT_ROOT+xfile)
    thresh_array = np.load(PROJECT_ROOT+yfile)
    ground_truth = thresh_array[gt_index]
    thresh_array = (thresh_array-ground_truth)/ground_truth*100
    plt.figure(figsize=(4.5,4)) #width, height
    if E_max: # if the maximum electric field is given, we can multiply this: titration factor -> actual E-fields
        thresh_array *= E_max
    if logtype == None:
        plt.plot(freq_array,thresh_array)
    if logtype == 'log':
        plt.loglog(freq_array,thresh_array)
    if logtype == 'logx':
        plt.semilogx(freq_array,thresh_array)
    if logtype == 'logy':
        plt.semilogy(freq_array,thresh_array)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()


#plots multiple saved numpy arrays that are loaded in (for different locations), (xfile and yfile don't contain .npy)
def plot_npy_loc(xfile,yfile,xlabel,ylabel,logtype=None,E_max=None):
    E_x,E_y,E_z = read_matfEF('Exyz_ChibaU.mat') #E_x, E_y, E_z in V/m
    E_mag = np.sqrt(E_x**2+E_y**2+E_z**2)
    ending = '_interpol:linear_logspace:(10, 2, 5)_cell_nr:7_dt_fact:100'
    for i in range(1,11):
        maxi = nth_max3D(E_mag,i)
        freq_array = np.load(PROJECT_ROOT+xfile+"_loc:"+str(maxi)+ending+'.npy')
        #print(freq_array)
        thresh_array = np.load(PROJECT_ROOT+yfile+'_loc:'+str(maxi)+ending+'.npy')
        if E_max: # if the maximum electric field is given, we can multiply this: titration factor -> actual E-fields
            thresh_array *= E_max
        if logtype == None:
            plt.plot(freq_array,thresh_array,label=str(maxi))
        if logtype == 'log':
            plt.loglog(freq_array,thresh_array,label=str(maxi))
        if logtype == 'logx':
            plt.semilogx(freq_array,thresh_array,label=str(maxi))
        if logtype == 'logy':
            plt.semilogy(freq_array,thresh_array,label=str(maxi))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.show() 


#plots multiple saved numpy arrays that are loaded in (for different interpolation methods), (xfile and yfile don't contain .npy)
def plot_npy_int(xfile,yfile,xlabel,ylabel,logtype=None,E_max=None):
    for i in ['linear','slinear','nearest','cubic']:
        freq_array = np.load(PROJECT_ROOT+xfile+i+'.npy')
        thresh_array = np.load(PROJECT_ROOT+yfile+i+'.npy')
        if E_max: # if the maximum electric field is given, we can multiply this: titration factor -> actual E-fields
            thresh_array *= E_max
        if logtype == None:
            plt.plot(freq_array,thresh_array,label=i)
        if logtype == 'log':
            plt.loglog(freq_array,thresh_array,label=i)
        if logtype == 'logx':
            plt.semilogx(freq_array,thresh_array,label=i)
        if logtype == 'logy':
            plt.semilogy(freq_array,thresh_array,label=i)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.show() 


#plots multiple saved numpy arrays that are loaded in (for different dt factors), (xfile and yfile don't contain .npy)
def plot_npy_dtfact(xfile,yfile,xlabel,ylabel,dt_factors,logtype=None,E_max=None):
    all = 0 #bool #if 1: all dt-factors need to be plotted, if 0: remove the useless ones (the ones with very high thresholds)
    for i in dt_factors:
        freq_array = np.load(PROJECT_ROOT+xfile+str(i)+'.npy')
        thresh_array = np.load(PROJECT_ROOT+yfile+str(i)+'.npy')
        if not(all):
            if np.sum(thresh_array>1e3) >0:
                continue
        if E_max: # if the maximum electric field is given, we can multiply this: titration factor -> actual E-fields
            thresh_array *= E_max
        if logtype == None:
            plt.plot(freq_array,thresh_array,label=i)
        if logtype == 'log':
            plt.loglog(freq_array,thresh_array,label=i)
        if logtype == 'logx':
            plt.semilogx(freq_array,thresh_array,label=i)
        if logtype == 'logy':
            plt.semilogy(freq_array,thresh_array,label=i)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.show() 


#plots multiple saved numpy arrays that are loaded in (for different dt factors), (xfile and yfile don't contain .npy)
def plot_npy_cells(xfile,yfile,xlabel,ylabel,logtype=None,E_max=None):
    for i in range(1,26):
        freq_array = np.load(PROJECT_ROOT+xfile+str(i)+'linear.npy')
        thresh_array = np.load(PROJECT_ROOT+yfile+str(i)+'linear.npy')
        if E_max: # if the maximum electric field is given, we can multiply this: titration factor -> actual E-fields
            thresh_array *= E_max
        if logtype == None:
            plt.plot(freq_array,thresh_array,label=i)
        if logtype == 'log':
            plt.loglog(freq_array,thresh_array,label=i)
        if logtype == 'logx':
            plt.semilogx(freq_array,thresh_array,label=i)
        if logtype == 'logy':
            plt.semilogy(freq_array,thresh_array,label=i)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.show() 


# implementation of plot_waveform in Python (doesn't really work yet)
def plot_waveformPython(ancat):
    # creates a graph
    g1 = h.Graph(0)
    # specify coordinate system: xmin, xmax, ymin, ymax
    g1.size(0,h.stim_time.size(),-1,h.AMP)
    h.stim_amp.plot(g1,h.stim_time)
    # mleft, mbottom, mwidth, mheight, wleft, wtop, wwidth, wheight
    # m stand for model coordinates within the window, w stands for screen coordinates for placement and size of the window
    if ancat == 'an':
        g1.view(0,-h.AMP/2,h.stim_time.x[h.stim_time.size()-1],h.AMP*2,1081,547,300.48,200.32)            
    if ancat == 'cat':
        g1.view(0,3*h.AMP/2,h.stim_time.x[h.stim_time.size()-1],-h.AMP*2,1081,547,300.48,200.32)
    time.sleep(2) # graph gets removed when going out of this function 
    plt.plot(h.stim_time,h.stim_amp)
    plt.show()

"----------------------------------------------------------------------------------------------------------------------"

"------------------------------------------""TEMPORAL STIMULATION WAVEFORM""-------------------------------------------"
# creates a sinusoidal stimulation waveform (as in h.setstim()) -> do not confuse with stim_waveform()
def stim_waveform(stim_length,freq,t_max,amp,dur,delay,plot=False):
    #stim_length = int(100*freq) #critical
    #print("freq = %.2e Hz, "%(freq),"stim length = %.2e samples"%(stim_length))
    if t_max < dur+delay:
        print("-"*10,"total duration is too small!","-"*10)
        t_max = delay+dur+1 #no need for a larger t_max than this one
    h.stim_amp.resize(stim_length)
    h.stim_time.resize(stim_length)

    stim_time = np.linspace(0,t_max,stim_length)
    for i in range(stim_length):
        h.stim_time.x[i] = stim_time[i]
    deldur = np.array([1 if (e > delay and e < delay+dur) else 0 for e in h.stim_time]) #taking into account the delay and duration
    stim_amp = amp*deldur#*np.sin(2*np.pi*freq*stim_time)
    for i in range(stim_length):
        h.stim_amp.x[i] = stim_amp[i]
    #attemptwVector: attempt to do this without for loop and with h.Vector
    #h.stim_time = h.Vector(np.linspace(0,t_max,stim_length))
    #h.stim_amp = h.Vector(amp*deldur*np.sin(2*np.pi*freq*stim_time))
    h.attach_stim()
    #h.plot_waveform(amp) #this is a build-in function to plot (in the GUI) # also possible to use Python version
    print("Generated waveform with del = %g ms, dur = %g ms, amp = %g uA or V/m"%(delay,dur,amp))

    if plot:
        stimamp_array = np.array(h.stim_amp)
        stimtime_array = np.array(h.stim_time)
        plt.plot(stimtime_array,stimamp_array)
        plt.xlabel('time [ms]')
        plt.ylabel('amplitude [$\mu A$]')
        plt.show()
"---------------------------------------------------------------------------------------------------------------------"

"------------------------------------------------""THRESHOLD SWEEPS""-------------------------------------------------"
# calculates the threshold for different frequencies of the sines of h.stim_amp (using calcThreshPython)
def thresh_logfreq(numb_freq,logstart,logstop,name,dt_factor):
    xSOM, ySOM, zSOM = tt.findSOMA()
    thresh_freq = np.zeros(numb_freq)
    freq_array = np.logspace(logstart,logstop,numb_freq)  

    for i,freq in enumerate(freq_array):
        print("iteration ",i+1," freq = ",freq)
        DUR_factor = 1
        h.dt = min(1/(dt_factor*freq) * 1e3, 0.025) #because h.dt is in ms, h.dt = 0.025 ms(default) when freq = 1e3 Hz
        h.DUR = int(np.ceil(max(5,DUR_factor*1/freq*1e3)))   # simulation should be minimally 5 ms (+ delay), low freq require higher sim times
                                       # h.DUR = 5 ms(default) when freq = 200 Hz (Tmin)
        h.tstop = h.DUR + h.DEL + 1 # +1 for safety
        #h.sine_freq = freq
        h.sine_freq_xtra = freq*1e-3
        print(freq,h.dt,h.DUR,h.tstop)
        thresh_freq[i] = tt.calcThreshSimpl(0,0,0,h.DEL,h.DUR,0)

    plt.loglog(freq_array,thresh_freq)
    plt.xlabel("frequency [Hz]")
    plt.ylabel("titration factor")
    plt.grid()
    plt.show() 
    np.save(PROJECT_ROOT+"/npy_data/week6/thresh_freq/freq_array_"+name,freq_array)
    np.save(PROJECT_ROOT+"/npy_data/week6/thresh_freq/thresh_array_"+name,thresh_freq)


# calculates the threshold for different frequencies of the sines of h.stim_amp (using calcThreshPython)
def thresh_freq(numb_freq,start,stop,name):
    xSOM, ySOM, zSOM = tt.findSOMA()
    thresh_freq = np.zeros(numb_freq)
    freq_array = np.linspace(start,stop,numb_freq)  

    for i,freq in enumerate(freq_array):
        thresh_freq[i] = tt.calcThreshSimpl(0,0,0,h.DEL,h.DUR,0)

    plt.plot(freq_array,thresh_freq)
    plt.xlabel("frequency [Hz]")
    plt.ylabel("threshold [mV]")
    plt.grid()
    plt.show() 
    np.save(PROJECT_ROOT+"/Freq_sweep/test2/non_log/freq_array_"+name,freq_array)
    np.save(PROJECT_ROOT+"/Freq_sweep/test2/non_log/thresh_freq_"+name,thresh_freq)


# calculates the threshold for different frequencies of the sines of h.stim_amp (using calcThreshPython) and for different dt_factors
def thresh_logfreqdt(numb_freq,logstart,logstop,name,dt_factors):
    xSOM, ySOM, zSOM = tt.findSOMA()
    thresh_freq = np.zeros(numb_freq)
    freq_array = np.logspace(logstart,logstop,numb_freq)  

    for dt_factor in dt_factors:
        for i,freq in enumerate(freq_array):
            print("iteration ",i+1," freq = ",freq," dt_factor = ",dt_factor)
            DUR_factor = 1
            h.dt = min(1/(dt_factor*freq) * 1e3, 0.025) #because h.dt is in ms, h.dt = 0.025 ms(default) when freq = 1e3 Hz
            #h.dt = 1/(dt_factor*freq) * 1e3
            h.DUR = int(np.ceil(max(5,DUR_factor*1/freq*1e3)))   # simulation should be minimally 5 ms (+ delay), low freq require higher sim times
                                        # h.DUR = 5 ms(default) when freq = 200 Hz (Tmin)
            h.tstop = h.DUR + h.DEL + 1 # +1 for safety
            #h.sine_freq = freq
            h.sine_freq_xtra = freq*1e-3
            print(freq,h.dt,h.DUR,h.tstop)
            thresh_freq[i] = tt.calcThreshSimpl(0,0,0,h.DEL,h.DUR,0)
            print(h.dt)

        plt.loglog(freq_array,thresh_freq,label=str(dt_factor))
        plt.xlabel("frequency [Hz]")
        plt.ylabel("titration factor")
        # plt.grid()
        # plt.show() 
        np.save(PROJECT_ROOT+"/Freq_sweep/week5/testdt/freq_array_titr_"+name+str(dt_factor),freq_array)
        np.save(PROJECT_ROOT+"/Freq_sweep/week5/testdt/thresh_freq_titr_"+name+str(dt_factor),thresh_freq)
    plt.legend()
    plt.grid()
    plt.show() 


# calculates the threshold for different h.dt
def thresh_dt(numb_dt,start,stop,name,error=False):
    xSOM, ySOM, zSOM = tt.findSOMA()
    thresh_dt = np.zeros(numb_dt)
    dt_array = np.linspace(start,stop,numb_dt)
    #print(dt_array)
    for i,dt in enumerate(dt_array):
        h.dt = dt
        print("iteration ",i," ",dt)
        thresh_dt[i] = tt.calcThreshSimpl(xSOM,ySOM,zSOM,h.DEL,h.DUR,0)
    plt.plot(dt_array,thresh_dt)
    plt.xlabel("dt [ms]")
    plt.ylabel("threshold [mV]")
    plt.grid()
    plt.show()
    if error:
        ground_truth = thresh_dt[0]
        err = (thresh_dt-ground_truth)/ground_truth*100
        plt.plot(dt_array,err)
        plt.xlabel("dt [ms]")
        plt.ylabel("relative error [%]")
        plt.grid()
        plt.show()
    np.save(PROJECT_ROOT+"/npy_data/week6/thresh_dt/dt_array_"+name,dt_array)
    np.save(PROJECT_ROOT+"/npy_data/week6/thresh_dt/thresh_array_"+name,thresh_dt)
    np.save(PROJECT_ROOT+"/npy_data/week6/thresh_dt/err_array_"+name,err)


# calculates the threshold for different h.dt
def thresh_dtPython(numb_dt,logstart,logstop,error=False):
    xSOM, ySOM, zSOM = tt.findSOMA()
    thresh_dt = np.zeros(numb_dt)
    dt_array = np.logspace(logstart,logstop,numb_dt)
    #print(dt_array)
    for i,dt in enumerate(dt_array):
        h.dt = dt
        thresh_dt[i] = calcThreshPython(100,h.DEL,h.DUR,0)
    plt.plot(dt_array,thresh_dt)
    plt.xlabel("t [ms]")
    plt.ylabel("threshold [mV]")
    plt.grid()
    plt.show()
    if error:
        ground_truth = thresh_dt[0]
        err = (thresh_dt-ground_truth)/ground_truth*100
        plt.plot(dt_array,err)
        plt.xlabel("t [ms]")
        plt.ylabel("relative error [%]")
        plt.grid()
        plt.show()


# calculates the threshold for different h.nseg
def thresh_nseg(nsegstart,nsegstop,nsegstep,error=False):
    xSOM, ySOM, zSOM = tt.findSOMA()
    #nsegstart and nsegstop need to be odd, ngsegstep needs to be even
    if (nsegstart%2 == 0) or (nsegstop%2 == 0) or (nsegstep%2 != 0):
        print("-"*10,"mulseg needs to be odd!","-"*10)
    numb_nseg = (nsegstop-nsegstart)//nsegstep + 1
    mulseg_array = np.linspace(nsegstart,nsegstop,numb_nseg,dtype=int)
    orig_nseg = np.array([],dtype=int)
    for sec in h.allsec():
        if "Scale" not in str(sec) and "Elec" not in str(sec):
            orig_nseg = np.append(orig_nseg,sec.nseg)
    #print(orig_nseg)
    thresh_nseg = np.zeros(numb_nseg)
    for i,mulseg in enumerate(mulseg_array):
        for j,sec in enumerate(h.allsec()):
            if "Scale" not in str(sec) and "Elec" not in str(sec): #if h.ismembrane("xtra"):
                #print(mulseg)
                sec.nseg *= mulseg   #*orig_nseg[j]
        h.setpointers()
        thresh_nseg[i] = tt.calcThreshSimpl(xSOM,ySOM,zSOM,h.DEL,h.DUR,0)
    plt.plot(mulseg_array,thresh_nseg)
    plt.xlabel("segmentation multiplication factor")
    plt.ylabel("threshold [mV]")
    plt.grid()    
    plt.show()  
    if error:
        ground_truth = thresh_nseg[-1]
        err = (thresh_nseg-ground_truth)/ground_truth*100
        plt.plot(mulseg_array,err)
        plt.xlabel("segmentation multiplication factor")
        plt.ylabel("relative error [%]")
        plt.grid()   
        plt.show() 


# calculates the threshold for different dx,dy and dz #TODO:generalize for all kind of fields(other param and ICMS?)
def thresh_dx(numb_dx,logstart,logstop,cell_nr,x_min,x_max,y_min,y_max,z_min,z_max,interpolation,error=False):
    xSOM, ySOM, zSOM = tt.findSOMA()
    thresh_dx = np.zeros(numb_dx)
    dx_array = np.logspace(logstart,logstop,numb_dx)
    #print(dx_array)

    field_gen(cell_nr,2,170,5,amp=55) #TODO:generalize for any field

    ground_truth = tt.calcThreshSimpl(xSOM,ySOM,zSOM,h.DEL,h.DUR,0)
    for i,dx in enumerate(dx_array):
        dy,dz = dx,dx
        if h.stim_mode == 1:
            es_mat,_ = es_matrix_ICMS(h.xe,h.ye,h.ze,h.sigma_e,dx,dy,dz,x_min,x_max,y_min,y_max,z_min,z_max)
        elif h.stim_mode == 2:
            es_mat,_ = es_matrix_uniform(h.phi,h.theta,dx,dy,dz,x_min,x_max,y_min,y_max,z_min,z_max)
        [x_min,x_max,y_min,y_max,z_min,z_max] = disc_minmmax(x_min,x_max,dx,y_min,y_max,dy,z_min,z_max,dz)
        x_values = create_coordvalues(x_min,x_max,dx)
        y_values = create_coordvalues(y_min,y_max,dy)
        z_values = create_coordvalues(z_min,z_max,dz)
        print("iteration :",i,", dx = ",dx,"mat dim: ",es_mat.shape)
        tt.calcESext2(0,0,0,0,0,0,'linear',es_mat,x_values,y_values,z_values)
        thresh_dx[i] = tt.calcThreshSimpl(xSOM,ySOM,zSOM,h.DEL,h.DUR,0)
    plt.plot(dx_array,thresh_dx)
    plt.xlabel("dx [$\mu m$]")
    plt.ylabel("threshold [mV]")
    plt.grid()
    plt.show()
    if error:
        err = (thresh_dx-ground_truth)/ground_truth*100
        plt.plot(dx_array,err)
        plt.xlabel("dx [$\mu m$]")
        plt.ylabel("relative error [%]")
        plt.grid()
        plt.show()
"----------------------------------------------------------------------------------------------------------------------"

"------------------------------------------""THRESHOLD (REIMPLEMENTATIONS)""-------------------------------------------"
# implementation of thresh_excited in python (TODO: implement this not only for sinusoidal stims)
def ThreshExcitedPython(freq):
    stim_waveform(1000,freq,h.DEL+h.DUR+1,h.ANCAT*h.stimamp,h.DUR,h.DEL) #t_max = del+dur+1
    #stim_waveform2(h.DUR,h.DEL,h.ANCAT*h.stimamp)
    h.AMP = h.ANCAT*h.stimamp
    h.run()
    return h.apc.n > 0


# implementation of threshold in python (TODO: implement this not only for sinusoidal stims)
def ThresholdPython(freq):
    low = 0
    high = 1e10
    if h.stimamp == 0: h.stimamp = 1
    while ((low == 0) or (high == 1e10)):
        if (ThreshExcitedPython(freq)):
            high = h.stimamp
            h.stimamp = high/2
        else:
            low = h.stimamp
            h.stimamp = 2*low
        if h.stoprun: return h.stimamp
        if (low > high): return high

    epsilon = 1e-8 + (1e-4)*high # 4 decimal places accuracy -> 1e-7 + 1e-3*high for 3
    h.stimamp = (high+low)/2
    while ((high-low) > epsilon):
        if ThreshExcitedPython(freq):
            high = h.stimamp
        else:
            low = h.stimamp
        h.stimamp = (high+low)/2
        if h.stoprun: break
    return h.stimamp


# redefenition of calcThresh, with an implementation completely in python (TODO: implement this not only for sinusoidal stims)
def calcThreshPython(freq,DEL,DUR,Ancat):
    phi, theta, psi = 0, 0, 0 #calcThreshSimpl
    h.DEL = DEL
    h.DUR = DUR
	#h.dt = min(DUR/TpRefdt,dtMAX)
    h.steps_per_ms = 1/h.dt				# Circumvent setdt()-function in stdinit
    h.tstop = max(tt.TpRefT*DUR,tt.Tmin)
    if (h.name_declared("apc")==0):
        h("objref apc")
        h("apc = new APCount(0.5)")
    h("stimamp = 1")
    h("ANCAT = 1") # here the ANCAT double is created in hoc (not possible to create this in python)
    h.ANCAT = (-1)**Ancat # here the ANCAT value is changed to the desired value
    print(h.ANCAT)
    #[xSOM0,ySOM0,zSOM0] = tt.findSOMA()
    #calcESext(xSOM-xSOM0,ySOM-ySOM0,zSOM-zSOM0,phi,theta,psi)
    threshold = ThresholdPython(freq)
    return h.ANCAT*threshold
"---------------------------------------------------------------------------------------------------------------------"

"--------------------------------------------------""UNCLASSIFIED""---------------------------------------------------"
# calculates the indices of the n'th maximum value of a matrix #1 is the highest number, 2 is the 2nd highest number...
def nth_max3D(mat,n):
    sorted = np.argsort(mat.ravel())
    argmax_sorted = np.unravel_index(sorted,mat.shape)
    return (argmax_sorted[0][-n],argmax_sorted[1][-n],argmax_sorted[2][-n])

def align_coord_syst():
    for sec in h.allsec():
        if 'Scale' in str(sec):
            continue
        #sec.x_xtra, sec.y_xtra, sec.z_xtra = sec.x_xtra, -sec.z_xtra, sec.y_xtra
        for seg in sec:
            seg.x_xtra, seg.y_xtra, seg.z_xtra = seg.x_xtra, -seg.z_xtra, seg.y_xtra
        for i in range(int(h.n3d(sec=sec))):
            x = sec.x3d(i)
            y = sec.y3d(i)
            z = sec.z3d(i)
            diam = sec.diam3d(i)
            new_x = x
            new_y = -z
            new_z = y
            h.pt3dchange(i, new_x, new_y, new_z, diam, sec=sec)

def align_to_normal(points, new_normal, prev_normal):
    # Normalize vectors
    z_axis = prev_normal #this is the z-axis in the first iteration but should be in all
    normal = np.array(new_normal) / np.linalg.norm(new_normal)
    
    # Compute rotation using Rodrigues' formula
    a = z_axis
    b = normal
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    
    if s == 0:
        return points.copy()  # Already aligned
    
    # Skew-symmetric matrix
    k_mat = np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])
    
    R = np.eye(3) + k_mat + (k_mat @ k_mat) * ((1 - c) / (s ** 2))
    
    # Apply rotation
    rotated_points = points @ R.T
    return rotated_points

def closest_meshes_to_point(mesh,point):

    # Compute centroids of all faces
    face_vertices = mesh.vertices[mesh.faces]  # Shape: (num_faces, 3, 3)
    centroids = face_vertices.mean(axis=1)     # Shape: (num_faces, 3)

    # Compute distances from point to each centroid
    distances = np.linalg.norm(centroids - point, axis=1)

    # Get indices of 10 closest faces
    closest_indices = np.argsort(distances)[:10]

    # For each face, compute centroid and normal
    results = []
    for idx in closest_indices:
        centroid = centroids[idx]
        normal = mesh.face_normals[idx]
        results.append({
            "face_index": idx,
            "centroid": centroid,
            "normal": normal
        })
    
    return results