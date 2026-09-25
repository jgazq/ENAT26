import functions as fcts
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
#----------------------------------------------------------------------------------------------------------#
"cell and field initialization"


# Create the parser
parser = argparse.ArgumentParser(description="Process some parameters.")
parser.add_argument("--cell", type=int, required=True, help="Cell number")
parser.add_argument("--subject", type=int, required=True, help="Subject number")
parser.add_argument("--direction", type=str, required=True, help="Direction (e.g., lateral)")
parser.add_argument("--subjectEAS", type=str, required=True, help="Subject name")
args = parser.parse_args()
cell_nr = args.cell
subject = args.subject
direction = args.direction
subjectEAS = args.subjectEAS

print(f"Cell: {cell_nr}, Subject: {subject}, Direction: {direction}")
from neuron import h, gui

h("NSTACK_size = 10000")

h("ss_init=1")
h("init_tstart = -1e+11")
h("init_dt = 1e+09")
h.load_file('init.hoc')
h.load_file('thresh4.hoc') 
h.load_file('get_es2.hoc') #for TMS
h.load_file('interp_coordinates.hoc') #for TMS (to calc Dx,Dy and Dz)
h.load_file('ssprocinit.hoc')

#cell_nr = 7 #parsed from command line
h.setParamsAdultHuman() #this needs to go before the cell chooser, otherwise it won't make a difference
h.cell_chooser(cell_nr)
#print(h.topology()) #print this to decide the code for the cell below

cell_name = h.cell_names.o(cell_nr-1).s


current_time = datetime.datetime.now()
now = datetime.datetime.strftime(current_time,'%d_%m_%Y_%H_%M_%S')
PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))

#----------------------------------------------------------------------------------------------------------#
"flags"

createfsweepEAS = 1

#----------------------------------------------------------------------------------------------------------#
"code"

if createfsweepEAS: #similar to createfsweep but es calculation needs to happen inside freq iteration as es is different for different frequencies (<10kHz: 300Hz, >10kHz: 100 kHz)
    TMS = 0 #also use the right x86 folder and change save folder
    Eunif = 0 #also use the right x86 folder and change save folder
    Bunif = 1 #same nrnmech.dll file as TMS (or folder on linux) 

    #h.v_init = -75
    interpol = 'linear'
    logspace = (10,2,5)
    dt_fact = 100
    dx,dy,dz = 0.49911274*1e3, 0.49893188*1e3, 0.49898493*1e3 #um
    dxyz = np.array([dx,dy,dz])
    subject = subjectEAS
    GMi = 75 if subject == 'MARTIN' else 72 if subject == 'MIDA' else None

    time_start = datetime.datetime.now()

    name = "EAS_"+subject+"_interpol_"+str(interpol)+"_logspace_"+str(logspace)+"_cell_nr_"+str(cell_nr)+"_dt_fact_"+str(dt_fact)+"_subject_"

    numb_freq, logstart, logstop = logspace

    print('loading matf...')
    matf = sio.loadmat(rf'Exyz_EAS/{subject}_head_E-field.mat')
    print('loaded matf ✓')
    
    sh0, sh1, sh2 = matf['Axis0'][0,:].shape[0] -1, matf['Axis1'][0,:].shape[0] -1, matf['Axis2'][0,:].shape[0] -1
    Exyz = np.nan_to_num(np.abs(matf['Snapshot0'].reshape(sh2,sh1,sh0,3)))
    E_x, E_y, E_z = Exyz[:,:,:,0], Exyz[:,:,:,1], Exyz[:,:,:,2]
    E_mag_trfl = np.sqrt(E_x**2+E_y**2+E_z**2)
    E_mag_fl = np.transpose(E_mag_trfl, (1, 2, 0))
    E_mag = np.flip(E_mag_fl, axis=1)

    data = np.fromfile(rf'Exyz_EAS/{subject}_head_voxels.raw', dtype=np.uint8)
    brainmap_0 = data.reshape((sh2,sh1,sh0,2))
    brainmap_trfl = brainmap_0[:,:,:,0]
    brainmap_fl = np.transpose(brainmap_trfl, (1, 2, 0))
    brainmap = np.flip(brainmap_fl, axis=1)

    E_mag_GM = (E_mag*(brainmap==GMi)).flatten()
    E_max_GM = np.max(E_mag*(brainmap==GMi))
    E_max_GM_99 = np.percentile(E_mag_GM[E_mag_GM != 0],99)
    print(f'max: {E_max_GM} , 99 percentile: {E_max_GM_99}')
    E_max_GMloc = np.unravel_index(np.argmax(E_mag*(brainmap==GMi)),E_mag.shape)
    #meshname = 'layer_1_depth_0.06' if cell_nr < 6 else 'layer_23_depth_0.40' if cell_nr < 11 else 'layer_4_depth_0.55' if cell_nr < 16 else 'layer_5_depth_0.65' if cell_nr < 21 else 'layer_6_depth_0.85' if cell_nr <26 else 'ERROR'
    meshname = 'layer_1_depth_0.06' if 'L1' in cell_name else 'layer_23_depth_0.40' if 'L23' in cell_name else 'layer_4_depth_0.55' if 'L4' in cell_name else 'layer_5_depth_0.65' if 'L5' in cell_name else 'layer_6_depth_0.85' if 'L6' in cell_name else 'ERROR'
    print('loading trimesh...')
    surface_mesh = trimesh.load_mesh(rf"tissue_meshes/EAS/{subject}/layers/{meshname}.stl")
    print('loaded trimesh ✓')
    polygons = fcts.closest_meshes_to_point(surface_mesh,E_max_GMloc)
    
    results = {'hotspot': {key: {} for key in range(10)}}
    prev_normal = np.array([0,0,1])

    for i_p,polygon in enumerate(polygons[:]):
        location, normal = polygon['centroid'], polygon['normal']
        print(f'location_{i_p}: {location}, {normal} ({subject})')
        _,x_values,y_values,z_values = fcts.es_matrix_matf(E_x,E_y,E_z,dx,(0,0,0),dy=dy,dz=dz)
        thresh_freqs = np.zeros(numb_freq)
        titr_freqs = np.zeros(numb_freq)
        freq_array = np.logspace(logstart,logstop,numb_freq)  

        for i_f,freq in enumerate(freq_array):
            print(f'frequency_{i_f}: {freq}')

            if TMS or Bunif:
                h.getcoords() #gets the coordinates and also calculates D_x, D_y and D_z at every segment
                #-> D_x, D_y and D_z get a value (before = 0)
                tt.calcESext3(*location*dxyz,normal,np.array([0,0,1]),interpol,E_x,E_y,E_z,x_values,y_values,z_values) #calculation of E_x, E_y and E_z at every segment
                #-> E_x, E_y and E_z get a value (before = 0)
                #h("load_potentials = 1")
                h.calc_pseudo_es()#h.getes2() #calculation of the potential at every segment
                #-> es gets a value (before = 0)
                    
            DUR_factor = 1
            h.dt = min(1/(dt_fact*freq) * 1e3, 0.025) #because h.dt is in ms, h.dt = 0.025 ms(default) when freq = 1e3 Hz
            h.DUR = int(np.ceil(max(5,DUR_factor*1/freq*1e3)))   # simulation should be minimally 5 ms (+ delay), low freq require higher sim times
                                        # h.DUR = 5 ms(default) when freq = 200 Hz (Tmin)
            h.tstop = h.DUR + h.DEL + 1 # +1 for safety
            #h.sine_freq = freq
            h.sine_freq_xtra = freq*1e-3
            print(f'freq = {freq}, h.dt = {h.dt}, h.DUR = {h.DUR}, h.tstop = {h.tstop}')
            h.finitialize()
            print("initialized")
            titr = tt.calcThreshSimpl(0,0,0,h.DEL,h.DUR,0)
            titr_freqs[i_f] = titr
            thresh_freqs[i_f] = titr * E_max_GM_99
            print(f'thresh = {thresh_freqs[i_f]}')
        results['hotspot'][i_p] = {'location': location, 'normal': normal, 'freq': freq_array, 'titr': titr_freqs, 'thresh': thresh_freqs}


    with open(PROJECT_ROOT+f"/npy_data/week6_26/{name}.pkl",'wb') as pf:
        pickle.dump(results,pf)

    time_end = datetime.datetime.now()
    time_diff = time_end-time_start
    print('total simulation time:  ',time_diff)