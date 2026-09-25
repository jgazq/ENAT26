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


# increases the stack in able to load cell_nr = 6
nrn_options = "-NSTACK 10000 -NFRAME 525"
# nrn_options = "-nogui -NSTACK 3000 -NFRAME 525"
os.environ["NEURON_MODULE_OPTIONS"] = nrn_options
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
if cell_nr == 2:
    cell = h.bNAC219_L1_NGCDA_e7cec642c3[0] # for cell = 2
elif cell_nr == 3:
    cell = h.bNAC219_L1_NGCDA_46b45974f4[0] # for cell = 3
elif cell_nr == 7:
    cell = h.cADpyr229_L23_PC_8ef1aa6602[0] # for cell = 7

cell_name = h.cell_names.o(cell_nr-1).s


current_time = datetime.datetime.now()
now = datetime.datetime.strftime(current_time,'%d_%m_%Y_%H_%M_%S')
PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))

#----------------------------------------------------------------------------------------------------------#
"flags"

createfsweepBunif = 0 #bool # this calculates the threshold for different frequencies for a uniform magnetic field (different Exyz depending on frequency)
save_coord_somas = 1
createfsweepBunif_SPFD = 0 #bool
createfsweepEAS = 0
convergence_dt_Bunif = 0
convergence_dur_Bunif = 0
createfsweeplocs = 0 #bool #this does a frequency sweep for different locations (updated)
createlocsweep = 0 #bool # this does a frequency sweep for different locations (deprecated)
titration = 0 #bool # plots titration factor if 1, otherwise actual stimulation
E_magsweep = 0 #bool #this calculates the threshold for a uniform E-field for different magnitudes

#----------------------------------------------------------------------------------------------------------#
"code"

#h.topology()
     
if createfsweepBunif: #similar to createfsweep but es calculation needs to happen inside freq iteration as es is different for different frequencies (<10kHz: 300Hz, >10kHz: 100 kHz)
    TMS = 0 #also use the right x86 folder and change save folder
    Eunif = 0 #also use the right x86 folder and change save folder
    Bunif = 1 #same nrnmech.dll file as TMS (or folder on linux) 

    #h.v_init = -75
    interpol = 'linear'
    logspace = (10,2,5)
    dt_fact = 100
    dx = 0.49997*1e3 #um

    time_start = datetime.datetime.now()
    #subject = 3 #parsed from command lines
    #direction = 'lateral' #anterior #lateral #superior #parsed from command line
    reverse = 0
    direction_map = direction
    if direction == 'medial':
        reverse = 1; direction_map = 'lateral'
    if direction == 'inferior':
        reverse = 1; direction_map = 'superior'
    if direction == 'posterior':
        reverse = 1; direction_map = 'anterior'

    name = "interpol_"+str(interpol)+"_logspace_"+str(logspace)+"_cell_nr_"+str(cell_nr)+"_dt_fact_"+str(dt_fact)+"_subject_"+str(subject)+"_direction_"+direction

    print(f'subject: {subject}, direction: {direction}')
    #h.cvode_active(1)

    numb_freq, logstart, logstop = logspace

    print('loading matf...')
    matf300,matf100k = mat73.loadmat(rf'Exyz\s{subject}_300Hz.mat'), mat73.loadmat(rf'Exyz\s{subject}_100kHz.mat') #"300Hz" if freq < 1e4 else "100KHz"
    print('loaded matf ✓')
    brainmap = matf300['TissueTypeIndices'] #should be the same as matf100k['TissueTypeIndices']
    E_x300, E_y300, E_z300 = matf300['E'][direction_map]['x'], matf300['E'][direction_map]['y'], matf300['E'][direction_map]['z']
    E_x100k, E_y100k, E_z100k = matf100k['E'][direction_map]['x'], matf100k['E'][direction_map]['y'], matf100k['E'][direction_map]['z']
    E_mag300 = np.sqrt(E_x300**2+E_y300**2+E_z300**2)
    E_mag100k = np.sqrt(E_x100k**2+E_y100k**2+E_z100k**2)
    E_mag300_GM = (E_mag300*(brainmap==16)).flatten()
    E_mag100k_GM = (E_mag100k*(brainmap==16)).flatten()
    E_max_GM300 = np.max(E_mag300*(brainmap==16))
    E_max_GM100k = np.max(E_mag100k*(brainmap==16))
    E_max_GM300_99 = np.percentile(E_mag300_GM[E_mag300_GM != 0],99)
    E_max_GM100k_99 = np.percentile(E_mag100k_GM[E_mag100k_GM != 0],99)
    print(f'300 Hz: max: {E_max_GM300} , 99 percentile: {E_max_GM300_99}')
    print(f'100 kHz: max: {E_max_GM100k} , 99 percentile: {E_max_GM100k_99}')
    E_max_GM300loc = np.unravel_index(np.argmax(E_mag300*(brainmap==16)),E_mag300.shape)
    E_max_GM100kloc = np.unravel_index(np.argmax(E_mag100k*(brainmap==16)),E_mag100k.shape)
    #meshname = 'layer_1_depth_0.06' if cell_nr < 6 else 'layer_23_depth_0.40' if cell_nr < 11 else 'layer_4_depth_0.55' if cell_nr < 16 else 'layer_5_depth_0.65' if cell_nr < 21 else 'layer_6_depth_0.85' if cell_nr <26 else 'ERROR'
    meshname = 'layer_1_depth_0.06' if 'L1' in cell_name else 'layer_23_depth_0.40' if 'L23' in cell_name else 'layer_4_depth_0.55' if 'L4' in cell_name else 'layer_5_depth_0.65' if 'L5' in cell_name else 'layer_6_depth_0.85' if 'L6' in cell_name else 'ERROR'
    print('loading trimesh...')
    surface_mesh = trimesh.load_mesh(rf"tissue_meshes\subjects\s{subject}\with holes ;(\layers\{meshname}.stl")
    print('loaded trimesh ✓')
    polygons300 = fcts.closest_meshes_to_point(surface_mesh,E_max_GM300loc)
    polygons100k = fcts.closest_meshes_to_point(surface_mesh,E_max_GM100kloc) if E_max_GM300loc != E_max_GM100kloc else []
    polygons = polygons300 + polygons100k
    
    results = {'hotspot_300': {key: {} for key in range(10)}, 'hotspot_100k': {key: {} for key in range(10)}}
    prev_normal = np.array([0,0,1])

    for i_p,polygon in enumerate(polygons[:10]):
        location, normal = polygon['centroid'], polygon['normal']
        _,x_values,y_values,z_values = fcts.es_matrix_matf(E_x300,E_y300,E_z300,dx,(0,0,0)) #E_x, E_y and E_z are only used for the shape in this case so doesn't matter which freq: 300 or 100k
        thresh_freqs = np.zeros(numb_freq)
        titr_freqs = np.zeros(numb_freq)
        freq_array = np.logspace(logstart,logstop,numb_freq)  

        for i_f,freq in enumerate(freq_array):

            if TMS or Bunif:
                h.getcoords() #gets the coordinates and also calculates D_x, D_y and D_z at every segment
                #-> D_x, D_y and D_z get a value (before = 0)
                tt.calcESext3(*location*dx,normal,np.array([0,0,1]),interpol,E_x300,E_y300,E_z300,x_values,y_values,z_values) if freq < 1e4 else calcESext3(*location*dx,normal,np.array([0,0,1]),interpol,E_x100k,E_y100k,E_z100k,x_values,y_values,z_values) #calculation of E_x, E_y and E_z at every segment
                #-> E_x, E_y and E_z get a value (before = 0)
                #h("load_potentials = 1")
                h.calc_pseudo_es()#h.getes2() #calculation of the potential at every segment
                #-> es gets a value (before = 0)
                if reverse: 
                    print('reversed')
                    for sec in h.allsec():
                        for seg in sec:
                            if "Scale" not in str(seg) and "Elec" not in str(seg):
                                seg.es_xtra = - seg.es_xtra
            DUR_factor = 1
            h.dt = min(1/(dt_fact*freq) * 1e3, 0.025) #because h.dt is in ms, h.dt = 0.025 ms(default) when freq = 1e3 Hz
            h.DUR = int(np.ceil(max(5,DUR_factor*1/freq*1e3)))   # simulation should be minimally 5 ms (+ delay), low freq require higher sim times
                                        # h.DUR = 5 ms(default) when freq = 200 Hz (Tmin)
            h.tstop = h.DUR + h.DEL + 1 # +1 for safety
            #h.sine_freq = freq
            h.sine_freq_xtra = freq*1e-3
            h.finitialize()

            titr = tt.calcThreshSimpl(0,0,0,h.DEL,h.DUR,0)
            titr_freqs[i_f] = titr
            thresh_freqs[i_f] = titr * E_max_GM300_99 if freq < 1e4 else titr * E_max_GM100k_99
            print(f'thresh = {thresh_freqs[i_f]}')
        results['hotspot_300'][i_p] = {'location': location, 'normal': normal, 'freq': freq_array, 'titr': titr_freqs, 'thresh': thresh_freqs}
    if E_max_GM300loc != E_max_GM100kloc:
        for i_p,polygon in enumerate(polygons[10:]):
            location, normal = polygon['centroid'], polygon['normal']
            print(f'location_{i_p}: {location}')
            _,x_values,y_values,z_values = es_matrix_matf(E_x300,E_y300,E_z300,dx,(0,0,0)) #E_x, E_y and E_z are only used for the shape in this case so doesn't matter which freq: 300 or 100k

            thresh_freqs = np.zeros(numb_freq)
            titr_freqs = np.zeros(numb_freq)
            freq_array = np.logspace(logstart,logstop,numb_freq)  

            for i_f,freq in enumerate(freq_array):
                print(f'frequency_{i_f}: {freq}')

                if TMS or Bunif:
                    h.getcoords() #gets the coordinates and also calculates D_x, D_y and D_z at every segment
                    #-> D_x, D_y and D_z get a value (before = 0)
                    tt.calcESext3(*location*dx,normal,np.array([0,0,1]),interpol,E_x300,E_y300,E_z300,x_values,y_values,z_values) if freq < 1e4 else calcESext3(*location*dx,normal,np.array([0,0,1]),interpol,E_x100k,E_y100k,E_z100k,x_values,y_values,z_values) #calculation of E_x, E_y and E_z at every segment
                    #-> E_x, E_y and E_z get a value (before = 0)
                    #h("load_potentials = 1")
                    h.calc_pseudo_es()#h.getes2() #calculation of the potential at every segment
                    #-> es gets a value (before = 0)
                    if reverse: 
                        print('reversed')
                        for sec in h.allsec():
                            for seg in sec:
                                if "Scale" not in str(seg) and "Elec" not in str(seg):
                                    seg.es_xtra = - seg.es_xtra

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
                thresh_freqs[i_f] = titr * E_max_GM300_99 if freq < 1e4 else titr * E_max_GM100k_99
                print(f'thresh = {thresh_freqs[i_f]}\n\n')
            results['hotspot_100k'][i_p] = {'location': location, 'normal': normal, 'freq': freq_array, 'titr': titr_freqs, 'thresh': thresh_freqs}
    else:
        results['hotspot_100k'] = copy.copy(results['hotspot_300'])

    time_end = datetime.datetime.now()
    time_diff = time_end-time_start
    print('total simulation time:  ',time_diff)


if save_coord_somas: #similar to createfsweep but es calculation needs to happen inside freq iteration as es is different for different frequencies (<10kHz: 300Hz, >10kHz: 100 kHz)
    TMS = 0 #also use the right x86 folder and change save folder
    Eunif = 0 #also use the right x86 folder and change save folder
    Bunif = 1 #same nrnmech.dll file as TMS (or folder on linux) 

    #h.v_init = -75
    interpol = 'linear'
    logspace = (10,2,5)
    dt_fact = 100
    dx = 0.49997*1e3 #um

    reverse = 0
    direction_map = direction
    if direction == 'medial':
        reverse = 1; direction_map = 'lateral'
    if direction == 'inferior':
        reverse = 1; direction_map = 'superior'
    if direction == 'posterior':
        reverse = 1; direction_map = 'anterior'

    name = "interpol_"+str(interpol)+"_logspace_"+str(logspace)+"_cell_nr_"+str(cell_nr)+"_dt_fact_"+str(dt_fact)+"_subject_"+str(subject)+"_direction_"+direction

    print(f'subject: {subject}, direction: {direction}')
    #h.cvode_active(1)

    print('loading matf...')
    matf300,matf100k = mat73.loadmat(rf'Exyz\s{subject}_300Hz.mat'), mat73.loadmat(rf'Exyz\s{subject}_100kHz.mat') #"300Hz" if freq < 1e4 else "100KHz"
    print('loaded matf ✓')
    brainmap = matf300['TissueTypeIndices'] #should be the same as matf100k['TissueTypeIndices']
    E_x300, E_y300, E_z300 = matf300['E'][direction_map]['x'], matf300['E'][direction_map]['y'], matf300['E'][direction_map]['z']
    E_mag300 = np.sqrt(E_x300**2+E_y300**2+E_z300**2)
    E_mag300_GM = (E_mag300*(brainmap==16)).flatten()
    E_max_GM300 = np.max(E_mag300*(brainmap==16))
    E_max_GM300_99 = np.percentile(E_mag300_GM[E_mag300_GM != 0],99)
    print(f'300 Hz: max: {E_max_GM300} , 99 percentile: {E_max_GM300_99}')
    E_max_GM300loc = np.unravel_index(np.argmax(E_mag300*(brainmap==16)),E_mag300.shape)
    #meshname = 'layer_1_depth_0.06' if cell_nr < 6 else 'layer_23_depth_0.40' if cell_nr < 11 else 'layer_4_depth_0.55' if cell_nr < 16 else 'layer_5_depth_0.65' if cell_nr < 21 else 'layer_6_depth_0.85' if cell_nr <26 else 'ERROR'
    meshname = 'layer_1_depth_0.06' if 'L1' in cell_name else 'layer_23_depth_0.40' if 'L23' in cell_name else 'layer_4_depth_0.55' if 'L4' in cell_name else 'layer_5_depth_0.65' if 'L5' in cell_name else 'layer_6_depth_0.85' if 'L6' in cell_name else 'ERROR'
    print('loading trimesh...')
    surface_mesh = trimesh.load_mesh(rf"tissue_meshes\subjects\s{subject}\layers\{meshname}.stl")
    print('loaded trimesh ✓')
    polygons300 = fcts.closest_meshes_to_point(surface_mesh,E_max_GM300loc)
    polygons = polygons300
    
    results = {'hotspot_300': {key: {} for key in range(10)}, 'hotspot_100k': {key: {} for key in range(10)}}
    prev_normal = np.array([0,0,1])

    locations = np.empty((10,3))

    for i_p,polygon in enumerate(polygons[:10]):
        location, normal = polygon['centroid'], polygon['normal']
        _,x_values,y_values,z_values = fcts.es_matrix_matf(E_x300,E_y300,E_z300,dx,(0,0,0)) #E_x, E_y and E_z are only used for the shape in this case so doesn't matter which freq: 300 or 100k

        h.getcoords() #gets the coordinates and also calculates D_x, D_y and D_z at every segment
        #-> D_x, D_y and D_z get a value (before = 0)
        locations[i_p] = tt.calcESext4(*location*dx,normal,np.array([0,0,1]),interpol,E_x300,E_y300,E_z300,x_values,y_values,z_values) #calculation of E_x, E_y and E_z at every segment

    print(locations)
    with open(rf"tissue_meshes\subjects\s3\layers\somas.pkl",'rb') as pf:
        pklsomas = pickle.load(pf)
    #pklsomas = {}
    print(pklsomas)
    pklsomas[cell_nr] = locations
    print(pklsomas)
    with open(rf"tissue_meshes\subjects\s3\layers\somas.pkl",'wb') as pf:
        pklsomas = pickle.dump(pklsomas,pf)


if createfsweepBunif_SPFD: #similar to createfsweep but es calculation needs to happen inside freq iteration as es is different for different frequencies (<10kHz: 300Hz, >10kHz: 100 kHz)
    TMS = 0 #also use the right x86 folder and change save folder
    Eunif = 0 #also use the right x86 folder and change save folder
    Bunif = 1 #same nrnmech.dll file as TMS (or folder on linux) 

    #h.v_init = -75
    interpol = 'linear'
    logspace = (10,2,5)
    dt_fact = 100
    dx = 0.49997*1e3 #um

    time_start = datetime.datetime.now()
    #subject = 3 #parsed from command lines
    #direction = 'lateral' #anterior #lateral #superior #parsed from command line
    reverse = 0
    direction_map = direction
    if direction == 'medial':
        reverse = 1; direction_map = 'lateral'
    if direction == 'inferior':
        reverse = 1; direction_map = 'superior'
    if direction == 'posterior':
        reverse = 1; direction_map = 'anterior'

    name = "interpol_"+str(interpol)+"_logspace_"+str(logspace)+"_cell_nr_"+str(cell_nr)+"_dt_fact_"+str(dt_fact)+"_subject_"+str(subject)+"_direction_"+direction

    print(f'subject: {subject}, direction: {direction}')
    #h.cvode_active(1)

    numb_freq, logstart, logstop = logspace

    print('loading matf...')
    matf300,matf100k = mat73.loadmat(rf'Exyz/s{subject}_300Hz.mat'), mat73.loadmat(rf'Exyz/s{subject}_100kHz.mat') #"300Hz" if freq < 1e4 else "100KHz"
    print('loading matf_SPFD...')
    matf300_SPFD,matf100k_SPFD = mat73.loadmat(rf'Exyz_SPFD/s{subject}_300Hz_SPFD.mat'), mat73.loadmat(rf'Exyz_SPFD/s{subject}_100kHz_SPFD.mat') #"300Hz" if freq < 1e4 else "100KHz"
    print('loaded matf ✓')
    brainmap = matf300['TissueTypeIndices'] #should be the same as matf100k['TissueTypeIndices']
    E_x300, E_y300, E_z300 = matf300_SPFD['E_SPFD'][direction_map]['x'], matf300_SPFD['E_SPFD'][direction_map]['y'], matf300_SPFD['E_SPFD'][direction_map]['z']
    E_x100k, E_y100k, E_z100k = matf100k_SPFD['E_SPFD'][direction_map]['x'], matf100k_SPFD['E_SPFD'][direction_map]['y'], matf100k_SPFD['E_SPFD'][direction_map]['z']
    E_mag300 = np.sqrt(E_x300**2+E_y300**2+E_z300**2)
    E_mag100k = np.sqrt(E_x100k**2+E_y100k**2+E_z100k**2)
    E_mag300_GM = (E_mag300*(brainmap==16)).flatten()
    E_mag100k_GM = (E_mag100k*(brainmap==16)).flatten()
    E_max_GM300 = np.max(E_mag300*(brainmap==16))
    E_max_GM100k = np.max(E_mag100k*(brainmap==16))
    E_max_GM300_99 = np.percentile(E_mag300_GM[E_mag300_GM != 0],99)
    E_max_GM100k_99 = np.percentile(E_mag100k_GM[E_mag100k_GM != 0],99)
    print(f'300 Hz: max: {E_max_GM300} , 99 percentile: {E_max_GM300_99}')
    print(f'100 kHz: max: {E_max_GM100k} , 99 percentile: {E_max_GM100k_99}')
    E_max_GM300loc = np.unravel_index(np.argmax(E_mag300*(brainmap==16)),E_mag300.shape)
    E_max_GM100kloc = np.unravel_index(np.argmax(E_mag100k*(brainmap==16)),E_mag100k.shape)
    #meshname = 'layer_1_depth_0.06' if cell_nr < 6 else 'layer_23_depth_0.40' if cell_nr < 11 else 'layer_4_depth_0.55' if cell_nr < 16 else 'layer_5_depth_0.65' if cell_nr < 21 else 'layer_6_depth_0.85' if cell_nr <26 else 'ERROR'
    meshname = 'layer_1_depth_0.06' if 'L1' in cell_name else 'layer_23_depth_0.40' if 'L23' in cell_name else 'layer_4_depth_0.55' if 'L4' in cell_name else 'layer_5_depth_0.65' if 'L5' in cell_name else 'layer_6_depth_0.85' if 'L6' in cell_name else 'ERROR'
    print('loading trimesh...')
    surface_mesh = trimesh.load_mesh(rf"tissue_meshes/subjects/s{subject}/layers/{meshname}.stl")
    print('loaded trimesh ✓')
    polygons300 = fcts.closest_meshes_to_point(surface_mesh,E_max_GM300loc)
    polygons100k = fcts.closest_meshes_to_point(surface_mesh,E_max_GM100kloc) if E_max_GM300loc != E_max_GM100kloc else []
    polygons = polygons300 + polygons100k
    
    results = {'hotspot_300': {key: {} for key in range(10)}, 'hotspot_100k': {key: {} for key in range(10)}}
    prev_normal = np.array([0,0,1])

    for i_p,polygon in enumerate(polygons[:10]):
        location, normal = polygon['centroid'], polygon['normal']
        print(f'location_{i_p}: {location}, {normal} ({subject},{direction})')
        _,x_values,y_values,z_values = fcts.es_matrix_matf(E_x300,E_y300,E_z300,dx,(0,0,0)) #E_x, E_y and E_z are only used for the shape in this case so doesn't matter which freq: 300 or 100k
        thresh_freqs = np.zeros(numb_freq)
        titr_freqs = np.zeros(numb_freq)
        freq_array = np.logspace(logstart,logstop,numb_freq)  

        for i_f,freq in enumerate(freq_array):
            print(f'frequency_{i_f}: {freq}')

            if TMS or Bunif:
                h.getcoords() #gets the coordinates and also calculates D_x, D_y and D_z at every segment
                #-> D_x, D_y and D_z get a value (before = 0)
                tt.calcESext3(*location*dx,normal,np.array([0,0,1]),interpol,E_x300,E_y300,E_z300,x_values,y_values,z_values) if freq < 1e4 else calcESext3(*location*dx,normal,np.array([0,0,1]),interpol,E_x100k,E_y100k,E_z100k,x_values,y_values,z_values) #calculation of E_x, E_y and E_z at every segment
                #-> E_x, E_y and E_z get a value (before = 0)
                #h("load_potentials = 1")
                h.calc_pseudo_es()#h.getes2() #calculation of the potential at every segment
                #-> es gets a value (before = 0)
                if reverse: 
                    print('reversed')
                    for sec in h.allsec():
                        for seg in sec:
                            if "Scale" not in str(seg) and "Elec" not in str(seg):
                                seg.es_xtra = - seg.es_xtra
                    
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
            thresh_freqs[i_f] = titr * E_max_GM300_99 if freq < 1e4 else titr * E_max_GM100k_99
            print(f'thresh = {thresh_freqs[i_f]}')
        results['hotspot_300'][i_p] = {'location': location, 'normal': normal, 'freq': freq_array, 'titr': titr_freqs, 'thresh': thresh_freqs}
    if E_max_GM300loc != E_max_GM100kloc:
        for i_p,polygon in enumerate(polygons[10:]):
            location, normal = polygon['centroid'], polygon['normal']
            print(f'location_{i_p}: {location}')
            _,x_values,y_values,z_values = fcts.es_matrix_matf(E_x300,E_y300,E_z300,dx,(0,0,0)) #E_x, E_y and E_z are only used for the shape in this case so doesn't matter which freq: 300 or 100k

            thresh_freqs = np.zeros(numb_freq)
            titr_freqs = np.zeros(numb_freq)
            freq_array = np.logspace(logstart,logstop,numb_freq)  

            for i_f,freq in enumerate(freq_array):
                print(f'frequency_{i_f}: {freq}')

                if TMS or Bunif:
                    h.getcoords() #gets the coordinates and also calculates D_x, D_y and D_z at every segment
                    #-> D_x, D_y and D_z get a value (before = 0)
                    tt.calcESext3(*location*dx,normal,np.array([0,0,1]),interpol,E_x300,E_y300,E_z300,x_values,y_values,z_values) if freq < 1e4 else calcESext3(*location*dx,normal,np.array([0,0,1]),interpol,E_x100k,E_y100k,E_z100k,x_values,y_values,z_values) #calculation of E_x, E_y and E_z at every segment
                    #-> E_x, E_y and E_z get a value (before = 0)
                    #h("load_potentials = 1")
                    h.calc_pseudo_es()#h.getes2() #calculation of the potential at every segment
                    #-> es gets a value (before = 0)
                    if reverse: 
                        print('reversed')
                        for sec in h.allsec():
                            for seg in sec:
                                if "Scale" not in str(seg) and "Elec" not in str(seg):
                                    seg.es_xtra = - seg.es_xtra

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
                thresh_freqs[i_f] = titr * E_max_GM300_99 if freq < 1e4 else titr * E_max_GM100k_99
                print(f'thresh = {thresh_freqs[i_f]}\n\n')
            results['hotspot_100k'][i_p] = {'location': location, 'normal': normal, 'freq': freq_array, 'titr': titr_freqs, 'thresh': thresh_freqs}
    else:
        results['hotspot_100k'] = copy.copy(results['hotspot_300'])
    with open(f"results/SPFD/{name}.pkl",'wb') as pf:
        pickle.dump(results,pf)

    time_end = datetime.datetime.now()
    time_diff = time_end-time_start
    print('total simulation time:  ',time_diff)


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

    name = "EAS_"+subject+"_interpol_"+str(interpol)+"_logspace_"+str(logspace)+"_cell_nr_"+str(cell_nr)+"_dt_fact_"+str(dt_fact)

    numb_freq, logstart, logstop = logspace

    print('loading matf...')
    matf = sio.loadmat(rf'Exyz_EAS/{subject}_head_E-field.mat')
    print('loaded matf ✓')

    sh0, sh1, sh2 = matf['Axis0'][0,:].shape[0] -1, matf['Axis1'][0,:].shape[0] -1, matf['Axis2'][0,:].shape[0] -1
    Exyz = np.nan_to_num(matf['Snapshot0'].reshape(sh2,sh1,sh0,3))
    Exyz = np.nan_to_num(np.abs(matf['Snapshot0'].reshape(sh2,sh1,sh0,3)))

    #tranpose this already
    Exyz_tr = np.transpose(Exyz, (1, 2, 0, 3))
    Exyz_trfl= np.flip(Exyz_tr, axis=1)

    E_x, E_y, E_z = Exyz_trfl[:,:,:,0], Exyz_trfl[:,:,:,1], Exyz_trfl[:,:,:,2]
    E_mag = np.sqrt(E_x**2+E_y**2+E_z**2)

    #E_mag_trfl = np.sqrt(E_x**2+E_y**2+E_z**2)
    #E_mag_fl = np.transpose(E_mag_trfl, (1, 2, 0))
    #E_mag_nan = np.flip(E_mag_fl, axis=1)
    #E_mag = np.nan_to_num(E_mag_nan)

    data = np.fromfile(rf'Exyz_EAS/{subject}_head_voxels.raw', dtype=np.uint8)
    brainmap_0 = data.reshape((sh2,sh1,sh0,2))
    brainmap_trfl = brainmap_0[:,:,:,0]
    brainmap_fl = np.transpose(brainmap_trfl, (1, 2, 0))
    brainmap = np.flip(brainmap_fl, axis=1)

    E_mag_GM = (E_mag*(brainmap==GMi)).flatten()
    E_max_GM = np.max(E_mag*(brainmap==GMi))
    E_max_GM_99 = np.percentile(E_mag_GM[E_mag_GM != 0],99)
    E_max_GM_99_9 = np.percentile(E_mag_GM[E_mag_GM != 0],99.9)
    E_max_GM_100 = np.percentile(E_mag_GM[E_mag_GM != 0],100)
    print(f'max: {E_max_GM} , 99 percentile: {E_max_GM_99}, 99.9 percentile: {E_max_GM_99_9}, 100 percentile: {E_max_GM_100}')
    E_max_GMloc = np.unravel_index(np.argmax(E_mag*(brainmap==GMi)),E_mag.shape)
    print(f'max @ {E_max_GMloc}')
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
        _,x_values,y_values,z_values = es_matrix_matf(E_x,E_y,E_z,dx,(0,0,0),dy=dy,dz=dz)

        freq = 1700

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
        print(f'freq = {freq}, h.dt = {h.dt}, h.DUR = {h.DUR}, h.tstop = {h.tstop}')
        h.finitialize()
        print("initialized")
        titr = tt.calcThreshSimpl(0,0,0,h.DEL,h.DUR,0)
        thresh = titr * E_max_GM_99 * h.A_mV_xtra
        print(f'titr = {titr}')
        print(E_max_GM_99)
        print(h.A_mV_xtra)
        print(f'thresh = {thresh}')
        results['hotspot'][i_p] = {'location': location, 'normal': normal, 'freq': freq, 'titr': titr, 'thresh': thresh}


    with open(f"results/FEM/{name}.pkl",'wb') as pf:
        pickle.dump(results,pf)

    time_end = datetime.datetime.now()
    time_diff = time_end-time_start
    print('total simulation time:  ',time_diff)


if convergence_dt_Bunif:
    TMS = 0
    Bunif = 1 #same nrnmech.dll file as TMS (or folder on linux) 

    #for periods
    #np.outer(np.logpsace(-1,1,3),[1,2,5]) [periods]
    #np.outer(np.logspace(0,1,2),[1,2,5])

    #h.v_init = -75
    interpol = 'linear'
    logspace = (3,2,4)
    dx = 0.49997*1e3 #um

    time_start = datetime.datetime.now()
    #subject = 3 #parsed from command lines
    #direction = 'lateral' #anterior #lateral #superior #parsed from command line
    reverse = 0
    direction_map = direction
    if direction == 'medial':
        reverse = 1; direction_map = 'lateral'
    if direction == 'inferior':
        reverse = 1; direction_map = 'superior'
    if direction == 'posterior':
        reverse = 1; direction_map = 'anterior'

    name = "convergence_dt_fact_"+"interpol_"+str(interpol)+"_logspace_"+str(logspace)+"_cell_nr_"+str(cell_nr)+"_subject_"+str(subject)+"_direction_"+direction

    print(f'subject: {subject}, direction: {direction}')
    #h.cvode_active(1)

    numb_freq, logstart, logstop = logspace

    print('loading matf...')
    matf300,matf100k = mat73.loadmat(rf'Exyz\s{subject}_300Hz.mat'), mat73.loadmat(rf'Exyz\s{subject}_100kHz.mat') #"300Hz" if freq < 1e4 else "100KHz"
    print('loaded matf ✓')
    brainmap = matf300['TissueTypeIndices'] #should be the same as matf100k['TissueTypeIndices']
    E_x300, E_y300, E_z300 = matf300['E'][direction_map]['x'], matf300['E'][direction_map]['y'], matf300['E'][direction_map]['z']
    E_x100k, E_y100k, E_z100k = matf100k['E'][direction_map]['x'], matf100k['E'][direction_map]['y'], matf100k['E'][direction_map]['z']
    E_mag300 = np.sqrt(E_x300**2+E_y300**2+E_z300**2)
    E_mag100k = np.sqrt(E_x100k**2+E_y100k**2+E_z100k**2)
    E_mag300_GM = (E_mag300*(brainmap==16)).flatten()
    E_mag100k_GM = (E_mag100k*(brainmap==16)).flatten()
    E_max_GM300 = np.max(E_mag300*(brainmap==16))
    E_max_GM100k = np.max(E_mag100k*(brainmap==16))
    E_max_GM300_99 = np.percentile(E_mag300_GM[E_mag300_GM != 0],99)
    E_max_GM100k_99 = np.percentile(E_mag100k_GM[E_mag100k_GM != 0],99)
    print(f'300 Hz: max: {E_max_GM300} , 99 percentile: {E_max_GM300_99}')
    print(f'100 kHz: max: {E_max_GM100k} , 99 percentile: {E_max_GM100k_99}')
    E_max_GM300loc = np.unravel_index(np.argmax(E_mag300*(brainmap==16)),E_mag300.shape)
    E_max_GM100kloc = np.unravel_index(np.argmax(E_mag100k*(brainmap==16)),E_mag100k.shape)
    #meshname = 'layer_1_depth_0.06' if cell_nr < 6 else 'layer_23_depth_0.40' if cell_nr < 11 else 'layer_4_depth_0.55' if cell_nr < 16 else 'layer_5_depth_0.65' if cell_nr < 21 else 'layer_6_depth_0.85' if cell_nr <26 else 'ERROR'
    meshname = 'layer_1_depth_0.06' if 'L1' in cell_name else 'layer_23_depth_0.40' if 'L23' in cell_name else 'layer_4_depth_0.55' if 'L4' in cell_name else 'layer_5_depth_0.65' if 'L5' in cell_name else 'layer_6_depth_0.85' if 'L6' in cell_name else 'ERROR'
    print('loading trimesh...')
    surface_mesh = trimesh.load_mesh(rf"tissue_meshes\subjects\s{subject}\layers\{meshname}.stl")
    print('loaded trimesh ✓')
    polygons300 = fcts.closest_meshes_to_point(surface_mesh,E_max_GM300loc)
    polygons100k = fcts.closest_meshes_to_point(surface_mesh,E_max_GM100kloc) if E_max_GM300loc != E_max_GM100kloc else []
    polygons = polygons300 + polygons100k
    
    results = {'hotspot_300': {key: {} for key in range(10)}, 'hotspot_100k': {key: {} for key in range(10)}}
    prev_normal = np.array([0,0,1])

    for dt_fact in np.round(np.logspace(np.log10(20),np.log10(500),13)):
        polygon = polygons[0]
        location, normal = polygon['centroid'], polygon['normal']
        print(f'location: {location}, {normal} ({subject},{direction})')
        _,x_values,y_values,z_values = fcts.es_matrix_matf(E_x300,E_y300,E_z300,dx,(0,0,0)) #E_x, E_y and E_z are only used for the shape in this case so doesn't matter which freq: 300 or 100k
        
        freq_array = np.logspace(logstart,logstop,numb_freq)  
        freq_array = [100.0, 400.0, 1000.0, 10000.0, 100000.0]
        thresh_freqs = np.zeros(len(freq_array))
        titr_freqs = np.zeros(len(freq_array))

        for i_f,freq in enumerate(freq_array):
            print(f'frequency_{i_f}: {freq}')

            if TMS or Bunif:
                h.getcoords() #gets the coordinates and also calculates D_x, D_y and D_z at every segment
                #-> D_x, D_y and D_z get a value (before = 0)
                tt.calcESext3(*location*dx,normal,np.array([0,0,1]),interpol,E_x300,E_y300,E_z300,x_values,y_values,z_values) if freq < 1e4 else calcESext3(*location*dx,normal,np.array([0,0,1]),interpol,E_x100k,E_y100k,E_z100k,x_values,y_values,z_values) #calculation of E_x, E_y and E_z at every segment
                #-> E_x, E_y and E_z get a value (before = 0)
                #h("load_potentials = 1")
                h.calc_pseudo_es()#h.getes2() #calculation of the potential at every segment
                #-> es gets a value (before = 0)
                if reverse: 
                    print('reversed')
                    for sec in h.allsec():
                        for seg in sec:
                            if "Scale" not in str(seg) and "Elec" not in str(seg):
                                seg.es_xtra = - seg.es_xtra
                    
            DUR_factor = 1
            h.dt = 1/(dt_fact*freq) * 1e3#min(1/(dt_fact*freq) * 1e3, 0.025) #because h.dt is in ms, h.dt = 0.025 ms(default) when freq = 1e3 Hz
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
            thresh_freqs[i_f] = titr * E_max_GM300_99 if freq < 1e4 else titr * E_max_GM100k_99
            print(f'thresh = {thresh_freqs[i_f]}')
        results['hotspot_300'][dt_fact] = {'location': location, 'normal': normal, 'freq': freq_array, 'titr': titr_freqs, 'thresh': thresh_freqs}
    if E_max_GM300loc != E_max_GM100kloc:
        for dt_fact in np.round(np.logspace(np.log10(20),np.log10(500),13)):
            polygon = polygons[10]
            location, normal = polygon['centroid'], polygon['normal']
            print(f'location: {location}')
            _,x_values,y_values,z_values = fcts.es_matrix_matf(E_x300,E_y300,E_z300,dx,(0,0,0)) #E_x, E_y and E_z are only used for the shape in this case so doesn't matter which freq: 300 or 100k
            
            freq_array = np.logspace(logstart,logstop,numb_freq)  
            freq_array = [100.0, 400.0, 1000.0, 10000.0, 100000.0]
            thresh_freqs = np.zeros(len(freq_array))
            titr_freqs = np.zeros(len(freq_array))

            for i_f,freq in enumerate(freq_array):
                print(f'frequency_{i_f}: {freq}')

                if TMS or Bunif:
                    h.getcoords() #gets the coordinates and also calculates D_x, D_y and D_z at every segment
                    #-> D_x, D_y and D_z get a value (before = 0)
                    tt.calcESext3(*location*dx,normal,np.array([0,0,1]),interpol,E_x300,E_y300,E_z300,x_values,y_values,z_values) if freq < 1e4 else calcESext3(*location*dx,normal,np.array([0,0,1]),interpol,E_x100k,E_y100k,E_z100k,x_values,y_values,z_values) #calculation of E_x, E_y and E_z at every segment
                    #-> E_x, E_y and E_z get a value (before = 0)
                    #h("load_potentials = 1")
                    h.calc_pseudo_es()#h.getes2() #calculation of the potential at every segment
                    #-> es gets a value (before = 0)
                    if reverse: 
                        print('reversed')
                        for sec in h.allsec():
                            for seg in sec:
                                if "Scale" not in str(seg) and "Elec" not in str(seg):
                                    seg.es_xtra = - seg.es_xtra

                DUR_factor = 1
                h.dt = 1/(dt_fact*freq) * 1e3 #min(1/(dt_fact*freq) * 1e3, 0.025) #because h.dt is in ms, h.dt = 0.025 ms(default) when freq = 1e3 Hz
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
                thresh_freqs[i_f] = titr * E_max_GM300_99 if freq < 1e4 else titr * E_max_GM100k_99
                print(f'thresh = {thresh_freqs[i_f]}\n\n')
            results['hotspot_100k'][dt_fact] = {'location': location, 'normal': normal, 'freq': freq_array, 'titr': titr_freqs, 'thresh': thresh_freqs}
    else:
        results['hotspot_100k'] = copy.copy(results['hotspot_300'])
    with open(f"results/FEM/{name}.pkl",'wb') as pf:
        pickle.dump(results,pf)

    time_end = datetime.datetime.now()
    time_diff = time_end-time_start
    print('total simulation time:  ',time_diff)


if convergence_dur_Bunif:
    TMS = 0
    Bunif = 1 #same nrnmech.dll file as TMS (or folder on linux) 

    #h.v_init = -75
    interpol = 'linear'
    logspace = (3,2,4)
    dx = 0.49997*1e3 #um
    dt_fact = 100

    time_start = datetime.datetime.now()
    #subject = 3 #parsed from command lines
    #direction = 'lateral' #anterior #lateral #superior #parsed from command line
    reverse = 0
    direction_map = direction
    if direction == 'medial':
        reverse = 1; direction_map = 'lateral'
    if direction == 'inferior':
        reverse = 1; direction_map = 'superior'
    if direction == 'posterior':
        reverse = 1; direction_map = 'anterior'

    name = "convergence_dur_fact_"+"interpol_"+str(interpol)+"_logspace_"+str(logspace)+"_cell_nr_"+str(cell_nr)+"_subject_"+str(subject)+"_direction_"+direction

    print(f'subject: {subject}, direction: {direction}')
    #h.cvode_active(1)

    numb_freq, logstart, logstop = logspace

    print('loading matf...')
    matf300,matf100k = mat73.loadmat(rf'Exyz\s{subject}_300Hz.mat'), mat73.loadmat(rf'Exyz\s{subject}_100kHz.mat') #"300Hz" if freq < 1e4 else "100KHz"
    print('loaded matf ✓')
    brainmap = matf300['TissueTypeIndices'] #should be the same as matf100k['TissueTypeIndices']
    E_x300, E_y300, E_z300 = matf300['E'][direction_map]['x'], matf300['E'][direction_map]['y'], matf300['E'][direction_map]['z']
    E_x100k, E_y100k, E_z100k = matf100k['E'][direction_map]['x'], matf100k['E'][direction_map]['y'], matf100k['E'][direction_map]['z']
    E_mag300 = np.sqrt(E_x300**2+E_y300**2+E_z300**2)
    E_mag100k = np.sqrt(E_x100k**2+E_y100k**2+E_z100k**2)
    E_mag300_GM = (E_mag300*(brainmap==16)).flatten()
    E_mag100k_GM = (E_mag100k*(brainmap==16)).flatten()
    E_max_GM300 = np.max(E_mag300*(brainmap==16))
    E_max_GM100k = np.max(E_mag100k*(brainmap==16))
    E_max_GM300_99 = np.percentile(E_mag300_GM[E_mag300_GM != 0],99)
    E_max_GM100k_99 = np.percentile(E_mag100k_GM[E_mag100k_GM != 0],99)
    print(f'300 Hz: max: {E_max_GM300} , 99 percentile: {E_max_GM300_99}')
    print(f'100 kHz: max: {E_max_GM100k} , 99 percentile: {E_max_GM100k_99}')
    E_max_GM300loc = np.unravel_index(np.argmax(E_mag300*(brainmap==16)),E_mag300.shape)
    E_max_GM100kloc = np.unravel_index(np.argmax(E_mag100k*(brainmap==16)),E_mag100k.shape)
    #meshname = 'layer_1_depth_0.06' if cell_nr < 6 else 'layer_23_depth_0.40' if cell_nr < 11 else 'layer_4_depth_0.55' if cell_nr < 16 else 'layer_5_depth_0.65' if cell_nr < 21 else 'layer_6_depth_0.85' if cell_nr <26 else 'ERROR'
    meshname = 'layer_1_depth_0.06' if 'L1' in cell_name else 'layer_23_depth_0.40' if 'L23' in cell_name else 'layer_4_depth_0.55' if 'L4' in cell_name else 'layer_5_depth_0.65' if 'L5' in cell_name else 'layer_6_depth_0.85' if 'L6' in cell_name else 'ERROR'
    print('loading trimesh...')
    surface_mesh = trimesh.load_mesh(rf"tissue_meshes\subjects\s{subject}\layers\{meshname}.stl")
    print('loaded trimesh ✓')
    polygons300 = fcts.closest_meshes_to_point(surface_mesh,E_max_GM300loc)
    polygons100k = fcts.closest_meshes_to_point(surface_mesh,E_max_GM100kloc) if E_max_GM300loc != E_max_GM100kloc else []
    polygons = polygons300 + polygons100k
    
    results = {'hotspot_300': {key: {} for key in range(10)}, 'hotspot_100k': {key: {} for key in range(10)}}
    prev_normal = np.array([0,0,1])

    for dur_fact in np.outer(np.logspace(-1,1,3),[1,2,5]).flatten():
        polygon = polygons[0]
        location, normal = polygon['centroid'], polygon['normal']
        print(f'location: {location}, {normal} ({subject},{direction})')
        _,x_values,y_values,z_values = fcts.es_matrix_matf(E_x300,E_y300,E_z300,dx,(0,0,0)) #E_x, E_y and E_z are only used for the shape in this case so doesn't matter which freq: 300 or 100k
        
        freq_array = np.logspace(logstart,logstop,numb_freq)  
        freq_array = [100.0, 200.0, 1000.0, 10000.0, 100000.0]
        thresh_freqs = np.zeros(len(freq_array))
        titr_freqs = np.zeros(len(freq_array))

        for i_f,freq in enumerate(freq_array):
            print(f'frequency_{i_f}: {freq}')

            if TMS or Bunif:
                h.getcoords() #gets the coordinates and also calculates D_x, D_y and D_z at every segment
                #-> D_x, D_y and D_z get a value (before = 0)
                tt.calcESext3(*location*dx,normal,np.array([0,0,1]),interpol,E_x300,E_y300,E_z300,x_values,y_values,z_values) if freq < 1e4 else calcESext3(*location*dx,normal,np.array([0,0,1]),interpol,E_x100k,E_y100k,E_z100k,x_values,y_values,z_values) #calculation of E_x, E_y and E_z at every segment
                #-> E_x, E_y and E_z get a value (before = 0)
                #h("load_potentials = 1")
                h.calc_pseudo_es()#h.getes2() #calculation of the potential at every segment
                #-> es gets a value (before = 0)
                if reverse: 
                    print('reversed')
                    for sec in h.allsec():
                        for seg in sec:
                            if "Scale" not in str(seg) and "Elec" not in str(seg):
                                seg.es_xtra = - seg.es_xtra
                    
            h.dt = min(1/(dt_fact*freq) * 1e3, 0.025) #because h.dt is in ms, h.dt = 0.025 ms(default) when freq = 1e3 Hz (when using 40 points per wavelength)
            h.DUR = dur_fact*1/freq*1e3  # simulation should be minimally 5 ms (+ delay), low freq require higher sim times
                                        # h.DUR = 5 ms(default) when freq = 200 Hz (Tmin) (when using 1 period of duration)
            h.tstop = h.DUR + h.DEL + 1 # +1 for safety
            #h.sine_freq = freq
            h.sine_freq_xtra = freq*1e-3
            print(f'freq = {freq}, h.dt = {h.dt}, h.DUR = {h.DUR}, h.tstop = {h.tstop}')
            h.finitialize()
            print("initialized")
            titr = tt.calcThreshSimpl(0,0,0,h.DEL,h.DUR,0)
            titr_freqs[i_f] = titr
            thresh_freqs[i_f] = titr * E_max_GM300_99 if freq < 1e4 else titr * E_max_GM100k_99
            print(f'thresh = {thresh_freqs[i_f]}')
        results['hotspot_300'][dur_fact] = {'location': location, 'normal': normal, 'freq': freq_array, 'titr': titr_freqs, 'thresh': thresh_freqs}
    if E_max_GM300loc != E_max_GM100kloc:
        for dur_fact in np.outer(np.logspace(-1,1,3),[1,2,5]).flatten():
            polygon = polygons[10]
            location, normal = polygon['centroid'], polygon['normal']
            print(f'location: {location}')
            _,x_values,y_values,z_values = fcts.es_matrix_matf(E_x300,E_y300,E_z300,dx,(0,0,0)) #E_x, E_y and E_z are only used for the shape in this case so doesn't matter which freq: 300 or 100k
            
            freq_array = np.logspace(logstart,logstop,numb_freq)
            freq_array = [100.0, 200.0, 1000.0, 10000.0, 100000.0]
            thresh_freqs = np.zeros(len(freq_array))
            titr_freqs = np.zeros(len(freq_array))

            for i_f,freq in enumerate(freq_array):
                print(f'frequency_{i_f}: {freq}')

                if TMS or Bunif:
                    h.getcoords() #gets the coordinates and also calculates D_x, D_y and D_z at every segment
                    #-> D_x, D_y and D_z get a value (before = 0)
                    tt.calcESext3(*location*dx,normal,np.array([0,0,1]),interpol,E_x300,E_y300,E_z300,x_values,y_values,z_values) if freq < 1e4 else calcESext3(*location*dx,normal,np.array([0,0,1]),interpol,E_x100k,E_y100k,E_z100k,x_values,y_values,z_values) #calculation of E_x, E_y and E_z at every segment
                    #-> E_x, E_y and E_z get a value (before = 0)
                    #h("load_potentials = 1")
                    h.calc_pseudo_es()#h.getes2() #calculation of the potential at every segment
                    #-> es gets a value (before = 0)
                    if reverse: 
                        print('reversed')
                        for sec in h.allsec():
                            for seg in sec:
                                if "Scale" not in str(seg) and "Elec" not in str(seg):
                                    seg.es_xtra = - seg.es_xtra

                h.dt = min(1/(dt_fact*freq) * 1e3, 0.025) #because h.dt is in ms, h.dt = 0.025 ms(default) when freq = 1e3 Hz
                h.DUR = dur_fact*1/freq*1e3  # simulation should be minimally 5 ms (+ delay), low freq require higher sim times
                                            # h.DUR = 5 ms(default) when freq = 200 Hz (Tmin)
                h.tstop = h.DUR + h.DEL + 1 # +1 for safety
                #h.sine_freq = freq
                h.sine_freq_xtra = freq*1e-3
                print(f'freq = {freq}, h.dt = {h.dt}, h.DUR = {h.DUR}, h.tstop = {h.tstop}')
                h.finitialize()
                print("initialized")
                titr = tt.calcThreshSimpl(0,0,0,h.DEL,h.DUR,0)
                titr_freqs[i_f] = titr
                thresh_freqs[i_f] = titr * E_max_GM300_99 if freq < 1e4 else titr * E_max_GM100k_99
                print(f'thresh = {thresh_freqs[i_f]}\n\n')
            results['hotspot_100k'][dur_fact] = {'location': location, 'normal': normal, 'freq': freq_array, 'titr': titr_freqs, 'thresh': thresh_freqs}
    else:
        results['hotspot_100k'] = copy.copy(results['hotspot_300'])
    with open(f"results/FEM/{name}.pkl",'wb') as pf:
        pickle.dump(results,pf)

    time_end = datetime.datetime.now()
    time_diff = time_end-time_start
    print('total simulation time:  ',time_diff)
