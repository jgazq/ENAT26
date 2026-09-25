# Run from anaconda python-prompt with exec(open("Interp3Dfield.py").read())

from neuron import h, gui
import numpy as np
import math
import sys, os
from scipy.interpolate import griddata, interpn
import matplotlib.pyplot as plt

# # Settings
# nsegMul = 1
# InterpolMethod = 'linear'
# #PotentialDistrFilePath = "PotentialDistr/PotentialDistr-6mm.txt"
# #PotentialDistrFilePath = "PotentialDistr/3. Hexagonal array/HexEl_optim_HCP-600um-20umMESH.txt"
dtMAX = 0.025	# Maximal discretization step  [ms]
TpRefdt = 10    # Refinement factor for small TP
Tmin = 5		# Minimal total simulation duration
TpRefT = 2		# Refinement factor for long TP

# h.load_file("init.hoc")
h.load_file("thresh4.hoc")  
# #PotArray = np.genfromtxt(PotentialDistrFilePath,comments='%')   # V [V], x,y,z [µm] 

# Increase nseg
# if (nsegMul!=1):
# 	print("changed segments")
# 	for sec in h.allsec():
# 		if h.ismembrane("xtra"):
# 			sec.nseg = nsegMul*sec.nseg
# 	h.setpointers()
 
# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def calcESext(xT,yT,zT,phi,theta,psi,InterpolMethod,PotArray):
	""" (xT,yT,zT) is translation vector, (phi,theta,psi) are Tait-Bryan extrinsic roll-pitch-yaw angles """
	Rtb = TaitBryan2rotMat([phi,theta,psi])
	xSOM,ySOM,zSOM = findSOMA()
	i=0
	for sec in h.allsec():
		if h.ismembrane("xtra"):
			for seg in sec:
				if "Scale" not in str(seg) and "Elec" not in str(seg): ##
					i=i+1
					print(i) if (i%100 == 0) else None ##
					[xD,yD,zD] = np.dot(Rtb,[(seg.x_xtra-xSOM),(seg.y_xtra-ySOM),(seg.z_xtra-zSOM)])+[xSOM,ySOM,zSOM]+[xT,yT,zT]
					seg.es_xtra = griddata(PotArray[:,0:3],PotArray[:,3],\
					[xD,yD,zD],method=InterpolMethod,fill_value=0) # [mV]

def calcESext2(xT,yT,zT,phi,theta,psi,InterpolMethod,PotArray,x,y,z): ##
	""" (xT,yT,zT) is translation vector, (phi,theta,psi) are Tait-Bryan extrinsic roll-pitch-yaw angles """
	Rtb = TaitBryan2rotMat([phi,theta,psi])
	xSOM,ySOM,zSOM = findSOMA()
	i=0
	for sec in h.allsec():
		if h.ismembrane("xtra"):
			for seg in sec:
				if "Scale" not in str(seg) and "Elec" not in str(seg): ##
					i=i+1
					print(i) if (i%50 == 0) else None ##
					[xD,yD,zD] = np.dot(Rtb,[(seg.x_xtra-xSOM),(seg.y_xtra-ySOM),(seg.z_xtra-zSOM)])+[xSOM,ySOM,zSOM]+[xT,yT,zT]
					seg.es_xtra = interpn((x,y,z),PotArray,\
					[xD,yD,zD],method=InterpolMethod,fill_value=0,bounds_error=False) # [mV]


def calcESext3(xT,yT,zT,new_normal,prev_normal,InterpolMethod,Arrayx,Arrayy,Arrayz,x,y,z): ##
    """ (xT,yT,zT) is translation vector, (phi,theta,psi) are Tait-Bryan extrinsic roll-pitch-yaw angles """
    xSOM,ySOM,zSOM = findSOMA()
    i=0
    for sec in h.allsec():
        if h.ismembrane("xtra"):
            for seg in sec:
                if "Scale" not in str(seg) and "Elec" not in str(seg) and "ExcNode" not in str(seg): ##
                    i=i+1
                    #print(i) if (i%20 == 0) else None ##
                    #first change the NEURON axes to the python axes to align the coordinate systems
                    xp, yp, zp = seg.x_xtra,seg.y_xtra,seg.z_xtra
                    x90, y90,z90 = xp, -zp, yp
                    #then rotate the neuron (if necessary, otherwise [0,0,1])
                    [xD,yD,zD] = align_to_normal(np.array([x90,y90,z90]),new_normal,prev_normal)+[xT,yT,zT]
                    seg.Ex_xtra = interpn((x,y,z),Arrayx,\
                    [xD,yD,zD],method=InterpolMethod,fill_value=0,bounds_error=False)[0] # [V/m]
                    seg.Ey_xtra = interpn((x,y,z),Arrayy,\
                    [xD,yD,zD],method=InterpolMethod,fill_value=0,bounds_error=False)[0] # [V/m]
                    seg.Ez_xtra = interpn((x,y,z),Arrayz,\
                    [xD,yD,zD],method=InterpolMethod,fill_value=0,bounds_error=False)[0] # [V/m]

def calcESext4(xT,yT,zT,new_normal,prev_normal,InterpolMethod,Arrayx,Arrayy,Arrayz,x,y,z): ##
    """ (xT,yT,zT) is translation vector, (phi,theta,psi) are Tait-Bryan extrinsic roll-pitch-yaw angles """
    xSOM,ySOM,zSOM = findSOMA()
    i=0
    for sec in h.allsec():
        if h.ismembrane("xtra"):
            for seg in sec:
                if "Scale" not in str(seg) and "Elec" not in str(seg) and "ExcNode" not in str(seg): ##
                    i=i+1
                    #print(i) if (i%20 == 0) else None ##
                    #first change the NEURON axes to the python axes to align the coordinate systems
                    xp, yp, zp = seg.x_xtra,seg.y_xtra,seg.z_xtra
                    x90, y90,z90 = xp, -zp, yp
                    #then rotate the neuron (if necessary, otherwise [0,0,1])
                    [xD,yD,zD] = align_to_normal(np.array([x90,y90,z90]),new_normal,prev_normal)+[xT,yT,zT]
                    if 'soma' in str(seg):
                        return (xD,yD,zD) 

def findSOMA():
	for sec in h.allsec():
		if h.ismembrane("xtra"):
			if (sec.type_xtra==1):
				return [sec.x_xtra, sec.y_xtra, sec.z_xtra]
				
def detRADIUS():
	[xSOM,ySOM,zSOM] = findSOMA()
	radius = []
	for sec in h.allsec():
		if h.ismembrane("xtra"):
			for seg in sec:
				radius.append(math.sqrt((seg.x_xtra-xSOM)**2+(seg.y_xtra-ySOM)**2+(seg.z_xtra-zSOM)**2))	
	return radius
	
def calcThresh(xSOM,ySOM,zSOM,phi,theta,psi,DEL,DUR,Ancat):
	h.DEL = DEL
	h.DUR = DUR
	#h.dt = min(DUR/TpRefdt,dtMAX) ## I change this in the thresh_(log)freq function instead
	h.steps_per_ms = 1/h.dt				# Circumvent setdt()-function in stdinit
	#h.tstop = max(TpRefT*DUR,Tmin) ## I change this in the thresh_(log)freq function instead
	if (h.name_declared("apc")==0):
		h("objref apc")
		h("apc = new APCount(0.5)")
	h("stimamp = 1")
	h("ANCAT = 1")
	h.ANCAT = (-1)**Ancat 
	#print(h.ANCAT)
	#print(h.apc.thresh)
	h("func thresh_excited() {setstim(DEL,DUR,ANCAT*stimamp) AMP = ANCAT*stimamp run() return apc.n > 0}")
	[xSOM0,ySOM0,zSOM0] = findSOMA()
	#calcESext(xSOM-xSOM0,ySOM-ySOM0,zSOM-zSOM0,phi,theta,psi)
	threshold = h.threshold(h._ref_stimamp)
	print("interp3D",h.dt,h.tstop)
	return h.ANCAT*threshold
	
def calcThreshSimpl(xSOM,ySOM,zSOM,DEL,DUR,Ancat):
	return calcThresh(xSOM,ySOM,zSOM,0,0,0,DEL,DUR,Ancat)
	
def SweepPD(xSOM,ySOM,zSOM,DEL,PD0,PD1,logNum,Ancat):
	Duration = np.logspace(math.log10(PD0),math.log10(PD1),num=logNum,endpoint=True,base=10.0)
	Strength = []
	for PDur in Duration:
		Strength.append(calcThresh(xSOM,ySOM,zSOM,0,0,0,DEL,PDur,Ancat))
	return (Duration,Strength)
	
def Sweeper(xSOM0,xSOM1,nx,xlinlog,ySOM0,ySOM1,ny,ylinlog,zSOM0,zSOM1,nz,zlinlog,\
	PD0,PD1,ndur,durlinlog,DEL0,DEL1,ndel,dellinlog,phi0,phi1,nphi,philinlog,theta0,\
	theta1,ntheta,thetalinlog,psi0,psi1,npsi,psilinlog,Ancat,TF):
	""" Sweeper in soma coordinates, pulse duration, delay, and Tait-Bryan angles
	Returns threshold-matrix, ExcSecref-matrix, ExcSecname-matrix, ExcLatency-matrix
	Excitation nodes calculated with TF*threshold (TF = threshold factor)"""
	if (xlinlog==0):
		xRange = np.linspace(xSOM0,xSOM1,num=nx,endpoint=True,retstep=False)
	else:
		xRange = np.logspace(math.log10(xSOM0),math.log10(xSOM1),num=nx,endpoint=True,base=10.0)
	if (ylinlog==0):
		yRange = np.linspace(ySOM0,ySOM1,num=ny,endpoint=True,retstep=False)
	else:
		yRange = np.logspace(math.log10(ySOM0),math.log10(ySOM1),num=ny,endpoint=True,base=10.0)
	if (zlinlog==0):
		zRange = np.linspace(zSOM0,zSOM1,num=nz,endpoint=True,retstep=False)
	else:
		zRange = np.logspace(math.log10(zSOM0),math.log10(zSOM1),num=nz,endpoint=True,base=10.0)
	if (durlinlog==0):
		PDRange = np.linspace(PD0,PD1,num=ndur,endpoint=True,retstep=False)
	else:
		PDRange = np.logspace(math.log10(PD0),math.log10(PD1),num=ndur,endpoint=True,base=10.0)
	if (dellinlog==0):
		DELRange = np.linspace(DEL0,DEL1,num=ndel,endpoint=True,retstep=False)
	else:
		DELRange = np.logspace(math.log10(DEL0),math.log10(DEL1),num=ndel,endpoint=True,base=10.0)
	if (philinlog==0):
		PHIRange = np.linspace(phi0,phi1,num=nphi,endpoint=True,retstep=False)
	else: 
		PHIRange = np.logspace(math.log10(phi0),math.log10(phi1),num=nphi,endpoint=True,base=10.0)
	if (thetalinlog==0):
		THETARange = np.linspace(theta0,theta1,num=ntheta,endpoint=True,retstep=False)
	else: 
		THETARange = np.logspace(math.log10(theta0),math.log10(theta1),num=ntheta,endpoint=True,base=10.0)	
	if (psilinlog==0):
		PSIRange = np.linspace(psi0,psi1,num=npsi,endpoint=True,retstep=False)
	else: 
		PSIRange = np.logspace(math.log10(psi0),math.log10(psi1),num=npsi,endpoint=True,base=10.0)
	
	MonApc, RecordAPt = MonitorAll()
	
	Itr = np.zeros([nx,ny,nz,ndur,ndel,nphi,ntheta,npsi]) # Threshold current
	ExcSecname = np.empty([nx,ny,nz,ndur,ndel,nphi,ntheta,npsi],dtype=object) # Excitation section name
	ExcSecref = np.empty([nx,ny,nz,ndur,ndel,nphi,ntheta,npsi],dtype=object) # Excitation section ref
	ExcLatency = np.zeros([nx,ny,nz,ndur,ndel,nphi,ntheta,npsi]) # Latency for excitation at I=TF*Itr
	
	for xI, x in enumerate(xRange):
		for yI, y in enumerate(yRange):
			for zI, z in enumerate(zRange):
				for PDI, PD in enumerate(PDRange):
					for DELI, DEL in enumerate(DELRange):
						for PHII, PHI in enumerate(PHIRange):
							for THETAI, THETA in enumerate(THETARange):
								for PSII, PSI in enumerate(PSIRange):
									Itr[xI,yI,zI,PDI,DELI,PHII,THETAI,PSII] = calcThresh(x,y,z,DEL,PD,Ancat,PHI,THETA,PSI)
									h.setstim(DEL,PD,TF*Itr[xI,yI,zI,PDI,DELI,PHII,THETAI,PSII])
									h.AMP = TF*Itr[xI,yI,zI,PDI,DELI,PHII,THETAI,PSII]
									h.run()
									ExcSecref[xI,yI,zI,PDI,DELI,PHII,THETAI,PSII], ExcSecname[xI,yI,zI,PDI,DELI,PHII,THETAI,PSII], \
									ExcLatency[xI,yI,zI,PDI,DELI,PHII,THETAI,PSII] = FindExcNode(MonApc,RecordAPt)
	
	return (xRange,yRange,zRange,PDRange,DELRange,PHIRange,THETARange, PSIRange, Itr,ExcSecref,ExcSecname,ExcLatency)
	
def calcDistExcProbe(xRange,yRange,zRange,phi,theta,psi,ExcSecref):
	Rtb = TaitBryan2rotMat([phi,theta,psi])
	xSOM,ySOM,zSOM = findSOMA()
	DistExcProbe = np.zeros([np.size(xRange),np.size(yRange),np.size(zRange)])
	for xI,x in enumerate(xRange):
		for yI,y in enumerate(yRange):
			for zI,z in enumerate(zRange):
				xSEC = ExcSecref[xI,yI,zI].sec.x_xtra
				ySEC = ExcSecref[xI,yI,zI].sec.y_xtra
				zSEC = ExcSecref[xI,yI,zI].sec.z_xtra
				xSEC,ySEC,zSEC = np.dot(Rtb,[(seg.x_xtra-xSOM),(seg.y_xtra-ySOM),(seg.z_xtra-zSOM)])+[xSOM,ySOM,zSOM]
				DistExcProbe[xI,yI,zI] = math.sqrt((xSEC+x-xSOM)**2+(ySEC+y-ySOM)**2+(zSEC+z-zSOM)**2)
	
	return DistExcProbe
	
def PlotSD(StrDur):
	""" Plot SD from StrDur-tuple """
	if StrDur[1][0]<0:
		StrDur[1][:] = [(-1)*s for s in StrDur[1][:]]
	plt.loglog(StrDur[0],StrDur[1])
	plt.xlabel("Pulse duration [ms]")
	plt.ylabel("Strength [V]")
	plt.title("Strength-duration curve")
	plt.show()

def SweepSPAT(xSOM0,xSOM1,nx,ySOM0,ySOM1,ny,zSOM0,zSOM1,nz,PD0,PD1,DEL,Ancat):
	""" Sweep over the space in argument for two pulse durations (PD0 very small, PD1 very large) 
	and determine rheobasic current and SD-time constant """
	xRange = np.linspace(xSOM0,xSOM1,num=nx,endpoint=True,retstep=False)
	yRange = np.linspace(ySOM0,ySOM1,num=ny,endpoint=True,retstep=False)
	zRange = np.linspace(zSOM0,zSOM1,num=nz,endpoint=True,retstep=False)
	I0 = np.zeros([nx,ny,nz])
	I1 = np.zeros([nx,ny,nz])
	for xI, x in enumerate(xRange):
		for yI, y in enumerate(yRange):
			for zI, z in enumerate(zRange):
				I0[xI,yI,zI] = calcThresh(x,y,z,0,0,0,DEL,PD0,Ancat)
				I1[xI,yI,zI] = calcThresh(x,y,z,0,0,0,DEL,PD1,Ancat)
	return (xRange,yRange,zRange,I0,I1)
	
def MonitorAll():
	""" Set up APCount-Point_processes to monitor all sections for excitation """
	MonApc = {}
	RecordAPt = {}
	for sec in h.allsec():
		if (h.ismembrane("xtra")):
			MonApc[h.secname()] = h.APCountInterp(0.5)
			RecordAPt[h.secname()] = h.Vector()
			MonApc[h.secname()].record(RecordAPt[h.secname()])
	return MonApc, RecordAPt
	
def ExcSecrefsDict():
	""" Determine ExcSecrefs dict """
	ExcSecrefsDict = {}
	for sec in h.allsec():
		if (h.ismembrane("xtra")):
			ExcSecrefsDict[h.secname()] = h.SectionRef()
	return ExcSecrefsDict
			
def FindExcNode(MonApc,RecordAPt):
	""" Finds the excitation node """
	t=np.inf
	secRefExc = ''
	secName = ''
	for sec in h.allsec():
		if (h.ismembrane("xtra")):
			nAP = MonApc[h.secname()].n
			if (nAP>0):
				tAP = RecordAPt[h.secname()].x[0]	
				if (tAP<t):
					t=min(t,tAP)
					secRefExc = h.SectionRef()
					secName = h.secname()
	return secRefExc, secName, t
	
def plotExcNode(ExcSecRef,plot_mode):
    xExc,yExc,zExc = ExcSecRef.sec.x_xtra, ExcSecRef.sec.y_xtra, ExcSecRef.sec.z_xtra
    h("create ExcNode")							# Not ExcNode = h.Section(), because this PyHOC-object is less stable after function exit
    h.ExcNode.pt3dclear()
    h.ExcNode.pt3dadd(xExc-5,yExc,zExc,1)
    h.ExcNode.pt3dadd(xExc+5,yExc,zExc,1)
    color_plotmaxPy(plot_mode)
    h("objref ExcNodeMark")
    h("ExcNode ExcNodeMark = new PointProcessMark(0.5)")
    h.shplot.point_mark(h.ExcNodeMark,2,"O",10)
    h.shplot.label(600,50,"Excitation node",1,1,0,0,2)	
    color_plotmaxPy(plot_mode)

def color_plotmaxPy(plot_mode):
        h.shplot.color_list(h.cell.axonal,1)
        if (plot_mode == 1):
                if (h.myelinate_ax):
                     h.shplot.color_list(h.iseg_secList,5)
                     h.shplot.color_list(h.Node_secList,2)
                     h.shplot.color_list(h.Myelin_secList,1)
                     h.shplot.color_list(h.Unmyelin_secList,5)
        elif (plot_mode == 2):
                h.shplot.color_list(h.main_ax_list,2)	
        h.shplot.color_list(h.cell.apical,3)
        h.shplot.color_list(h.cell.basal,4)
        h.shplot.exec_menu("View = plot")
		

def TaitBryan2rotMat(theta) :
	"""  Calculates Rotation Matrix given Tait-Bryan angles.
	theta = [phi,theta,psi] Tait-Bryan angles """      
	R_x = np.array([[1,         0,                  0                   ],
					[0,         math.cos(theta[0]), -math.sin(theta[0]) ],
					[0,         math.sin(theta[0]), math.cos(theta[0])  ]
					])         
	R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
					[0,                     1,      0                   ],
					[-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
					])
                 
	R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
					[math.sin(theta[2]),    math.cos(theta[2]),     0],
					[0,                     0,                      1]
					])                  
	R = np.dot(R_z, np.dot( R_y, R_x ))
	return R