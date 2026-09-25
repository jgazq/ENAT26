: $Id: xtra.mod,v 1.4 2014/08/18 23:15:25 ted Exp ted $
: 2018/05/20 Modified by Aman Aberra 

NEURON {
	SUFFIX xtra
	RANGE es : (es = max amplitude of the potential)	
	RANGE Ex,Ey,Ez :TMS
	RANGE Dx,Dy,Dz	:TMS
	RANGE x, y, z, type, order
	GLOBAL stim : (stim = normalized waveform)
	POINTER ex 
	GLOBAL sine_freq : added
}

UNITS {	
	PI = (pi) (1) : added
}

PARAMETER {	
	es = 0 (mV)
	x = 0 (1) : spatial coords
	y = 0 (1)
	z = 0 (1)
	Ex = 0 (1) : E-field components :TMS
	Ey = 0 (1) :TMS
	Ez = 0 (1) :TMS
	Dx = 0 (1) : unit vector of orientation of compartment (central-diff) :TMS
	Dy = 0 (1) :TMS
	Dz = 0 (1) :TMS		
	type = 0 (1) : numbering system for morphological category of section - unassigned is 0
	order = 0 (1) : order of branch/collateral. 
	sine_freq = 1.7 (kHz) : added, this frequency needs to be changed
	alpha = 1.3900012397674093 (1/ms)
	A_mV = 97.3278421120322 (mV)
}

ASSIGNED {
	v (millivolts)
	ex (millivolts)
	stim (unitless) 		
	area (micron2)
}

INITIAL {
	:ex = stim*es : original
	ex = stim * A_mV * exp(-alpha*(t-1)) * sin(2*PI*sine_freq*(t-1) + PI)*es : added :1ms delay and first wave negative
}


BEFORE BREAKPOINT { : before each cy' = f(y,t) setup
  :ex = stim*es : original
  ex = stim * A_mV * exp(-alpha*(t-1)) * sin(2*PI*sine_freq*(t-1) + PI)*es : added :1ms delay and first wave negative
}

