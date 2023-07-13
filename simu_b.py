# -*- coding: utf-8 -*-
"""
Created on june 2023

@author: Alix
"""
import time
import numpy as np
import imp
import warnings
import os
warnings.filterwarnings('ignore')
import math

_, path, _ = imp.find_module('ImSimpyA')
print(path)
start = time.perf_counter()

#'uniforme_flux2255'#'test_new_boucle_simu_2' #'test_new_boucle_simu_fonctions'#'test_differentiel_d'#'test_new_boucle_simu_snl' #'test_new_boucle_bruit'
#'GRB_90423_persistance_1V_persistance.fits'
########################################################################################################################
#paramètres à préciser
########################################################################################################################
seeing=1
moonage=7
band='J'                       # Bande photométrique de la simulation ('J' ou 'H' )
t_rampe =  20#60               # Durée de la rampe de secondes
RA_image = 214.477             # Coordonnées de l'observation RA : 214.477 ou 148.7834 ou 159.3031719 par ex. Ne sera pas utilisé si AddGRB = True
DEC_image = -45.411            # Coordonnées de l'observation DEC :-45.411 ou 18.167 ou 33.71224656 par ex. Ne sera pas utilisé si AddGRB = True
nomrampe=  'test_propre'       # Nom à donner à la rampe simulée

AddGRB =True                   # si l'on ajoute un GRB simulé (True) ou non (False)
tdeb_obs = 300                 # Si GRB : temps du début de l'observationaprès détection du sursaut
cheminGRB ='/home/alix/anaconda3/dcorre-ImSimpy-42ac6cb/ImSimpy/GRB90423_J.txt'  # chemin du GRB simulé dans la base de donnée (coordonnée et magnitude en fonction du temps)

nomPersistance =  'carte_persistance.fits'              # nom du fichier des pixels saturés de l'acquition précédente si persistance
Treset = 0                     #délai entre l'acquisition et la précédente (depuis le premier reset)
########################################################################################################################
########################################################################################################################

#Calcul du nombre de frames de la rampe
Texp=1.33                       # Durée d'une frame
Nfin=np.round(t_rampe/1.33)
Nframes=np.arange(0,Nfin,1)     # Liste des frames


#Si l'on simule un GRB :

if AddGRB==True :
    # temps auquel on commence à observer
    tdeb_obs=tdeb_obs- 1.33

    #On prend la base de donnée du GRB que l'on souhaite simuler, magnitude en fonction du temps et coordonnées
    GRB= np.loadtxt(cheminGRB)
    tGRB = GRB[0]
    idebGRB = np.intersect1d(np.argwhere(tGRB>= tdeb_obs),np.argwhere(tGRB< tdeb_obs+1.33 ))
    Nobs = [int(idebGRB+ x) for x in Nframes]          #liste des indice des magnitudes correspondant aux temps que l'on veut simuler
    magGRB=GRB[1]
    magGRB=magGRB[Nobs]                                #calcul de la magnitude associée à chaque temps, à partir du temps de début (par une loin de puissance)
    raGRB = GRB[2,0]
    decGRB = GRB[3,0]

    #on centre l'image sur le GRB +/- un décallage entre 0 et 0.16 deg.
    DRA = np.random.normal(loc=0.0, scale = 0.16, size = None)
    DDEC = np.random.normal(loc=0.0, scale = 0.16, size = None)
    RA_image= 148.7834#+0.0435]#  +0.16]#[raGRB+DRA] #[148.7834] #[159.3031719] #[266.4083]#,[159.3031719]#RA_image=[260,30]
    DEC_image= 18.167#+0.1205]# -0.05] #+0.06  #[decGRB+DDEC] #[18.167]  #[33.71224656] #[-29.0064]#,[33.71224656] #DEC_image=[-30,30]
    print('localisation centre image',RA_image,DEC_image)

#Configuration : fihhiers de sortie et choix du fichier de configuration
output_dir = 'CAGIRE_saturation_test2'
configFile = 'CAGIRE_saturation_test2.hjson'

#######################################################################################################################
#créer une image
#######################################################################################################################

from ImSimpyA import ImageSimulator_UTR
colibri_IS = ImageSimulator_UTR(configFile=configFile, name_telescope='colibri')

# Read the configfile
colibri_IS.readConfigs()

colibri_IS.config['seeing_zenith'] = seeing
colibri_IS.config['moon_age'] = moonage

colibri_IS.config['SourcesList']['generate']['radius'] = 21.7
colibri_IS.config['SourcesList']['generate']['catalog'] = 'II/246'

# Load existing PSF instead of computing new ones for speeding up the calculations
colibri_IS.config['PSF']['total']['method'] = 'load'

colibri_IS.config['nom'] = nomrampe
colibri_IS.config['nomPersistance'] = nomPersistance
colibri_IS.config['Treset'] = Treset

if AddGRB == True:
    colibri_IS.config["addGRB"] = 'yes'
    colibri_IS.config['SourcesToAdd']['gen']['listeMAG'] = magGRB
    print('mag grb added', colibri_IS.config['SourcesToAdd']['gen']['listeMAG'])
    colibri_IS.config['SourcesToAdd']['gen']['RA'] = raGRB
    colibri_IS.config['SourcesToAdd']['gen']['DEC'] = decGRB

else :
    colibri_IS.config["addGRB"] = 'no'

ra = RA_image
dec = DEC_image
colibri_IS.config['SourcesList']['generate']['RA'] = ra
colibri_IS.config['SourcesList']['generate']['DEC'] = dec
colibri_IS.config['RA'] = ra
colibri_IS.config['DEC'] = dec

colibri_IS.config['filter_band'] = band
colibri_IS.config['PSF']['total']['file'] = 'total_PSF/%s/PSF_total_%s.fits' % (output_dir, band)
colibri_IS.config['exptime'] = Texp
colibri_IS.config['Nfin'] = Nfin


colibri_IS.config['output'] = 'CAGIRE_saturation2/'+ nomrampe + '.fits'

# Run the Image Simulator
colibri_IS.simulate('data')

t1=time.perf_counter()
print('temps totale',t1-start)


'''
#############################################################################################
# Si n'existent pas : génerer une carte de cross-talk, une carte de gain, une carte de cosmics

#générer cross talk matrice
ct=np.array([[0.0008, 0.008, 0.0008], [0.008, 1-(0.008*4+0.0008*4), 0.008], [0.0008, 0.008, 0.0008]])
SaveFitComplet(ct, 'crosstalk', 'Cross talk', str(0.2)+'pct', 3, path=path + '/data/Crosstalk/')

#carte de gain
from ImSimpy.utils.generateCalib import Offset, Vignetting, GainMap, GainMap_Alix
#GainMap(filename=path + '/data/GainMap/%s/Gain_nir.fits' % output_dir,Type='random', mean=7.5, std=0.005, Nampl=32, xsize=2048, ysize=2048)
GainMap_Alix(filename=path + '/data/GainMap/%s/Gain_nir.fits' % output_dir,Type='random', mean=10, std=0.05, xsize=2048, ysize=2048)

# générer une carte de cosmics aleatoire
actifs = np.loadtxt('/home/alix/anaconda3/dcorre-ImSimpy-42ac6cb/ImSimpy/data/PixViolet.txt').astype(int)
nom = 'Cosmics_b'                        # nom du fichier de cosmic à générer
nbCos = 150                              # nombre de cosmics à simuler 
xsize = 2048                             # nombre pixels detecteur : x-axis 
ysize = 2048                             # nombre pixels detecteur : x-axis 

np.random.seed()
pos = np.random.randint(low=0, high=xsize * ysize, size=nbCos)
pos = np.intersect1d(pos, actifs)
nbCos = len(pos)
np.random.seed()
energy = np.random.uniform(low=300, high=12000, size=nbCos)  # uniform ?

np.random.seed()
temps = np.random.randint(low=0, high=nbrampe, size=nbCos)

tabCos = np.zeros([3, nbCos])
tabCos[0] = pos
tabCos[1] = energy
tabCos[2] = temps

_, path, _ = imp.find_module('ImSimpy')
SaveFitComplet(tabCos, nom, 'nbframes', str(len(Nframes)-1), 3, path=path + '/data/Cosmics/')

'''





