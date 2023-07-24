# Simulateur de télescope adapté pour CAGIRE
Le répertoire simu comporte tous les codes et cartes nécessaires à la simulation d'une rampe d'images acquises par la caméra CAGIRE. Le simulateur à été adapté à partir de celui dévellopé par David Corre, qui se trouve sur le lien suivant https://github.com/dcorre/ImSimpy/tree/master.   
Le simulateur d'image prend également en compte les données de l'ETC dévellopé par David Corre, sur ce lien : https://github.com/dcorre/pyETC

## Création d'un rampe 
Pour lancer la création d'une rampe simulé, il faut lancer le "main" simu_b.py et régler différents paramètres présentés ci-dessous : 

```
seeing=1
moonage=7
band='J'                       # Bande photométrique de la simulation ('J' ou 'H' )
t_rampe =  60                  # Durée de la rampe de secondes
RA_image = 214.477             # Coordonnées de l'observation RA : 214.477 ou 148.7834 ou 159.3031719 par ex. Ne sera pas utilisé si AddGRB = True
DEC_image = -45.411            # Coordonnées de l'observation DEC :-45.411 ou 18.167 ou 33.71224656 par ex. Ne sera pas utilisé si AddGRB = True
nomrampe=  'GRB90423_60s'      # Nom à donner à la rampe simulée

AddGRB = False                 # si l'on ajoute un GRB simulé (True) ou non (False)
tdeb_obs = 300                 # Si GRB : temps du début de l'observationaprès détection du sursaut
cheminGRB ='GRB90423_J.txt'    # chemin du GRB simulé dans la base de donnée (coordonnée et magnitude en fonction du temps)

nomPersistance =  'carte_persistance.fits'              # nom de la carte de persistance de l'acquisition précédente si persistance
Treset = 0                      # délai entre l'acquisition simulée et la précédente
```
Dans ce cas, on lance la simulation d'une rampe de 60 s en bande J, observant la zone du ciel centrée autour de `RA_image ; DEC_image`, sans simulation de sursaut, mais en ajoutant la persistance de la carte `carte_persistance.fits` créee lors d'ue acquisition précédente.    
Si l'on écrit `AddGRB = True`, on obtient une rampe de 60 s en bande J, centrée autour d'une position aléatoire autour du surssaut simulée (position du sursaut +/- 0.16 deg).   

Il suffit ensuite de lancer ce code. 


## Paramétrage des effets à simuler
Pour paraméter les effets à simuler, les modifications se font dans le fichier ImSimpyA/data/CAGIRE_saturation_test2.hjson.    
Chaque effet peut être ajouté en lui donnant la valeur `True`.   
Le choix des effets que l'on peut simuler ou modifier pour la simulation de CAGIRE se fait uniquement par ce fichier .hjson n  dans la section IMAGE SIMULATOR : 

```
########### IMAGE SIMULATOR #####################

...

#----- Compulsory ones ------
#add sources
"addSources" : yes

#add grb
"addGRB" : yes

#add noise?
"shotNoise" : yes

#add cosmetics?
"cosmetics" : yes
# Path starts at ImSimpy/data/Cosmetics/
"DeadPixFile" : CAGIRE_saturation/cold.fits

#add cosmic rays?
"cosmicRays" : yes
"CosmicsFile" : Cosmics.fits

# ---- Optional ones -----
#add sky background?
"background" : yes

#add readout time?
"readoutNoise":yes

#add dark current?
"darkCurrent" : yes
"DarkFile" : DC.fits

#cut values over saturation level?
"saturation" : yes
"SaturationFile":saturation_1V.fits

#Add CrossTalk?
"CrossTalk" : no
"CrossTalkFile": Crosstalk/crosstalk.fits

#convert to ADU
"ADU": yes
"GainMapFile": CAGIRE_saturation_test/Gain_nir.fits

#Non linearity ?
"nonlinearity": yes
"NonLinearityFile": gamma_1V_electrons.fits


#FlatField effect
"FlatField": yes
"FlatFieldFile":FlatField/flatfield_CEA_borne.fits

#Créer une carte de persistance pour l'aqcuisition suivante
"CreatePersistance":no

#Appliquer une carte de persistance de l'aqcuisition précédente
"Persistance":yes
"PersistanceAmp":Persistance/amp_2FW.fits
"PersistanceTau": Persistance/tau_2FW.fits
"PersistanceConv": Persistance/conv_2FW.fits
"Treset":0 #temps entre les deux acquisitions

#Offset
"Offset": yes
# Path starts at ImSimpy/data/Offset/
"OffsetFile":offset_1V.fits
#offset_det4.fits
#CAGIRE_saturation/Offset_vis.fits

#pixels de référence 
"PixRefFile":data/ref_test.fits

#pixels actifs
"PixActifFile":data/PixViolet.fits

```
Ce fichier permet ainsi de sélectionner les effets à simuler grâce au code ImSimpyA.py
