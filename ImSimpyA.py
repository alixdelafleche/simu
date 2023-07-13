# -*- coding: utf-8 -*-

"""Main module."""
import matplotlib.pyplot as plt

parameters = {'axes.labelsize': 18, 'axes.titlesize': 20, 'xtick.labelsize': 18, 'ytick.labelsize': 18}
plt.rcParams.update(parameters)
"""
Optical/NIR Telescope Image Simulator
=============================================

This file contains an image simulatorfor optical/NIR telescope

Contact Information
-------------------
mail: david.corre@lam.fr

"""
import os, sys, datetime, math, shutil
from collections import OrderedDict
import hjson
import scipy
from scipy.ndimage import interpolation, zoom
from scipy import ndimage
from astropy.io import fits
import numpy as np
import numexpr as ne
from utils import PSFUtils
from scipy.signal import fftconvolve
import imp
import numpy as np
from scipy import signal
import pyETC.utils

convolution = fftconvolve


class ImageSimulator_UTR():
    """
    Image Simulator for optical/NIR telescope

    """

    def __init__(self, configFile='default_input.hjson', name_telescope='default', seed=None, debug=False,
                 random=False):
        """
        Class Constructor.

        :configFile : name of the cofig file
        :seed: value used to set the random number generator (default: None --> use current time).
        """
        try:
            _, path, _ = imp.find_module('ImSimpyA')
            path = path[0:34]+'/'

        except:
            print('path to pyETC can not be found.')

        self.path = path
        self.configfile_nopath = configFile
        self.configfile = self.path +  configFile
        self.name_telescope = name_telescope



        if seed != None:
            seed = int(seed)
            np.random.seed(seed=seed)  # fix the seed
            print('Setting the random number generator seed to {}'.format(seed))
        else:
            np.random.seed()
            print('Setting the random number generator seed: current time')
        self.debug = bool(debug)
        self.random = bool(random)

        # load instrument model, these values are also stored in the FITS header
        self.information = OrderedDict()

        # update settings with defaults. It will be used if they are missing if the config file
        self.information.update(dict(psfoversampling=1.0,
                                     xsize=2048,
                                     ysize=2048,
                                     FWC=125000,
                                     DC=0.001,
                                     RN=54.0,
                                     bias=500.0,
                                     gain=10.0,
                                     zeropoint=23.5,
                                     Nexp=1,
                                     exptime=565.0,
                                     readouttime=5.,
                                     sky_brightness=19.3,
                                     RA=123.0,
                                     DEC=45.0,
                                     mode='same'))

    def readConfigs(self):
        """
        Reads the config file information using configParser.
        """
        # Load the input file in hjson format into a dictionary
        with open(self.configfile, encoding='utf-8') as f:
            self.config = hjson.load(f)

    def etc(self, config_type):
        """ Execute the Exposure Time Calculator to get some information (zeropoint, grb mag,...) """

        try:
            from pyETC.pyETC import etc
            try:
                _, path_etc, _ = imp.find_module('pyETC/pyETC')
                path_etc = path_etc[0:32]
                print(path_etc)
            except:
                print('path to pyETC can not be found.')
        except ValueError:
            print('Package ETC not found, you have to install it')

        # copy config file to ETC config folder
        shutil.copy(self.configfile, path_etc  + self.configfile_nopath)

        if config_type == 'file':
            etc_info = etc(configFile=self.configfile_nopath, config_type=config_type, name_telescope=self.name_telescope)
        elif config_type == 'data':
            etc_info = etc(configFile=self.config, config_type=config_type, name_telescope=self.name_telescope)
        etc_info.sim()
        # Update the config file
        self.config['camera_type'] = etc_info.information['cameras'][etc_info.information['channel']]['camera_type']
        self.config['sensor'] = etc_info.information['cameras'][etc_info.information['channel']]['sensor']
        self.config['RN'] = etc_info.information['cameras'][etc_info.information['channel']]['RN']
        self.config['DC'] = etc_info.information['cameras'][etc_info.information['channel']]['DC']
        self.config['FWC'] = etc_info.information['cameras'][etc_info.information['channel']]['FWC']
        self.config['gain'] = etc_info.information['cameras'][etc_info.information['channel']]['gain']
        self.config['bits'] = etc_info.information['cameras'][etc_info.information['channel']]['bits']
        self.config['xsize'] = etc_info.information['cameras'][etc_info.information['channel']]['Nphotocell_X']
        self.config['ysize'] = etc_info.information['cameras'][etc_info.information['channel']]['Nphotocell_Y']
        self.config['xPixSize'] = etc_info.information['cameras'][etc_info.information['channel']]['Photocell_SizeX'] * \
                                  etc_info.information['binning_X']
        self.config['yPixSize'] = etc_info.information['cameras'][etc_info.information['channel']]['Photocell_SizeY'] * \
                                  etc_info.information['binning_Y']
        self.config['dig_noise'] = etc_info.information['dig_noise']
        self.config['D_M1'] = etc_info.information['D_M1']
        self.config['D_M2'] = etc_info.information['D_M2']
        self.config['M2_factor'] = etc_info.information['M2_factor']
        self.config['FoV_1axis'] = etc_info.information['FoV_axis1']
        self.config['Focal_length'] = etc_info.information['foc_len']
        self.config['filter_folder'] = etc_info.information['filter_folder']
        self.config['Sky_CountRate'] = etc_info.information['Sky_CountRate']
        # self.config['sky_brightness']=etc_info.information['sky_brightness']
        self.config['SB_eff'] = etc_info.information['SB_eff']
        self.config['zeropoint'] = etc_info.information['zeropoint']
        self.config['eff_wvl'] = etc_info.information['effWavelength']
        self.config['pixelScale_X'] = etc_info.information['pixelScale_X']
        self.config['pixelScale_Y'] = etc_info.information['pixelScale_Y']
        self.config['airmass'] = etc_info.information['airmass']
        self.config['seeing'] = etc_info.information['seeing_los_arcsec']
        self.config['camera'] = etc_info.information['channel']
        self.config['sky_site'] = etc_info.information['sky_site']
        self.config['verbose'] = str(etc_info.information['verbose'])

        if self.config['object_type'] == 'grb_sim':
            self.config['grb_mag'] = etc_info.information['mag']
            self.config['SNR'] = etc_info.information['SNR']

    def processConfigs(self):
        """
        Processes configuration information and save the information to a dictionary self.information.

        """
        # update the information dictionary
        self.information.update(self.config)

        # force gain to be float
        self.information['gain'] = float(self.config['gain'])

        #  If resized image is present, need to change image size
        if 'ImageResized' in self.information:
            self.information['xsize'] = self.information['ImageResized'][0]
            self.information['ysize'] = self.information['ImageResized'][1]

        # name of the output file, include CCDs
        # self.information['output'] = '{%s}'.format(self.infor)

        # booleans to control the flow
        if self.config['shotNoise'].lower() == 'yes':
            self.shotNoise = True
        else:
            self.shotNoise = False
        if self.config['addSources'].lower() == 'yes':
            self.Addsources = True
        else:
            self.Addsources = False
        if self.config['bleeding'].lower() == 'yes':
            self.bleeding = True
        else:
            self.bleeding = False
        if self.config['cosmicRays'].lower() == 'yes':
            self.cosmicRays = True
        else:
            self.cosmicRays = False
        if self.config['cosmetics'].lower() == 'yes':
            self.cosmetics = True
        else:
            self.cosmetics = False

        # Ajout Alix
        if self.config['addGRB'].lower() == 'yes':
            self.AddGRB = True
        else:
            self.AddGRB = False

        # these don't need to be in the config file
        try:
            val = self.config['readoutNoise']
            if val.lower() == 'yes':
                self.readoutNoise = True
            else:
                self.readoutNoise = False
        except:
            self.readoutNoise = False
        try:
            val = self.config['digiNoise']
            if val.lower() == 'yes':
                self.digiNoise = True
            else:
                self.digiNoise = False
        except:
            self.digiNoise = False
        try:
            val = self.config['background']
            if val.lower() == 'yes':
                self.background = True
            else:
                self.background = False
        except:
            self.background = False
        try:
            val = self.config['darkCurrent']
            if val.lower() == 'yes':
                self.darkCurrent = True
            else:
                self.darkCurrent = False
        except:
            self.darkCurrent = False
        try:
            val = self.config['shutterOpen']
            if val.lower() == 'yes':
                self.shutterOpen = True
            else:
                self.shutterOpen = False
        except:
            self.shutterOpen = False
        try:
            val = self.config['nonlinearity']
            if val.lower() == 'yes':
                self.nonlinearity = True
            else:
                self.nonlinearity = False
        except:
            self.nonlinearity = False
        try:
            val = self.config['Vignetting']
            if val.lower() == 'yes':
                self.Vignetting = True
            else:
                self.Vignetting = False
        except:
            self.Vignetting = False
        try:
            val = self.config['ADU']
            if val.lower() == 'yes':
                self.ADU = True
            else:
                self.ADU = False
        except:
            self.ADU = False
        try:
            val = self.config['Offset']
            if val.lower() == 'yes':
                self.Offset = True
            else:
                self.Offset = False
        except:
            self.Offset = False
        try:
            val = self.config['intscale']
            if val.lower() == 'yes':
                self.intscale = True
            else:
                self.intscale = False
        except:
            self.intscale = True

        try:
            val = self.config['saturation']
            if val.lower() == 'yes':
                self.saturation = True
            else:
                self.saturation = False
        except:
            self.saturation = False

        try:
            val = self.config['CrossTalk']
            if val.lower() == 'yes':
                self.CrossTalk = True
            else:
                self.CrossTalk = False
        except:
            self.CrossTalk = False
        try:
            val = self.config['FlatField']
            if val.lower() == 'yes':
                self.FlatField = True
            else:
                self.FlatField = False
        except:
            self.FlatField = False

        try:
            val = self.config['CreatePersistance']
            if val.lower() == 'yes':
                self.CreatePersistance = True
            else:
                self.CreatePersistance = False
        except:
            self.CreatePersistance = False

        try:
            val = self.config['Persistance']
            if val.lower() == 'yes':
                self.Persistance = True
            else:
                self.Persistance = False
        except:
            self.Persistance = False

        self.information['variablePSF'] = False

        self.booleans = dict(shotNoiseoise=self.shotNoise,
                             addsources=self.Addsources,
                             bleeding=self.bleeding,
                             cosmicRays=self.cosmicRays,
                             cosmetics=self.cosmetics,
                             background=self.background,
                             darkCurrent=self.darkCurrent,
                             readoutNoise=self.readoutNoise,
                             digiNoise=self.digiNoise,
                             nonLinearity=self.nonlinearity,
                             Vignetting=self.Vignetting,
                             ADU=self.ADU,
                             Offset=self.Offset,
                             intscale=self.intscale,
                             shutterOpen=self.shutterOpen,
                             saturation=self.saturation,
                             CrossTalk=self.CrossTalk,
                             FlatField=self.FlatField,
                             addGRB=self.AddGRB,
                             CreatePersistance=self.CreatePersistance,
                             Persistance=self.Persistance)

        if self.debug:
            pprint.pprint(self.information)

    def set_fits_header(self):
        """ Save information to save in FITS file header """

        self.fits_header = OrderedDict()
        self.fits_header['xsize'] = self.information['xsize']
        self.fits_header['ysize'] = self.information['ysize']
        self.fits_header['FWC'] = self.information['FWC']
        self.fits_header['DC'] = round(self.information['DC'], 3)
        self.fits_header['RN'] = self.information['RN']
        self.fits_header['gain'] = self.information['gain']
        self.fits_header['ZP'] = round(self.information['zeropoint'], 3)
        self.fits_header['NEXP'] = self.information['Nexp']
        self.fits_header['EXPTIME'] = self.information['exptime']
        self.fits_header['SB'] = self.information['SB_eff']
        self.fits_header['airmass'] = round(self.information['airmass'], 3)
        self.fits_header['seeing'] = round(self.information['seeing'], 3)
        self.fits_header['SKYSITE'] = self.information['sky_site']
        self.fits_header['camera'] = self.information['camera']
        # self.fits_header['Temp_cam'] = self.information['Temp_cam']
        self.fits_header['bits'] = self.information['bits']
        self.fits_header['camera_type'] = self.information['camera_type']
        self.fits_header['sensor'] = self.information['sensor']
        self.fits_header['binning_X'] = self.information['binning_X']
        self.fits_header['binning_Y'] = self.information['binning_Y']
        self.fits_header['filter'] = self.information['filter_folder'] + '-' + self.information['filter_band']
        self.fits_header['D_M1'] = self.information['D_M1']
        self.fits_header['D_M2'] = self.information['D_M2']
        self.fits_header['M2_factor'] = self.information['M2_factor']

        # add WCS to the header
        self.fits_header['WCSAXES'] = 2
        self.fits_header['CRPIX1'] = self.information['ysize'] / 2.
        self.fits_header['CRPIX2'] = self.information['xsize'] / 2.
        self.fits_header['CRVAL1'] = self.information['RA']
        self.fits_header['CRVAL2'] = self.information['DEC']
        self.fits_header['CTYPE1'] = 'RA---TAN'
        self.fits_header['CTYPE2'] = 'DEC--TAN'
        # north is up, east is left
        # self.fits_header['CD1_1'] = -self.config['pixelScale_X'] / 3600. #pix size in arc sec / deg
        # self.fits_header['CD1_2'] = 0.0
        # self.fits_header['CD2_1'] = 0.0
        # self.fits_header['CD2_2'] = self.config['pixelScale_Y'] / 3600.

        self.fits_header['CDELT1'] = -self.config['pixelScale_X'] / 3600.  # pix size in arc sec / deg
        self.fits_header['CDELT2'] = self.config['pixelScale_Y'] / 3600.
        self.fits_header['CROTA2'] = 0.0
        self.fits_header['DATE-OBS'] = datetime.datetime.isoformat(datetime.datetime.now())
        self.fits_header['INSTRUME'] = 'ImSimpyA'

        # create a new FITS file, using HDUList instance
        hdu = fits.PrimaryHDU()

        # new image HDU
        # hdu.data=self.image.astype(np.float32)

        # add input keywords to the header
        for key, value in self.fits_header.items():
            # truncate long keys
            if len(key) > 8:
                key = key[:7]
            try:
                hdu.header.set(key.upper(), value)
            except:
                try:
                    hdu.header.set(key.upper(), str(value))
                except:
                    pass

        # write booleans
        for key, value in self.booleans.items():
            # truncate long keys
            if len(key) > 8:
                key = key[:7]
            hdu.header.set(key.upper(), str(value), 'Boolean Flags')

        hdu.header.add_history('If questions, please contact David Corre (david.corre at lam.fr).')
        hdu.header.add_history('Created by ImSimpyA at %s' % datetime.datetime.isoformat(datetime.datetime.now()))
        hdu.verify('fix')

        # Create directory if not existing
        os.makedirs(os.path.dirname(self.path + '/images/' + self.information['output']), exist_ok=True)

        # write the actual file
        # hdu.writeto(self.path +'/images/'+self.information['output'],overwrite=True)
        # print (hdu.header)
        self.hdu_header = hdu.header

    def _createEmpty(self):
        """
        Creates and empty array of a given x and y size full of zeros.
        """
        self.image = np.zeros((self.information['ysize'], self.information['xsize']), dtype=np.float64)

    def objectOnDetector(self, object):
        """
        Version initiale de la fonction
        Tests if the object falls on the detector area being simulated.

        :param object: object to be placed to the self.image being simulated.
        :type object: list

        :return: whether the object falls on the detector or not
        :rtype: bool
        """
        ny, nx = self.finemap[object[3]].shape
        xt = object[0]  # position de l'objet
        yt = object[1]

        # the bounding box should be in the nominal scale
        fac = 1. / self.information['psfoversampling']

        # Assess the boundary box of the input image
        xlo = (1 - nx) * 0.5 * fac + xt
        xhi = (nx - 1) * 0.5 * fac + xt
        ylo = (1 - ny) * 0.5 * fac + yt
        yhi = (ny - 1) * 0.5 * fac + yt

        i1 = np.floor(xlo + 0.5)
        i2 = np.ceil(xhi + 0.5) + 1
        j1 = np.floor(ylo + 0.5)
        j2 = np.ceil(yhi + 0.5) + 1

        if i2 < 1 or i1 > self.information['xsize']:
            return False

        if j2 < 1 or j1 > self.information['ysize']:
            return False

        return True

    def overlayToCCD(self, data, obj):
        """
        Overlay data from a source object onto the self.image

        :param data: ndarray of data to be overlaid on to self.image
        :type data: ndarray
        :param obj: object information such as x,y position
        :type obj: list
        """
        # object centre x and y coordinates (only in full pixels, fractional has been taken into account already)
        xt = np.floor(obj[0]) - 1  # zero indexing  #position de l'objet
        yt = np.floor(obj[1]) - 1  # zero indexing

        # input array size
        nx = data.shape[1]
        ny = data.shape[0]

        # Assess the boundary box of the input image
        xlo = (1 - nx) * 0.5 + xt
        xhi = (nx - 1) * 0.5 + xt + 1
        ylo = (1 - ny) * 0.5 + yt
        yhi = (ny - 1) * 0.5 + yt + 1

        i1 = int(np.floor(xlo + 0.5))
        if i1 < 1:
            i1 = 0

        i2 = int(np.floor(xhi + 0.5))
        if i2 > self.information['xsize']:
            i2 = self.information['xsize']

        j1 = int(np.floor(ylo + 0.5))
        if j1 < 1:
            j1 = 0

        j2 = int(np.floor(yhi + 0.5))
        if j2 > self.information['ysize']:
            j2 = self.information['ysize']

        if i1 > i2 or j1 > j2:
            # print ('Object does not fall on the detector...')
            return

        ni = i2 - i1
        nj = j2 - j1

        # add to the image
        if ni == nx and nj == ny:
            # full frame will fit
            self.image[j1:j2, i1:i2] += data
        elif ni < nx and nj == ny:
            # x dimensions shorter
            if int(np.floor(xlo + 0.5)) < 1:
                # small values, left side
                self.image[j1:j2, i1:i2] += data[:, nx - ni:]
            else:
                # large values, right side
                self.image[j1:j2, i1:i2] += data[:, :ni]
        elif nj < ny and ni == nx:
            # y dimensions shorter
            if int(np.floor(ylo + 0.5)) < 1:
                # small values, bottom
                self.image[j1:j2, i1:i2] += data[ny - nj:, :]
            else:
                # large values, top
                self.image[j1:j2, i1:i2] += data[:nj, :]
        else:
            # both lengths smaller, can be in any of the four corners
            if int(np.floor(xlo + 0.5)) < 1 > int(np.floor(ylo + 0.5)):
                # left lower
                self.image[j1:j2, i1:i2] += data[ny - nj:, nx - ni:]
            elif int(np.floor(xlo + 0.5)) < 1 and int(np.floor(yhi + 0.5)) > self.information['ysize']:
                # left upper
                self.image[j1:j2, i1:i2] += data[:nj, nx - ni:]
            elif int(np.floor(xhi + 0.5)) > self.information['xsize'] and int(np.floor(ylo + 0.5)) < 1:
                # right lower
                self.image[j1:j2, i1:i2] += data[ny - nj:, :ni]
            else:
                # right upper
                self.image[j1:j2, i1:i2] += data[:nj, :ni]

        # ne pas ajouter des étoiles sur pix ref
        ref = np.loadtxt(self.path + self.information['PixRefFile']).astype(int)
        im = np.ndarray.flatten(self.image)
        im[ref] = 0

        self.image = np.reshape(im, [2048, 2048])


    def writeFITSfile(self, data, filename, unsigned16bit=False):
        """
        Writes out a simple FITS file.

        :param data: data to be written
        :type data: ndarray
        :param filename: name of the output file
        :type filename: str
        :param unsigned16bit: whether to scale the data using bzero=32768
        :type unsigned16bit: bool

        :return: None
        """
        if os.path.isfile(filename):
            os.remove(filename)

        # create a new FITS file, using HDUList instance
        # hdulist = fits.HDUList(fits.PrimaryHDU())
        hdu = fits.PrimaryHDU()

        # new image HDU
        # hdu = fits.ImageHDU(data=data)
        hdu.data = data

        # convert to unsigned 16bit int if requested
        if unsigned16bit:
            hdu.scale('int16', '', bzero=32768)
            hdu.header.add_history('Scaled to unsigned 16bit integer!')

        # add input keywords to the header
        for key, value in self.fits_header.items():
            # truncate long keys
            if len(key) > 8:
                key = key[:7]
            try:
                hdu.header.set(key.upper(), value)
            except:
                try:
                    hdu.header.set(key.upper(), str(value))
                except:
                    pass

        # write booleans
        for key, value in self.booleans.items():
            # truncate long keys
            if len(key) > 8:
                key = key[:7]
            hdu.header.set(key.upper(), str(value), 'Boolean Flags')

        # update and verify the header
        hdu.header.add_history('This is an intermediate data product no the final output!')
        hdu.header.add_history('Created by ImSimpyA at %s' % (datetime.datetime.isoformat(datetime.datetime.now())))
        hdu.verify('fix')

        # hdulist.append(hdu)

        # write the actual file
        # hdulist.writeto(filename)
        hdu.writeto(filename, overwrite=True)

    def configure(self, config_type):
        """
        Configures the simulator with input information and creates and empty array to which the final image will
        be build on.
        """
        if config_type == 'file': self.readConfigs()
        self.etc(config_type)
        self.processConfigs()
        self._createEmpty()
        self.set_fits_header()

    def generateObjectList(self):
        """ Generate object to simulate """

        if 'generate' in self.config['SourcesList']:
            from astropy.io import fits

            if 'output' in self.information['SourcesList']['generate']:
                output = self.path + '/data/catalog/' + self.information['SourcesList']['generate']['output']
            else:
                output = self.path + '/data/catalog/SourcesCatalog.txt'
            if 'frame' in self.information['SourcesList']['generate']:
                frame = self.information['SourcesList']['generate']['frame']
            else:
                frame = 'icrs'
            if 'band' in self.information['SourcesList']['generate']:
                band = self.information['SourcesList']['generate']['band']
            else:
                band = self.information['filter_band']

            RA = self.information['SourcesList']['generate']['RA']
            DEC = self.information['SourcesList']['generate']['DEC']
            radius = self.information['SourcesList']['generate']['radius']
            # _header=fits.open(self.path+'/images/'+self.information['output'])
            # header=_header['PRIMARY'].header

            if self.information['SourcesList']['generate']['catalog'] == 'Panstarrs':
                print('Downloading objects from Panstarrs catalog')
                from utils.createCatalogue import PanstarrsCatalog
                PanstarrsCatalog(RA, DEC, radius, band, self.config['eff_wvl'], self.hdu_header, frame=frame,
                                 output=output)
            else:
                from utils.createCatalogue import Viziercatalog
                print('Downloading objects from Vizier')
                print(RA, DEC, radius, band, self.config['eff_wvl'], self.hdu_header, self.information['SourcesList']['generate']['catalog'], frame, output)
                Viziercatalog(RA, DEC, radius, band, self.config['eff_wvl'], self.hdu_header, catalog=self.information['SourcesList']['generate']['catalog'], frame=frame, output=output)
            self.objects = np.loadtxt(output)


        elif "file" in self.config['SourcesList']:
            self.objects = np.loadtxt(self.path + '/data/catalog/' + self.information['SourcesList']['file'])

    def AddObjectToList(self,k):
        """Alix :  Créer un catalogue ne contenant que le GRB à simuler  """

        if 'gen' in self.config['SourcesToAdd']:
            from astropy.io import fits

            if 'output' in self.information['SourcesToAdd']['gen']:
                output = self.path + '/data/catalog/' + self.information['SourcesToAdd']['gen']['output']
            else:
                output = self.path + '/data/catalog/SourcesToAdd.txt'

            RA = self.information['SourcesToAdd']['gen']['RA']
            DEC = self.information['SourcesToAdd']['gen']['DEC']
            MAG = self.information['SourcesToAdd']['gen']['listeMAG'][k-1]
            radius = self.information['SourcesList']['generate']['radius']

            from utils.createCatalogue import CreateObject
            CreateObject(RA, DEC, MAG, 2, self.hdu_header, radius, output=output)

            object = np.loadtxt(output)
            object = object[np.newaxis, :]

            self.objects = object

    def readObjectlist(self):
        """
        Reads object list using numpy.loadtxt, determines the number of object types,
        and finds the file that corresponds to a given object type.

        The input catalog is assumed to contain the following columns:

            #. x coordinate
            #. y coordinate
            #. apparent magnitude of the object
            #. type of the object [0=star, number=type defined in the objects.dat]
            #. rotation [0 for stars, [0, 360] for galaxies]

        This method also displaces the object coordinates based on the
        CCD to be simulated.

        .. Note:: If even a single object type does not have a corresponding input then this method
                  forces the program to exit.
        """
        # self.objects = np.loadtxt(self.path+self.information['SourcesList'])
        # Add GRB on the object list
        if self.config['object_type'] == 'grb_sim':
            # If coordinates given in pixels
            if self.config['grb_coord_type'] == 'pixels':
                grb_coord_pix = self.config['grb_coords']
            elif self.config['grb_coord_type'] == 'RADEC':
                from astropy.io import fits
                import astropy.units as u
                import astropy.coordinates as coord
                from astropy.wcs import WCS

                c = coord.SkyCoord(self.config['grb_coords'][0], self.config['grb_coords'][1], unit=(u.deg, u.deg),
                                   frame='icrs')
                # _header=fits.open(self.path + '/images/' +self.information['output'])
                # header=_header['PRIMARY'].header

                w = WCS(self.hdu_header)
                world = np.array([[c.ra.deg, c.dec.deg]])
                # print (world)
                pix = w.all_world2pix(world, 1)
                # print (pix)
                # first transform WCS into pixels
                grb_coord_pix = pix[0]
            self.config['grb_coords_pix_X'] = grb_coord_pix[1]
            self.config['grb_coords_pix_Y'] = grb_coord_pix[0]
            self.objects = np.vstack(
                (self.objects, [grb_coord_pix[0], grb_coord_pix[1], self.config['grb_mag'], 1000, 0]))
            # Add GRB to the object list as a point source
            txt = 'GRB positionned at pixel coordinates (X,Y): {0:.2f},{1:.2f} with mag= {2:.2f}'.format(
                grb_coord_pix[0], grb_coord_pix[1], round(self.config['grb_mag'], 2))
            print(txt)

        # if only a single object in the input, must force it to 2D
        try:
            tmp_ = self.objects.shape[1]
        except:
            self.objects = self.objects[np.newaxis, :]

        # read in object types
        data = open(self.path + 'data/objects.dat').readlines()

        # only 2D array will have second dimension, so this will trigger the exception if only one input source
        tmp_ = self.objects.shape[1]
        # find all object types
        self.sp = np.asarray(np.unique(self.objects[:, 3]), dtype=int)

        # generate mapping between object type and data
        objectMapping = {}
        for stype in self.sp:

            if stype == 0 or stype == 1000 or stype == 2:
                # delta function
                objectMapping[stype] = 'PSF'
            else:
                for line in data:
                    tmp = line.split()
                    if int(tmp[0]) == stype:
                        # found match
                        if tmp[2].endswith('.fits'):
                            d, header = fits.getdata(self.path + '/' + tmp[2], header=True)
                            # print (type(d),d.shape,d,np.max(d))
                            if 'PIXSIZE1' in header:
                                ratio = float(header['PIXSIZE1'] / (self.information['xPixSize'] * 1e6))
                                print('ratio finemaps: %.2f' % ratio)
                                d2 = scipy.ndimage.zoom(d / d.sum(), ratio, order=3)
                                d2[d2 < 1e-6] = 0
                                if np.sum(d2) != 1: d2 = d2 / np.sum(d2)
                                image_size = d2.shape
                                # Assume same size horizontally and vertically
                                if image_size[0] % 2 == 0:
                                    width = int(image_size[0] / ratio / 4)
                                else:
                                    width = int((image_size[0] / ratio - 1) / 4)

                                # Assume same size horizontally and vertically
                                if np.ceil(image_size[0]) % 2 == 0:
                                    center = int((np.ceil(image_size[0])) / 2)
                                else:
                                    center = int((np.ceil(image_size[0]) - 1) / 2)
                                center = [center, center]
                                # print (ratio)
                                # print (center)
                                # print (width)
                                # print (center[0]-width,center[0]+width,center[1]-width,center[1]+width)
                                d3 = d2[center[0] - width:center[0] + width, center[1] - width:center[1] + width]

                                # print (type(d3),d3.shape,d3,np.max(d3))
                            else:
                                print('No pixel size found in header. Assume the same as the current telescope.')

                                d3 = d
                        else:
                            pass
                        objectMapping[stype] = dict(file=tmp[2], data=d3)
                        break

        self.objectMapping = objectMapping

    def generatePSF(self):
        """ Compute PSF if needed """

        PSF = dict()
        # Load atmosphere and instrument PSF
        if self.information['PSF']['total']['method'] == 'compute':
            for keyword in self.information['PSF']:
                if keyword != "total":
                    if "file" not in self.information['PSF'][keyword]:
                        if self.information['PSF'][keyword]['type'] == 'moffat':
                            if 'beta' in self.information['PSF'][keyword]:
                                beta = self.information['PSF'][keyword]['beta']
                            else:
                                beta = 2
                        else:
                            beta = 2
                        if 'seeing' in self.information['PSF'][keyword]:
                            seeing = self.information['PSF'][keyword]['seeing']
                        else:
                            seeing = self.config['seeing']

                        #  If PSF size bigger than image --> Limit PSF size to image size
                        if self.information['PSF'][keyword]['size'][0] > self.information['xsize']:
                            self.information['PSF'][keyword]['size'][0] = self.information['xsize']
                            print(
                                'PSF size along x axis bigger than image size!\nPSF size limited to image size along x axis now: %d Pixels' % (
                                self.information['xsize']))

                        if self.information['PSF'][keyword]['size'][1] > self.information['ysize']:
                            self.information['PSF'][keyword]['size'][1] = self.information['ysize']
                            print(
                                'PSF size along y axis bigger than image size!\nPSF size limited to image size along y axis now: %d Pixels' % (
                                self.information['ysize']))

                        PSFUtils.createPSF(
                            filename=self.path + '/data/psf/' + self.information['PSF'][keyword]['output'],
                            PSF_type=self.information['PSF'][keyword]['type'],
                            imsize=self.information['PSF'][keyword]['size'],
                            pixel_size=[self.config['xPixSize'], self.config['yPixSize']],
                            pixel_scale=self.config['pixelScale_X'], eff_wvl=self.config['eff_wvl'], seeing=seeing,
                            DM1=self.config['D_M1'], DM2=self.config['D_M2'], focal_length=self.config['Focal_length'],
                            oversamp=self.config['psfoversampling'], beta=beta, disp=False, unsigned16bit=False)

                        PSF[keyword] = self.path + '/data/psf/' + self.information['PSF'][keyword]['output']

                    else:
                        # Check pixel size and oversample if needed
                        hdr_ = fits.getheader(
                            self.path + '/data/psf/' + self.information['PSF'][keyword]['file'] + '.fits')
                        try:
                            if hdr_['XPIXELSZ'] != self.information['cameras'][self.information['channel']][
                                'Photocell_SizeX'] / oversamp or hdr_['YPIXELSZ'] != \
                                self.information['cameras'][self.information['channel']]['Photocell_SizeY'] / oversamp:
                                resampling = [self.information['cameras'][self.information['channel']][
                                                  'Photocell_SizeX'] / oversamp,
                                              self.information['cameras'][self.information['channel']][
                                                  'Photocell_SizeY'] / oversamp]

                                PSFUtils.resize(
                                    filename1=self.path + '/data/psf/' + self.information['PSF']['keyword']['file'],
                                    filename2=self.path + self.information['PSF']['keyword']['file'] + '_oversammpled',
                                    type='factor', resampling=resampling, overwrite=True, unsigned16bit=False)

                                PSF[keyword] = self.path + '/data/psf/' + self.information['PSF'][keyword][
                                    'file'] + '_oversampled'
                            else:
                                PSF[keyword] = self.path + '/data/psf/' + self.information['PSF'][keyword]['file']
                        except:
                            PSF[keyword] = self.path + '/data/psf/' + self.information['PSF'][keyword]['file']
            print('PSF convolution')
            # convolve atmosphere and instrument PSF to get the total PSF
            PSFUtils.convolvePSF(filename1=PSF['atmosphere'], filename2=PSF['instrument'],
                                 filename3=self.path + '/data/psf/' + self.information['PSF']['total']['file'])
            # PSFUtils.convolvePSF(filename1=PSF['instrument'],filename2=PSF['atmosphere'],filename3=self.path+self.information['PSF']['total']['output']+'_oversampled')
            # PSFUtils.resize(filename1=self.path+self.information['PSF']['total']['output']+'_oversampled',filename2=self.path+self.information['PSF']['total']['output'],resampling=32/self.information['psfoversampling'],type='sum')
            # PSFUtils.resize(filename1=self.path+self.information['PSF']['total']['output']+'_oversampled',filename2=self.path+self.information['PSF']['total']['output'],resampling=self.information['psfoversampling']/32,type='zoom')
            print('done')

    def readPSFs(self):
        """
        Reads in a PSF from a FITS file.

        .. Note:: at the moment this method supports only a single PSF file.
        """
        if self.information['variablePSF']:
            # grid of PSFs
            print('Spatially variable PSF not implemented -- exiting')
            sys.exit(-9)
        else:
            # single PSF
            self.PSF = fits.getdata(self.path + '/data/psf/' + self.information['PSF']['total']['file']).astype(
                np.float64)
            # Normalise if needed
            if np.sum(self.PSF) != 1: self.PSF /= np.sum(self.PSF)
            self.PSFx = self.PSF.shape[1]
            self.PSFy = self.PSF.shape[0]

    def generateFinemaps(self):
        """
        Generates finely sampled images of the input data.
        """
        self.finemap = {}
        self.shapex = {}
        self.shapey = {}
        for k, stype in enumerate(self.sp):
            if stype == 0 or stype == 1000 or stype == 2:
                data = self.PSF.copy().astype(np.float64)
                data /= np.sum(data)
                self.finemap[stype] = data
                self.shapex[stype] = 0
                self.shapey[stype] = 0
            else:

                #  Rescaled to pixel size

                if self.information['psfoversampling'] > 1.0:
                    data = scipy.ndimage.zoom(self.objectMapping[stype]['data'],
                                              self.information['psfoversampling'],
                                              order=0)
                else:
                    data = self.objectMapping[stype]['data']

                data[data < 0.] = 0.0
                if data.sum() != 1: data /= np.sum(data)
                self.finemap[stype] = data

    def addObjects(self,A,k):
        """
        Add objects from the object list to the CCD image (self.image).

        Scale the object's brightness in electrons and size using the input catalog magnitude.

        """
        # total number of objects in the input catalogue and counter for visible objects
        n_objects = self.objects.shape[0]
        visible = 0

        print('Total number of objects in the input catalog = %i' % n_objects)

        # calculate the scaling factors from the magnitudes
        # intscales = 10.0**(-0.4 * self.objects[:, 2])*self.information['magzero']) * self.information['exptime']
        # calculate the number of electrons from the magnitudes
        mag2elec = 10.0 ** (-0.4 * (self.objects[:, 2] - self.information['zeropoint'])) * self.information['exptime']


        intscales = mag2elec
        if ~self.random:
            # Using a fixed size-magnitude relation (equation B1 from Miller et al. 2012 (1210.8201v1).

            # testin mode will bypass the small random scaling in the size-mag relation
            # loop over exposures
            for i in range(self.information['Nexp']):
                # loop over the number of objects
                for j, obj in enumerate(self.objects):

                    stype = obj[3]

                    if self.objectOnDetector(obj):
                        visible += 1
                        if stype == 0 or stype == 1000:
                            # point source, apply PSF
                            if stype == 0:
                                txt = "Star: " + str(j + 1) + "/" + str(n_objects) + \
                                      " mag=" + str(round(obj[2], 2)) + " nb of el.=" + str(round(mag2elec[j], 2))
                            elif stype == 1000:
                                txt = "GRB: " + str(j + 1) + "/" + str(n_objects) + \
                                      " mag=" + str(round(obj[2], 2)) + " nb of el.=" + str(round(mag2elec[j], 2))

                            if self.debug: print(txt)

                            data = self.finemap[stype].copy()
                            # print (data.shape)
                            # map the data to new grid aligned with the centre of the object and scale
                            yind, xind = np.indices(data.shape)
                            if self.information['psfoversampling'] != 1.0:
                                yi = yind.astype(float) + self.information['psfoversampling'] * (obj[0] % 1)
                                xi = xind.astype(float) + self.information['psfoversampling'] * (obj[1] % 1)
                            else:
                                yi = yind.astype(float) + (obj[0] % 1)
                                xi = xind.astype(float) + (obj[1] % 1)
                            data = ndimage.map_coordinates(data, [yi, xi], order=1, mode='nearest')
                            if self.information['psfoversampling'] != 1.0:
                                data = scipy.ndimage.zoom(data, 1. / self.information['psfoversampling'], order=1)
                            # suppress negative numbers, renormalise and scale with the intscale
                            data[data < 0.0] = 0.0
                            sum = np.sum(data)
                            sca = mag2elec[j] / sum

                            """
                            print ('Obj coord: %.2f %.2f  / mag: %.2f   / finemap: min: %f  max: %f  mean: %f' % (obj[0],obj[1],obj[2],np.min(data*sca),np.max(data*sca),np.mean(data*sca)))
                            tolerance_min_finemap=1e-2
                            test_min=np.min(data*sca)
                            zoom=2
                            while test_min > tolerance_min_finemap:
                               print ('increased fineap map by 2 ')
                               data = scipy.ndimage.zoom(data, zoom)
                               test_min=np.min(data*sca)
                               print (np.min(data*sca))
                            """
                            # sca = intscales[j] / sum
                            # data = ne.evaluate("data * sca")
                            # sca = mag2elec[j]

                            # numexpr apparently faster than numpy for big arrays
                            data = ne.evaluate("data * sca")

                            # print ('Obj coord: %.2f %.2f  / mag: %.2f   / finemap: min: %f  max: %f  mean: %f' % (obj[0],obj[1],obj[2],np.min(data),np.max(data),np.mean(data)))
                            # data[data < 0.0] = 0.0

                            # overlay the scaled PSF on the image
                            self.overlayToCCD(data, obj)

                        elif stype == 2:  # ajout Alix objet variable GRB
                            if 'gen' in self.config['SourcesToAdd']:
                                Nframe = self.config['Nframe']
                                objectVariable = self.config['SourcesToAdd']['gen']['listeMAG']
                                mag = objectVariable[k-1]
                                print('Magnitude GRB frame numero',k,'=' ,mag)
                                magelec = 10.0 ** (-0.4 * (mag - self.information['zeropoint'])) * 1.33


                            mag2elec[j] = magelec
                            print('mag2elec variable', magelec)
                            data = self.finemap[stype].copy()
                            # print (data.shape)
                            # map the data to new grid aligned with the centre of the object and scale
                            yind, xind = np.indices(data.shape)
                            if self.information['psfoversampling'] != 1.0:
                                yi = yind.astype(float) + self.information['psfoversampling'] * (obj[0] % 1)
                                xi = xind.astype(float) + self.information['psfoversampling'] * (obj[1] % 1)
                            else:
                                yi = yind.astype(float) + (obj[0] % 1)
                                xi = xind.astype(float) + (obj[1] % 1)
                            data = ndimage.map_coordinates(data, [yi, xi], order=1, mode='nearest')
                            if self.information['psfoversampling'] != 1.0:
                                data = scipy.ndimage.zoom(data, 1. / self.information['psfoversampling'], order=1)
                            # suppress negative numbers, renormalise and scale with the intscale
                            data[data < 0.0] = 0.0
                            sum = np.sum(data)
                            sca = mag2elec[j] / sum

                            # sca = intscales[j] / sum
                            # data = ne.evaluate("data * sca")
                            # sca = mag2elec[j]

                            # numexpr apparently faster than numpy for big arrays
                            data = ne.evaluate("data * sca")

                            # print ('Obj coord: %.2f %.2f  / mag: %.2f   / finemap: min: %f  max: %f  mean: %f' % (obj[0],obj[1],obj[2],np.min(data),np.max(data),np.mean(data)))
                            # data[data < 0.0] = 0.0

                            # overlay the scaled PSF on the image
                            self.overlayToCCD(data, obj)



                        else:
                            # extended source, rename finemap
                            data = self.finemap[stype].copy()
                            # map the data to new grid aligned with the centre of the object
                            yind, xind = np.indices(data.shape)
                            if self.information['psfoversampling'] != 1.0:
                                yi = yind.astype(float) + self.information['psfoversampling'] * (obj[0] % 1)
                                xi = xind.astype(float) + self.information['psfoversampling'] * (obj[1] % 1)
                            else:
                                yi = yind.astype(float) + (obj[0] % 1)
                                xi = xind.astype(float) + (obj[1] % 1)

                            # yi = yind.astype(np.float) + (obj[0] % 1)
                            # xi = xind.astype(np.float) + (obj[1] % 1)
                            data = ndimage.map_coordinates(data, [yi, xi], order=1, mode='nearest')

                            conv = convolution(data, self.PSF, self.information['mode'])

                            # suppress negative numbers
                            conv[conv < 0.0] = 0.0

                            # renormalise and scale to the right magnitude
                            sum = np.sum(conv)
                            # sca = intscales[j] / sum
                            sca = mag2elec[j] / sum
                            conv = ne.evaluate("conv * sca")

                            # tiny galaxies sometimes end up with completely zero array
                            # checking this costs time, so perhaps this could be removed
                            if np.isnan(np.sum(conv)):
                                continue

                            # overlay the convolved image on the image
                            self.overlayToCCD(conv, obj)

                    else:
                        # not on the screen
                        # print ('Object %i is outside the detector area' % (j + 1))
                        pass
            #print('normal', self.information['exptime'], self.objects[13, 2], mag2elec[13])
            print('ZERO POINT', self.information['zeropoint'])
            print('%i/%i objects were placed on the detector' % (visible, n_objects))

            '''plt.figure('image telescope')
            plt.imshow(self.image,vmin=np.quantile(self.image, 0.01), vmax=np.quantile(self.image, 0.99))
            plt.colorbar()
            plt.show()'''



    def addCosmetics(self,A):
        """ Add the cosmetics effects """
        deadPixs = fits.getdata(self.path + '/data/Cosmetics/' + self.information['DeadPixFile'])
        # HotPixs=fits.getdata(self.path+'/data/Cosmetics/'+self.information['HotPixFile'])

        '''plt.figure('av dead')
        plt.imshow(self.image, vmin=np.quantile(self.image, 0.05), vmax=np.quantile(self.image, 0.95))
        plt.colorbar()'''

        A *= deadPixs

        '''plt.figure('dead')
        plt.imshow(self.image, vmin=np.quantile(self.image, 0.05), vmax=np.quantile(self.image, 0.95))
        plt.colorbar()
        plt.show()'''

        # Alix : fonction passée dans le courant d'obscurité
        # on multiplie le flux du pixel chaud ( en e-/s) par le temps d'exposition.
        # Texp = self.information['exptime']
        # self.image += HotPixs * Texp

        # initial
        # self.image+=HotPixs
        return(A)

    def applyCreatePersistance(self,A,k):
        """ Creation de la carte de persistance à enregistrer pour l'exposition suivante en mV"""
        Nfin = int(np.round(self.information['Nfin']))
        actif = np.loadtxt(self.path + self.information['PixActifFile']).astype(int)


        if k == Nfin-1:
            nom = self.information['nom'] + '_persistance.fits'
            path = '/home/alix/anaconda3/dcorre-ImSimpyA-42ac6cb/ImSimpyA/data/Persistance/'
            satu = np.ndarray.flatten(fits.getdata(self.path + '/data/' + self.information['SaturationFile']))

            indPers = np.intersect1d(np.argwhere(np.ndarray.flatten(A) > satu), actif)
            #print('indpers',len(indPers))
            carte = np.zeros(2048*2048)
            carte[indPers]=1

            primary = fits.PrimaryHDU()
            image_hdu = fits.ImageHDU(carte)
            hdul = fits.HDUList([primary, image_hdu])
            hdul.writeto(path + nom, overwrite=True)


    def applyPersistance(self,k,A):
        """ Application de la carte de persistance à appliquer (exposition précédente) déjà en electrons"""

        #actif = np.loadtxt('/home/alix/anaconda3/dcorre-ImSimpyA-42ac6cb/ImSimpyA/data/PixViolet.txt').astype(int)
        path = '/home/alix/anaconda3/simu/ImSimpyA/data/Persistance/'
        nom = self.information['nomPersistance']
        Treset = self.information['Treset']  # 5*60  # temps depuis le premier reset

        PixPeristants = fits.open(path + nom)
        PixPeristants = PixPeristants[1].data

        indPers = np.argwhere(np.ndarray.flatten(PixPeristants) > 0)

        Texp = self.information['exptime']*k + Treset

        conv = fits.open(self.path + '/data/' + self.information['PersistanceConv'])
        conv = conv[0].data
        conv[np.argwhere(conv == 0)] = 1

        amp = fits.open(self.path + '/data/' + self.information['PersistanceAmp'])
        amp = amp[0].data
        amp_1 = conv * amp[0]
        amp_2 = conv * amp[1]
        amp_3 = conv * amp[2]

        tau = fits.open(self.path + '/data/' + self.information['PersistanceTau'])
        tau = tau[0].data
        tau_1 = conv * tau[0]
        tau_2 = conv * tau[1]
        tau_3 = conv * tau[2]

        if k == 1 :
            persistance0 = amp_1 * (1 - np.exp(-Texp / tau_1)) + amp_2 * (1 - np.exp(-Texp / tau_2)) + amp_3 * (
                    1 - np.exp(-Texp / tau_3))  # persistance si on avait commencé à Trest =0

            ptrest= (amp_1 * (1 - np.exp((-Treset) / tau_1)) + amp_2 * (1 - np.exp((-Treset) / tau_2)) + amp_3 * (1 - np.exp((-Treset ) / tau_3)))

            persistance = persistance0 - ptrest

            '''print(Texp, Treset)
            print('persistance totale', persistance0[500,500])
            print('persistance à trest', ptrest[500, 500])
            print('persistance acq', persistance[500,500])'''
            # persistance - persistance accumulée du reset jusqu'au début de l'acquisition
            persistance = np.ndarray.flatten(persistance)
            persistance[np.isnan(persistance)] = 0

        else :
            persistance0 = amp_1 * (1 - np.exp(-Texp / tau_1)) + amp_2 * (1 - np.exp(-Texp / tau_2)) + amp_3 * ( 1 - np.exp(-Texp / tau_3))  # persistance si on avait commencé à Trest =0
            #print('persistance tot',persistance0[500,500])
            Tprec = Texp - 1.33
            #print(Tprec, Texp)
            persistance = persistance0 - (amp_1 * (1 - np.exp((-Tprec ) / tau_1)) + amp_2 * ( 1 - np.exp((-Tprec) / tau_2)) + amp_3 * (1 - np.exp((-Tprec) / tau_3)))
            # persistance - persistance accumulée du reset jusqu'au début de l'acquisition
            #print('persistance acq', persistance[500,500])
            persistance = np.ndarray.flatten(persistance)
            persistance[np.isnan(persistance)] = 0

            '''plt.figure('persistance tot')
        plt.imshow(np.reshape(persistance,[2048, 2048]), vmin=np.quantile(persistance, 0.05), vmax=np.quantile(persistance, 0.95))
        plt.colorbar()'''

        map = np.zeros(2048 * 2048)
        map[indPers] = 1
        pers = persistance * map
        pers[np.isnan(pers)] = 0
        persistance = np.reshape(pers, [2048, 2048])

        '''plt.figure('image avant persistance')
        plt.imshow(self.image, vmin=np.quantile(self.image, 0.05), vmax=np.quantile(self.image, 0.95))
        plt.colorbar()'''

        A = A + persistance
        '''plt.figure('persistance')
        plt.imshow(persistance, vmin=np.quantile(persistance, 0.05), vmax=np.quantile(persistance, 0.95))
        plt.colorbar()
        plt.figure('image avec persistance')
        plt.imshow(A, vmin=np.quantile(A, 0.05), vmax=np.quantile(A, 0.95))
        plt.colorbar()
        plt.show()'''

        return(A)

        '''primary = fits.PrimaryHDU()
        image_hdu = fits.ImageHDU(persistance/10)
        hdul = fits.HDUList([primary, image_hdu])
        hdul.writeto('/home/alix/Documents/Programmes/SimuPersistance/Persistance'+str(Texp), overwrite=True)'''



    def addCosmicRays(self,k,A):

        """ Add cosmic rays """
        Texp =  k* 1.33  # temps de l'impact
        Nframe = k
        cosmic = fits.open(self.path + '/data/Cosmics/' + self.information['CosmicsFile'])
        pos = (cosmic[1].data).astype(int)
        energie = cosmic[2].data
        temps = (cosmic[3].data).astype(int)
        nbCos = len(pos)

        imageCos = np.zeros(2048 * 2048)

        '''plt.figure('im cos avant')
        plt.imshow(self.image, vmin=np.quantile(self.image, 0.05), vmax=np.quantile(self.image, 0.95))'''

        for i in range(nbCos):
            if Texp >= temps[i] and Texp < temps[i]+1.33:
                # if Texp*Nframe <= temps[i] and Texp*(Nframe+1)>= temps[i] :
                imageCos[pos[i]] = energie[i] * 10


        A = A+ np.reshape(imageCos, [2048, 2048])



        '''plt.figure('im cos')
        plt.imshow(self.image, vmin=np.quantile(self.image, 0.05), vmax=np.quantile(self.image, 0.95))
        plt.show()'''

        # pass
        return(A)

    def applyVignetting(self):
        """ Add vignetting  """
        vignetting = fits.getdata(self.path + '/data/Vignetting/' + self.information['VignettingFile'])
        self.image *= vignetting

    def applyFlatField(self):
        """ Add FlatField  """

        '''plt.figure('avant ff')
        self.image[np.isnan(self.image)] = 0
        plt.imshow(self.image, vmin=np.quantile(self.image, 0.05), vmax=np.quantile(self.image, 0.95))'''


        # FlatField calculé sur le det ALFA
        FlatField = fits.getdata(self.path + '/data/' + self.information['FlatFieldFile'])

        ref = np.loadtxt(self.path +self.information['PixRefFile']).astype(int)
        FlatField = np.ndarray.flatten(FlatField)  # sinon pb lors de la correction par les pixels de ref, leur non linéarité ne peux pas être calibrée par illumination : pas sensible light
        FlatField[ref] = 1
        FlatField[np.argwhere(FlatField <= 0)] = 1
        FlatField = np.reshape(FlatField, [2048, 2048])

        self.image *= FlatField

        # SaveFit(self.image, 'CarteEntrée', 'CARTE', 'entreeTelescope', 1, path='/home/alix/anaconda3/dcorre-ImSimpyA-42ac6cb/rampes/')

        '''plt.figure('ff')
        self.image[np.isnan(self.image)]=0
        plt.imshow(self.image,vmin=180, vmax= 206) #min = np.quantile(self.image, 0.05),vmax = np.quantile(self.image, 0.95))
        plt.colorbar()
        plt.show()'''

    def applyNonLinearity(self,k,A):
        """ Add non linearity  """

        NL = fits.getdata(self.path + '/data/NonLinearity/' + self.information['NonLinearityFile'])

        ref = np.loadtxt(self.path +self.information['PixRefFile']).astype(int)
        NL = np.ndarray.flatten(NL)  # sinon pb lors de la correction par les pixels de ref, leur non linéarité ne peux pas être calibrée par illumination : pas sensible light
        NL[ref] = 0

        offset = np.ndarray.flatten(fits.getdata(self.path + '/data/Offset/' + self.information['OffsetFile']))
        satu = np.ndarray.flatten(fits.getdata(self.path + '/data/' + self.information['SaturationFile']))
        sat = (satu - offset) * 10  # pour mettre en electrons
        im = np.ndarray.flatten(A)
        a = np.argwhere(im > 1.1 * sat)
        NL[a] = 0

        #av = np.copy(A)
        #print(np.median(NL))
        #print(np.median(A))
        NL = np.reshape(NL, [2048, 2048])
        A = A + A*A*NL*k      # on met un + car NL déjà négatif # alpha doit être calulé sur des cartes non corrigées des pix ref


        return(A)
        '''plt.figure('NL')
        self.A[np.isnan(self.A)]=0
        plt.imshow(self.A,vmin=190, vmax= 280) #min = np.quantile(self.image, 0.05),vmax = np.quantile(self.image, 0.95))
        plt.colorbar()
        plt.show()

        plt.figure('NL')
        self.A[np.isnan(self.A)] = 0
        rand = np.random.rand(len(self.A))
        self.A=self.A+rand
        rand = np.random.rand(len(av))
        av = av + rand
        plt.hist(np.ndarray.flatten(av), bins=500, range=[100, 280], alpha=0.6, label='sans non-linéarité')
        plt.hist(np.ndarray.flatten(self.A),bins = 500, range = [100,280],alpha = 0.6, label='avec non-linéarité')
        plt.legend(prop={'size': 20})
        plt.show()'''
        # pass

    def applyDarkCurrent(self):
        """
        Apply dark current. Scales the dark with the exposure time.

        """
        #self.image = 2255 * np.ones([2048, 2048]) * self.information['exptime']  # test d'une illumination constante.

        DC = np.reshape(fits.getdata(self.path + '/data/DarkCurrent/' + self.information['DarkFile']), [2048, 2048])

        ref = np.loadtxt(self.path +self.information['PixRefFile']).astype(int)
        dc = np.ndarray.flatten(DC)
        dc[ref] = np.median(dc)

        av = np.copy(self.image)

        # Alix : on multiplie le  DC du pixel  ( en e-/s) par le temps d'exposition : prend en compte DC et pixels chauds.
        Texp = self.information['exptime']
        DC = np.reshape(dc,[2048,2048])
        self.image += DC * Texp

        '''plt.figure('DC')
        plt.imshow(self.image,vmin=180, vmax= 206) #vmin = np.quantile(self.image, 0.05),vmax = np.quantile(self.image, 0.95))
        plt.colorbar()'''

        '''plt.figure('dc hist')
        rand = np.random.rand(len(self.image))
        self.image=self.image+rand
        rand = np.random.rand(len(av))
        av = av + rand
        self.image[np.isnan(self.image)] = 0
        plt.hist(np.ndarray.flatten(av),range =[150,240],bins = 500, alpha=0.6, label='sans courant d obscurite')
        plt.hist(np.ndarray.flatten(self.image), range =[150,240] ,bins = 500, alpha=0.6, label='avec courant d obscurite')
        plt.legend(prop={'size': 20})
        plt.xscale('log')
        plt.show()'''

    def applySkyBackground(self):
        """
        Apply dark the sky background. Scales the background with the exposure time.

        Additionally saves the image without noise to a FITS file.
        """
        av = np.copy(self.image)


        sky_back_el = self.information['Sky_CountRate']

        bcgr = self.information['exptime'] * sky_back_el


        # sky_image = np.random.poisson(bcgr,size=self.image.shape).astype(np.float64)
        sky_image = np.ones([2048, 2048]) * bcgr  # on considère le fond de ciel constant sur le détecteur : ligne Alix

        # ne pas ajouter le bruit de font de ciel aux pix de reférence
        ref = np.loadtxt(self.path +self.information['PixRefFile']).astype(int)
        sky_f = np.ndarray.flatten( sky_image)  # sinon pb lors de la correction par les pixels de ref, leur non linéarité ne peux pas être calibrée par illumination : pas sensible light
        sky_f[ref] = 0
        sky_image = np.reshape(sky_f, [2048, 2048])

        self.image += sky_image

        # SaveFit(self.image, 'CarteEntree', 'CARTE', 'entreeTelescope', 1,path='/home/alix/anaconda3/dcorre-ImSimpyA-42ac6cb/rampes/')

        '''plt.figure('sky im')
        plt.imshow(self.image,vmin = np.quantile(self.image, 0.05),vmax = np.quantile(self.image, 0.99))
        plt.colorbar()
        plt.show()'''

        '''plt.figure('sky hist')
        self.image[np.isnan(self.image)] = 0
        rand = np.random.rand(len(self.image))
        self.image=self.image+rand
        rand = np.random.rand(len(av))
        av = av + rand
        plt.hist(np.ndarray.flatten(av), range=[0, 500], bins=500, alpha=0.6, label='sans fond de ciel')
        plt.hist(np.ndarray.flatten(self.image), range = [0,500],bins = 500,alpha=0.6, label='avec fond de ciel')
        plt.legend(prop={'size': 20})
        plt.yscale('log')
        plt.show()'''

    def applyShotNoise(self,A):
        """
        Add Poisson noise to the image.
        """
        im = np.ndarray.flatten(A)
        im[np.argwhere(im < 0)] = 0

        actif = np.loadtxt(self.path +self.information['PixActifFile']).astype(int)

        image = im[actif]
        av = A
        image= np.reshape(image, [2040, 2040])
        image = np.random.poisson(image).astype(np.float64)

        im[actif] = np.ndarray.flatten(image)

        print(np.shape(A), np.shape(im))
        A = np.reshape(im,[2048,2048])

        #print('random',self.A[1500,1800])

        '''plt.figure('shot noise')
        plt.imshow(self.image, vmin=190, vmax= 280) #np.quantile(self.image, 0.05), vmax=np.quantile(self.image, 0.95))
        plt.colorbar()
        plt.show()'''

        '''plt.figure('shot noise hist')
        rand = np.random.rand(len(A))
        A=A+rand
        rand = np.random.rand(len(av))
        av = av + rand
        A[np.isnan(A)] = 0
        plt.hist(np.ndarray.flatten(av), range = [100,280], bins = 500,alpha=0.6, label='sans bruit de photons')
        plt.hist(np.ndarray.flatten(A),range = [100,280], bins = 500,alpha=0.6, label='avec bruit de photons')
        plt.legend(prop={'size': 20})
        plt.show()'''
        return(A)

    def applyReadoutNoise(self,A):
        """
        Applies readout noise to the image being constructed.

        The noise is drawn from a Normal (Gaussian) distribution with average=0.0 and std=readout noise.
        """

        noise = np.random.normal(loc=0.0, scale=self.information['RN'], size=A.shape)
        print('readout',self.information['RN'])

        av = np.copy(A)

        '''plt.figure('noise')
        plt.hist(np.ndarray.flatten(noise),bins=500)

        print(np.median(noise),np.std(noise))
        print(np.min(noise),np.max(noise))'''

        # add to the image
        A += (noise/10)

        '''plt.figure('rn')
        plt.imshow(A,vmin = np.quantile(A, 0.05),vmax = np.quantile(A, 0.95))
        plt.colorbar()
        plt.show()'''

        '''plt.figure('rn hist')
        rand = np.random.rand(len(A))
        A=A+rand
        rand = np.random.rand(len(av))
        av = av + rand
        A[np.isnan(A)] = 0
        plt.hist(np.ndarray.flatten(av),range = [5,400],bins = 500, alpha=0.6, label='sans bruit de lecture')
        plt.hist(np.ndarray.flatten(A), range = [5,400],bins = 500,alpha=0.6, label='avec bruit de lecture')
        plt.legend(prop={'size': 20})
        plt.show()'''

        return(A)


    def electrons2ADU(self,A):
        """
        Convert from electrons to ADUs using the value read from the configuration file.
        """
        gain_map = fits.getdata(self.path + '/data/GainMap/' + self.information['GainMapFile'])

        A /= gain_map

        #print('max electrons', np.nanmax(A))

        '''self.image[np.isnan(self.image)]=0
        plt.figure('ADU')
        plt.imshow(self.image,vmin = np.quantile(self.image, 0.05),vmax = np.quantile(self.image, 0.95))
        plt.colorbar()
        plt.show()'''
        return(A)

    def addOffset(self,F):
        """
        Add the offset (bias) in ADU
        """
        offset = fits.getdata(self.path + '/data/Offset/' + self.information['OffsetFile'])
        offset = np.reshape(offset, [2048, 2048])
        print('max offset', np.max(offset))
        F += offset

        F[np.isnan(F)] = 0
        #print('max image offset', np.max(self.image))

        '''self.image[np.isnan(self.image)]=0
        plt.figure('offset')
        plt.imshow(self.image,vmin = np.quantile(self.image, 0.05),vmax = np.quantile(self.image, 0.95))
        plt.colorbar()
        plt.show()'''
        return(F)


    def writeOutputs(self,F):
        """
        Writes out a FITS file using PyFITS and converts the image array to 16bit unsigned integer as
        appropriate for VIS.

        Updates header with the input values and flags used during simulation.
        """
        # if os.path.isfile(self.path +  self.information['output_dir']+self.information['output']):
        #    os.remove( self.path + self.information['output_dir']+self.information['output'])

        # create a new FITS file, using HDUList instance
        hdu = fits.PrimaryHDU()

        # hdu=fits.open(self.path + '/images/'+self.information['output'])

        # new image HDU
        # float 32 bits
        # hdu['PRIMARY'].data=self.image.astype(np.float32)
        # UNsigned Integer 16 bits (0 to 65535)
        hdu.data = F.astype(np.uint16)
        hdu.header = self.hdu_header

        # write the actual file. Path should exists, already checked while creating the headers.
        hdu.writeto(self.path + 'images/' + self.information['output'], overwrite=False)

    def applySaturation(self,F):
        """
        Met au maximum de remplissage des puits les pixels de l'image (self.image) qui dépassent le niveau de saturation
        défini par l'argument max.

        :param max: maximum value the the integer array may contain [default 120 000 électrons ce qui donne 350000
        en prenant en compte l'offset qui est
        :type max: float

        :return: None
        """
        # max=self.information['FWC']
        # satu = fits.getdata('/home/alix/Documents/Programmes/Cartes_det4_Pixel_Reset/CarteSaturation_Det4.fits')
        satu = fits.open(self.path + '/data/' + self.information['SaturationFile'])
        maxi = np.ndarray.flatten(satu[1].data)
        # max=120000

        # cut of the values larger than max
        im = np.ndarray.flatten(F)
        a = np.argwhere(im >= maxi)

        im[a] = maxi[a]


        F = np.reshape(im, [2048, 2048])

        return(F)

        #print('max', np.max(F))

        '''self.image[np.isnan(self.image)]=0
        plt.figure('satu')
        plt.imshow(self.image,vmin = np.quantile(self.image, 0.05),vmax = np.quantile(self.image, 0.95))
        plt.show()'''

    def applyCrossTalk(self,A):
        """
        appliquer l'impact du cross talk aux images
        :param ct: matrice 3*3 des coeficients de cross talk
        :type ct: array
        :return: None
        """
        crosstalk = fits.open(self.path + '/data/' + self.information['CrossTalkFile'])
        ct = np.zeros([3, 3])
        ct[0] = crosstalk[1].data
        ct[1] = crosstalk[2].data
        ct[2] = crosstalk[3].data
        # print('ct',np.shape(ct),ct)
        # ct = np.array([[0.0002, 0.002, 0.0002], [0.002, 1-(0.002*4+0.0002*4), 0.002], [0.0002, 0.002, 0.0002]])

        # on coupe les valeurs au dela de la saturation pour ne pas générer un cross talk enorme lorsque l'on dépasse de beaucoup la saturation
        offset = np.ndarray.flatten(fits.getdata(self.path + '/data/Offset/' + self.information['OffsetFile']))
        #satu = np.ndarray.flatten(fits.getdata('/home/alix/anaconda3/dcorre-ImSimpyA-42ac6cb/ImSimpyA/data/saturation_1V.fits'))
        satu = fits.open(self.path + '/data/' + self.information['SaturationFile'])
        satu = np.ndarray.flatten(satu[1].data)

        sat = (satu - offset) * 10
        im = np.ndarray.flatten(A)

        actif = np.loadtxt(self.path +self.information['PixActifFile']).astype(int)
        a = np.intersect1d(actif, np.argwhere(im > 1.1 * sat))
        im[a] = 1.1 * sat[a]
        av = im
        im = np.reshape(im, [2048, 2048])

        '''plt.figure('cross talk avant')
        plt.imshow(self.image, vmin=np.quantile(self.image, 0.05), vmax=np.quantile(self.image, 0.95))'''

        image = np.ndarray.flatten(signal.convolve2d(im, ct, mode='same', boundary='fill'))

        refct = np.loadtxt('/home/alix/anaconda3/dcorre-ImSimpyA-42ac6cb/ImSimpyA/data/ref_ct.txt').astype(int)
        image[refct] = np.ndarray.flatten(im)[refct]

        A = np.reshape(image, [2048, 2048])

        '''plt.figure('cross talk')
        plt.imshow(self.image,vmin = np.quantile(self.image, 0.05),vmax = np.quantile(self.image, 0.95))
        plt.figure('ct hist')
        rand = np.random.rand(len(self.image))
        self.image=self.image+rand
        rand = np.random.rand(len(av))
        av = av + rand
        self.image[np.isnan(self.image)] = 0
        plt.hist(np.ndarray.flatten(av),bins = 500,range = [0,300], alpha=0.6, label='sans cross talk')
        plt.hist(np.ndarray.flatten(self.image),bins = 500,range = [0,300],  alpha=0.6, label='avec cross talk')
        plt.legend(prop={'size': 20})
        plt.legend()
        plt.show()'''
        return(A)

    def simulate(self, config_type='file'):
        """
        Create a single simulated image defined by the configuration file.
        Will do all steps defined in the config file sequentially.

        :return: None
        """
        # if self.config['verbose'] == 'True': print ("Read config file and execute ETC")
        print("Read config file and execute ETC")
        self.configure(config_type)
        print("Building image: %s:" % self.information['output'])
        # print (self.information)

        Nfin = int(np.round(self.information['Nfin']))
        nf =  np.arange(1,Nfin+1)

        F=np.zeros([len(nf),2048,2048])

        if self.Addsources:
            # if self.config['verbose'] == 'True': print ("Read objecct list")
            print("\tGENERATE OBJECTS CATALOG")
            self.generateObjectList()
            #if self.AddGRB:
            #    self.AddObjectToList()  # ajouter un objet fixe à la liste: fonction crée Alix
            self.readObjectlist()

            # if self.config['verbose'] == 'True': print ("Generate PSF")
            print("\tGENERATE PSF")
            self.generatePSF()
            self.readPSFs()
            self.generateFinemaps()

            # if self.config['verbose'] == 'True': print ("Add objects")
            print("\tADD OBJECTS")
            self.addObjects(self.image,1)

        if self.background:
            # if self.config['verbose'] == 'True': print ("Add Sky background")
            print("\tAdd Sky background")
            self.applySkyBackground()

        if self.FlatField:
            # if self.config['verbose'] == 'True': print ("Add FlatField")
            print("\tAdd FlatField")
            self.applyFlatField()

        if self.darkCurrent:
            # if self.config['verbose'] == 'True': print ("Add dark current")
            print("\tAdd dark current")
            self.applyDarkCurrent()


        image = np.copy(self.image)
        A = image
        plt.figure('A')
        plt.imshow(A, vmin=np.quantile(A, 0.1), vmax=np.quantile(A, 0.9))
        plt.colorbar()
        #plt.show()


        for k in nf:
            A=image

            if self.AddGRB:
                self.image = np.zeros((self.information['ysize'], self.information['xsize']), dtype=np.float64)
                # if self.config['verbose'] == 'True': print ("Read objecct list")
                print("\tGENERATE OBJECTS CATALOG")
                #self.generateObjectList()
                if self.AddGRB:
                    self.AddObjectToList(k)  # ajouter un objet fixe à la liste: fonction crée Alix
                self.readObjectlist()

                # if self.config['verbose'] == 'True': print ("Generate PSF")
                print("\tGENERATE PSF")
                self.generatePSF()
                self.readPSFs()
                self.generateFinemaps()
                # if self.config['verbose'] == 'True': print ("Add objects")
                print("\tADD OBJECTS")
                self.addObjects(A,k) #a copier, modifier et faire pour cas GRB : juste if grb sur A et pas self.image
                A = self.image + A


            if self.Persistance:
                print("\tCreate Persistance")
                A= self.applyPersistance(k,A)

            if self.shotNoise:
                # if self.config['verbose'] == 'True': print ("Apply Shot noise")
                print("\tApply Shot noise")
                A=self.applyShotNoise(A)

            if self.cosmicRays:
                # if self.config['verbose'] == 'True': print ("Add cosmic Rays")
                print("\tAdd cosmic Rays")
                A=self.addCosmicRays(k,A)

            if self.CrossTalk:
                # if self.config['verbose'] == 'True': print ("Apply Saturation")
                print("\tApply Cross Talk")
                A=self.applyCrossTalk(A)

            if self.nonlinearity:
                # if self.config['verbose'] == 'True': print ("Add non linearity")
                print("\tAdd non linearity")
                A=self.applyNonLinearity(k,A)

            if self.cosmetics:
                # if self.config['verbose'] == 'True': print ("Add cosmetics")
                print("\tAdd cosmetics")
                A=self.addCosmetics(A)

            if self.ADU:
                # if self.config['verbose'] == 'True': print ("electrons2adu")
                print("\telectrons2adu")
                A=self.electrons2ADU(A)

            dF = A

            if k == 1:
                F[k-1] = dF
            else :
                F[k-1]=F[k-2]+dF


        '''plt.figure('rampe av offset')
        # plt.plot(np.median(F,axis=(1,2)))
        plt.plot(F[:, 1000, 1000])
        plt.plot(F[:, 1200, 1200])
        plt.plot(F[:, 2000, 1400])
        plt.show()'''

        satu = fits.open(self.path + '/data/' + self.information['SaturationFile'])
        satu = np.ndarray.flatten(satu[1].data)


        for k in range(0,len(nf)):

            if self.readoutNoise:
                # if self.config['verbose'] == 'True': print ("Add Readout Noise")
                print("\tAdd Readout Noise")
                F[k]= self.applyReadoutNoise(F[k])

            if self.Offset:
                # if self.config['verbose'] == 'True': print ("Add offset")
                # Nframe = int(np.round(self.information['Nframe']))
                # if Nframe == 1:
                print("\tAdd offset")
                F[k]=self.addOffset(F[k])

            if self.CreatePersistance:
                print("\tCreate Persistance")
                self.applyCreatePersistance(F[k],k)

        for k in range(0, len(nf)):

            if self.saturation:
                # if self.config['verbose'] == 'True': print ("Apply Saturation")
                print("\tApply Saturation")
                F[k] =self.applySaturation(F[k])

            plt.figure('rampe')
            # plt.plot(np.median(F,axis=(1,2)))
            plt.plot(F[:, 1000, 1000])
            plt.plot(F[:, 1200, 1200])
            plt.plot(F[:, 2000, 1400])
            #plt.plot(F[:, 735, 267])
            plt.plot(F[:, 736, 267])
            #plt.plot(np.repeat(satu[267+2048* 735],len(nf)),'--')
            plt.plot(np.repeat(satu[1507595],len(nf)),'--')
            plt.show()



        plt.figure('image')
        plt.imshow(F[len(nf)-2], vmin=np.quantile(F[len(nf)-2], 0.01), vmax=np.quantile(F[5], 0.99))
        plt.colorbar()

        plt.figure('rampe')
        #plt.plot(np.median(F,axis=(1,2)))
        plt.plot(F[:,1000,1000])
        plt.plot(F[:, 1200, 1200])
        plt.plot(F[:, 2000, 1400])
        plt.plot(F[:,736 , 267])
        plt.show()

        # if self.config['verbose'] == 'True': print ("Write outputs")
        print("\tWrite outputs")
        self.writeOutputs(F)


if __name__ == '__main__':
    # run the simulator
    IS = ImageSimulator2()
    IS.simulate()
