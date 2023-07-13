import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.special import jn
from scipy.integrate import quad
from astropy.io import fits
from scipy.signal import fftconvolve as convolution
from math import ceil


def I_airy(u,epsilon):
    #Intensity of Airy pattern
    #I_airy = 1/(1-epsilon**2)**2*( 2*jn(1,u)/u - epsilon*2*jn(1,epsilon*u)/u )**2
    if epsilon == 0: I_airy = ( 2*jn(1,(u+1e-10))/(u+1e-10))**2
    else: I_airy = 1/(1-epsilon**2)**2*( 2*jn(1,(u+1e-10))/(u+1e-10) - epsilon*2*jn(1,epsilon*(u+1e-10))/(u+1e-10) )**2
    return I_airy

def I_gauss(u,sigma):

    I_gauss = np.exp(-u**2/(2*sigma**2))
    return I_gauss

def I_moffat(u,fwhm,beta=2):
    alpha=fwhm/(2*np.sqrt(2**(1/beta) -1))
    I_moffat=(1+(u/alpha)**2)**(-beta)
    return I_moffat

def plotPSF_EE():

    #set radius
    r_norm = np.linspace(0.1,20,1000)

    #fractional obscuration
    obs=DM2/DM1

    band_vis=['g','r','i','z','y']
    band_nir=['J','H']

    #Wavelength
    eff_wvl={'g':480,'r':620.5,'i':760,'z':867.4,'y':1002.4,'J':1254,'H':1636.8}  #nm
    wvl= eff_wvl[band]   #nm
    wvl*=1e-9    #nm-->m

    #Pixel size
    if band in band_vis:
       pixel_size=15   #micrometers
       pixel_scale=0.38    #arcsecond/pixel
       N=6.2   #f-number visible
    elif band in band_nir:
       pixel_size=18   #micrometers
       pixel_scale=0.76    #arcsecond/pixel
       N=3.7   # f-number nir

    #focal length
    f=N*DM1

    #Normalised radius
    r_meters = r_norm * wvl * N / np.pi
    r_micrometers =r_meters*1e6

    #Fitting airy with gaussian
    #Same I0: sigma=0.42 * lambda * N
    #Same volume: sigma = 0.45 * lambda * N
    sigma_gaussian_atm= seeing/pixel_scale*pixel_size/(2*np.sqrt(2*np.log(2)))
    sigma_gaussian_inst=0.45*wvl*N *1e6

    seeing_fwhm = seeing/pixel_scale*pixel_size
    print ('seeing FWHM:  %.2f"/ %.2f microns' % (seeing,seeing_fwhm))
    #Encircled Energy
    #EE_airy = (1 - jn(0,x)**2 -jn(1,x)**2 )

    #integration
    _integrand_airy = lambda x : 2*np.pi*I_airy(x,obs)*x
    E0_airy = quad(_integrand_airy,0,np.inf)[0]
    EE_airy=[]
    for r in r_norm:
        EE_airy.append(quad(_integrand_airy,0,r)[0])
    EE_airy=np.array(EE_airy)/E0_airy

    _integrand_gauss_inst = lambda x : 2*np.pi*I_gauss(x,obs,sigma_gaussian_inst)*x
    E0_gauss_inst = quad(_integrand_gauss_inst,0,np.inf)[0]
    print ('E0_gauss_inst: %f  sigma: %f   %f' % (E0_gauss_inst,sigma_gaussian_inst,E0_gauss_inst/(sigma_gaussian_inst**2)))
    EE_gauss_inst=[]
    for r in r_micrometers:
        EE_gauss_inst.append(quad(_integrand_gauss_inst,0,r)[0])
    EE_gauss_inst=np.array(EE_gauss_inst)/E0_gauss_inst

    _integrand_gauss_atm = lambda x : 2*np.pi*I_gauss(x,obs,sigma_gaussian_atm)*x
    E0_gauss_atm = quad(_integrand_gauss_atm,0,np.inf)[0]
    print ('E0_gauss_atm: %f sigma: %f   %f' % (E0_gauss_atm,sigma_gaussian_atm,E0_gauss_atm/(sigma_gaussian_atm**2)))

    EE_gauss_atm=[]
    for r in r_micrometers:
        EE_gauss_atm.append(quad(_integrand_gauss_atm,0,r)[0])
    EE_gauss_atm=np.array(EE_gauss_atm)/E0_gauss_atm

    _integrand_moffat = lambda x : 2*np.pi*I_moffat(x,seeing_fwhm)*x
    E0_moffat = quad(_integrand_moffat,0,np.inf)[0]
    EE_moffat=[]
    for r in r_micrometers:
        EE_moffat.append(quad(_integrand_moffat,0,r)[0])
    EE_moffat=np.array(EE_moffat)/E0_moffat

    #Find relation between d80 and FWHM
    from scipy.interpolate import interp1d
    f = interp1d(I_airy(r_norm,obs),r_micrometers,kind='linear')
    f2 = interp1d(EE_airy,r_micrometers,kind='linear')
    r_FWHM = f(0.5)
    r_d80 = f2(0.8)
    print ('r_FWHM: %f    r_d80: %f    ratio: %f' % (r_FWHM,r_d80,r_FWHM/r_d80))
    plt.plot(r_micrometers,I_airy(r_norm,obs),label='I_airy',color='red',lw=1.5)
    #plt.plot(r_micrometers/pixel_size_vis,I_airy(r_norm,obs),label='I_airy_norm',color='red',lw=1.5,ls='--')
    #plt.plot(radius,I_gauss(radius,0),label='I_gauss',color='red',lw=1.5,ls='--')
    #plt.plot(r_norm,EE_airy,label='EE_airy_norm',color='blue',lw=1.5,ls='--')
    plt.plot(r_micrometers,I_gauss(r_micrometers,obs,sigma_gaussian_inst),label='I_gauss',color='blue',lw=1.5)
    plt.plot(r_micrometers,I_moffat(r_micrometers,seeing_fwhm),label='I_moffat',color='green',lw=1.5)
    plt.plot(r_micrometers,I_gauss(r_micrometers,obs,sigma_gaussian_atm),label='I_gauss_atm_0.1"',color='black',lw=1.5)

    plt.plot(r_micrometers,EE_airy,label='EE_airy',color='red',lw=1.5,ls='--')
    plt.plot(r_micrometers,EE_gauss_inst,label='EE_gauss',color='blue',lw=1.5,ls='--')
    plt.plot(r_micrometers,EE_moffat,label='EE_moffat',color='green',lw=1.5,ls='--')
    plt.plot(r_micrometers,EE_gauss_atm,label='EE_gauss_atm',color='black',lw=1.5,ls='--')
    plt.ylabel('Intensity  /  EE')
    plt.xlabel(r'radius ($\mu$m)')
    #plt.xlim(0,10)
    plt.ylim(0,1.1)
    plt.grid(True)
    plt.minorticks_on()
    plt.legend(loc='lower right')
    plt.savefig('I_EE.png')
    if disp: plt.show()



def createPSF(filename='moffat_1024_J_s50.fits',PSF_type='moffat',imsize=[1025,1025],pixel_size=15,seeing=5,DM1=1.3,DM2=0.58,eff_wvl=7600,pixel_scale=0.38,focal_length=8.124,beta=2,oversamp=1,oversamp2=15,disp=True,unsigned16bit=False):

    # Create directory if not existing
    if os.path.dirname(filename) != '': os.makedirs(os.path.dirname(filename),exist_ok=True)

    #set radius
    r_norm = np.linspace(0.1,20,1000)

    #fractional obscuration
    obs=DM2/DM1
    #print (eff_wvl)
    eff_wvl*=1e-10    #ang-->m

    #pixel_size in micrometers
    #pixel_scale in arcsecond/pixel
    N= focal_length/DM1 #f-number visible

    #Normalised radius
    r_meters = r_norm * eff_wvl * N / np.pi
    r_micrometers =r_meters*1e6
   
    #print (eff_wvl,N,pixel_size,PSF_type,oversamp,seeing,obs) 
    #Fitting airy with gaussian
    #Same I0: sigma=0.42 * lambda * N
    #Same volume: sigma = 0.45 * lambda * N
    sigma_gaussian_atm= seeing/pixel_scale*pixel_size[0]/(2*np.sqrt(2*np.log(2)))
    #sigma_gaussian_inst=0.45*wvl*N *1e6

    seeing_fwhm = seeing/pixel_scale*pixel_size[0]
    #print ('seeing FWHM:  %.2f"/ %.2f microns / %.2f pixels ' % (seeing,seeing_fwhm*1e6,seeing_fwhm/pixel_size[0])) 

    #Create PSF
    X,Y=np.meshgrid(range(imsize[0]*oversamp2),range(imsize[1]*oversamp2))
    X0=(imsize[0]*oversamp2)/2-np.ceil(oversamp2/2)
    Y0=(imsize[1]*oversamp2)/2-np.ceil(oversamp2/2)   # -np.ceil(oversamp2/2) to align the center of the PSF on the 9x9 subpixel grid
    #print (X0,Y0)
    #print (PSF_type,eff_wvl,N,obs,DM1,DM2,pixel_size,oversamp,X0,Y0,oversamp2,focal_length)
    #obs=0
    #pixel_size=0.383e-6
    r=np.sqrt((X-X0)**2+(Y-Y0)**2)*pixel_size[0]/oversamp/oversamp2
    if PSF_type == 'moffat': Z=I_moffat(r,seeing_fwhm)
    elif PSF_type == 'airy': Z=I_airy(r*np.pi/(eff_wvl * N ),obs)
    elif PSF_type == 'gaussian': Z=I_gauss(r,sigma_gaussian_atm)

    Z2=_resize_sum(Z,oversamp2)

    if disp:
        plt.figure() 
        plt.imshow(Z2/Z2.sum())
        plt.colorbar()
        plt.show()

    mask = Z2 < 0
    Z2[mask] = 0

    #create a new FITS file
    hdu = fits.PrimaryHDU()
    #new image HDU
    hdu.data=Z2/np.sum(Z2)

    #convert to unsigned 16bit int if requested
    if unsigned16bit:
        hdu.scale('int16', '', bzero=32768)
        hdu.header.add_history('Scaled to unsigned 16bit integer!')
    #Add headers from image1 and 2

    #write the info to the header
    hdu.header.add_comment('This PSF is the convolution of a ZEMAX file with simulated atm PSF')
    hdu.verify('fix')
    hdu.writeto(filename,overwrite=True)


def _resize_sum(model,factor):
    """ Sum pixel to get the real size """

    im1=model.shape
    
    # Size of new image
    new_image=np.zeros((int(im1[0]/factor),int(im1[1]/factor)))
    #print (new_image.shape)
    
    ix=0
    for x in range(new_image.shape[0]):
         iy=0
         for y in range(new_image.shape[1]):
               new_image[x,y]=np.sum(model[x+ix*(factor-1):x+(ix+1)*(factor-1)+1,y+iy*(factor-1):y+(iy+1)*(factor-1)+1])
               iy+=1
         ix+=1

    return new_image

def resize(filename1='zemax_moffat_1024_J_ideal_s30.fits',filename2='zemax_moffat_128_J_ideal_s30.fits',type='sum',factor_type='factor',ImageSizeNew=[128,128],resampling=2,overwrite=True,unsigned16bit=False):
   """ Sum pixel to get the real size """

   # Create directory if not existing
   os.makedirs(os.path.dirname(filename2),exist_ok=True)


   data,header=fits.getdata(filename1,header=True)
   
   """ 
   # Sum 
   new_image=np.zeros(ImageSizeReal)
   ix=0
   for x in range(ImageSizeReal[0]):
       iy=0
       for y in range(ImageSizeReal[1]):
           new_image[x,y]=np.sum(data[x+ix*(factor-1):x+(ix+1)*(factor-1)+1,y+iy*(factor-1):y+(iy+1)*(factor-1)+1])
           iy+=1
       ix+=1
   """
   if factor_type == 'factor': factor = resampling
   elif factor_type == 'size': factor=[ImageSizeNew[0]/data.shape[0],ImageSizeNew[1]/data.shape[1]]

   if type == 'sum':
       new_image=_resize_sum(data,factor)

   elif type=='zoom':
       new_image = ndimage.zoom(data,factor,order=3)
       #Renormalisation
       new_image/=np.sum(new_image)

   #create a new FITS file
   hdu=fits.PrimaryHDU()

   #new image HDU
   hdu.data=new_image

   #convert to unsigned 16bit int if requested
   if unsigned16bit:
       hdu.scale('int16', '', bzero=32768)
       hdu.header.add_history('Scaled to unsigned 16bit integer!')

   #Add headers from the two files
   for key, value in header.items():
       hdu.header.set(key.upper(), value)
   #hdu.header.set('PIXSIZE',header['PIXSIZE']/factor[0])
   hdu.verify('fix')

   hdu.writeto(filename2,overwrite=True)


def convolvePSF(filename1='moffat_1024_J_s30.fits',filename2='PSF_J_ideal.fits',filename3='zemax_moffat_1024_J_ideal_s30.fits',overwrite=True, unsigned16bit=False):
   """ Convolution of 2 images """

   # Create directory if not existing
   os.makedirs(os.path.dirname(filename3),exist_ok=True)


   data1,header1=fits.getdata(filename1,header=True)
   data2,header2=fits.getdata(filename2,header=True)

   #Check for normalisation
   if np.sum(data1) !=1: data1 /= np.sum(data1)
   if np.sum(data2) !=1: data2 /= np.sum(data2)

   """
   #Cnonvolution of the instrumental PSF with a gaussian representing the atmosphere
   try:
       #If convolution from scipy. Seems to be more robust than the one from astropy
       data3 = convolution(data1,data2,mode='same')
   except:
       #If convoltuion from astropy
       data3 = convolution(data1,data2,allow_huge=True)
   """


   data1 = np.array(data1, dtype=float)
   data2 = np.array(data2, dtype=float)

   data3 = convolution(data1,data2,mode='same')

   mask = data3 < 0
   data3[mask] = 0

   #Renormalisation
   data3 /= np.sum(data3)

   #create a new FITS file
   hdu = fits.PrimaryHDU()

   #new image HDU
   hdu.data=data3

   #convert to unsigned 16bit int if requested
   if unsigned16bit:
       hdu.scale('int16', '', bzero=32768)
       hdu.header.add_history('Scaled to unsigned 16bit integer!')

   #Add headers from image1 and 2
   for key, value in header1.items():
       hdu.header.set(key.upper(), value)
   for key, value in header2.items():
       hdu.header.set(key.upper(), value)

   #write the info to the header
   hdu.header.add_comment('This PSF is the convolution of a ZEMAX file with simulated atm PSF')

   hdu.verify('fix')

   hdu.writeto(filename3,overwrite=True)


def psf_degradation(filename1='PSF_J_ideal.fits',factor=2,filename2='PSF_J_ideal_z2.fits', overwrite=True, unsigned16bit=False):
   """ zoom in or out the input image 

       Assume same size horizontally and vertically
   """

   # Create directory if not existing
   os.makedirs(os.path.dirname(filename2),exist_ok=True)


   data,header=fits.getdata(filename1,header=True)

   image_size=data.shape

   #Assume same size horizontally and vertically
   if image_size[0] % 2 == 0:
       width=int(image_size[0]/2)
   else:
       width=int((image_size[0]-1)/2)

   #Assume same size horizontally and vertically
   if ceil(image_size[0]*factor) % 2 == 0:
       center=int((ceil(image_size[0]*factor))/2)
   else:
       center=int((ceil(image_size[0]*factor)-1)/2)
   center=[center,center]

   """
   center=np.unravel_index(data.argmax(), data.shape)
   center=np.asarray(center)
   center[0]=int(ceil(center[0]*factor))
   center[1]=int(ceil(center[1]*factor))
   """

   new_image = ndimage.zoom(data,factor,order=3)
   new_image_resized=new_image[center[0]-width:center[0]+width,center[1]-width:center[1]+width]

   #Renormalisation
   new_image_resized/=np.sum(new_image_resized)

   #create a new FITS file
   hdu = fits.PrimaryHDU()

   hdu.data=new_image_resized

   #convert to unsigned 16bit int if requested
   if unsigned16bit:
       hdu.scale('int16', '', bzero=32768)
       hdu.header.add_history('Scaled to unsigned 16bit integer!')


   #Add headers from the two files
   for key, value in header.items():
       hdu.header.set(key.upper(), value)

   hdu.verify('fix')

   hdu.writeto(filename2,overwrite=True)



def test_ndimage_zoom(image):
    """ test some functions of ndimage package """
    data = fits.getdata("%s.fits" % image)


    over2real_size(image,[128,128],8,"test_sum")
    image_undersampled=fits.getdata("test_sum.fits")
    factor=[128/data.shape[0],128/data.shape[1]]
    test = ndimage.zoom(data,factor,order=3)
    test2 = ndimage.zoom(data,factor,order=0)
    test3 = ndimage.zoom(data,factor,order=1)

    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax.flat[1].imshow(test/np.sum(test),interpolation='none')
    ax.flat[0].imshow(image_undersampled/np.sum(image_undersampled),interpolation='none')
    ax.flat[2].imshow(test2/np.sum(test2),interpolation='none')
    ax.flat[3].imshow(test3/np.sum(test3),interpolation='none')

    plt.show()


if  __name__ == '__main__':

    #generatePSF(directory='../data/psf/atmosphere',filename='moffat_1024_J_s30',PSF_type='moffat',imsize=[1025,1025],pixel_size=0.383,band='J',seeing=3,DM1=1.3,DM2=0.58,disp=True,unsigned16bit=False)
    #psf_degradation(directory1='../data/psf/zemax',filename1='PSF_J_ideal',factor=2,directory2='../data/psf/zemax',filename2='PSF_J_ideal_z2')
    convolvePSF(filename1='moffat_1024_J_s30.fits',filename2='PSF_J_ideal.fits',filename3='zemax_moffat_1024_J_ideal_s30.fits')
    resizePSF(filename1='zemax_moffat_1024_J_ideal_s30.fits',filename2='zemax_moffat_128_J_ideal_s30.fits',ImageSizeNew=[128,128])


