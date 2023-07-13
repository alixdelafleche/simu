"""
Generating Object Catalogue
===========================

"""
import numpy as np
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import math, os


def starCatalog(stars=400, xmax=2048, ymax=2066, magmin=23, magmax=26):
    """
    Generate a catalog with stars at random positions.
    """
    xcoords = np.random.random(stars) * xmax
    ycoords = np.random.random(stars) * ymax
    mags = np.linspace(magmin, magmax, stars)

    fh = open('starsFaint.dat', 'w')
    fh.write('#   1 X                Object position along x                                    [pixel]\n')
    fh.write('#   2 Y                Object position along y                                    [pixel]\n')
    fh.write('#   3 MAG              Object magnitude                                           [AB]\n')
    fh.write('#   4 TYPE             Object type                                                [0=star, others=FITS]\n')
    fh.write('#   5 ORIENTATION      Objects orientation                                        [deg]\n')

    for x, y, m in zip(xcoords, ycoords, mags):
        fh.write('%f %f %f %i %f \n' % (x, y, m, 0, 0.0))
    fh.close()


def starCatalogFixedMagnitude(stars=400, xmax=2048, ymax=2066, mag=18, random=True, pergrid=51, out='starsSameMag.dat'):
    """
    Generate a catalog with stars of a given magnitude either at random positions or in a rectangular grid.
    """
    fh = open(out, 'w')
    fh.write('#   1 X                Object position along x                                    [pixel]\n')
    fh.write('#   2 Y                Object position along y                                    [pixel]\n')
    fh.write('#   3 MAG              Object magnitude                                           [AB]\n')
    fh.write('#   4 TYPE             Object type                                                [0=star, others=FITS]\n')
    fh.write('#   5 ORIENTATION      Objects orientation                                        [deg]\n')

    if random:
        #random positions
        xcoords = np.random.random(stars) * xmax
        ycoords = np.random.random(stars) * ymax

        for x, y in zip(xcoords, ycoords):
            fh.write('%f %f %f %i %f \n' % (x, y, mag, 0, 0.0))
    else:
        #grid
        xcoords = np.linspace(30, xmax-30, pergrid)
        ycoords = np.linspace(30, ymax-30, pergrid)
        for x in xcoords:
            for y in ycoords:
                fh.write('%f %f %f %i %f \n' % (x, y, mag, 0, 0.0))

    fh.close()

def SED_calibration(catalog):
   """ spectrum to use for filling mising bands in catalos """
   # Standard spectrum
   wvl_eff=np.array([3600,4300,5500,7000,9000,12600,16000,22201])  # Angstroms
   flux=np.array([15.3,59.1,122,186,254,206,214,143])*1e-6     # Jy

   if catalog == "NOMAD-1":
       # Define effective wavelength and F0 flux (for which m=0)
       catalog_wvl_eff=np.array([0.43,0.55,0.7,1.26,1.60,2.22])*1e4  # microns --> angstroms
       catalog_F0=np.array([4130,3781,2941,1594,1024,666.7])     # Jy
   elif catalog == "USNO-A2":
       # Define effective wavelength and F0 flux (for which m=0)
       catalog_wvl_eff=np.array([0.43,0.7])*1e4  # microns --> angstroms
       catalog_F0=np.array([4130,2941])     # Jy
   elif catalog == "II/246":
       # 2MASS catalog
       # Define effective wavelength and F0 flux (for which m=0)
       catalog_wvl_eff=np.array([1.26,1.60,2.22])*1e4  # microns --> angstroms
       catalog_F0=np.array([1594,1024,666.7])     # Jy
   else:
       catalog_wvl_eff=None
       catalog_F0 = None
   return wvl_eff, flux,catalog_wvl_eff,catalog_F0



def Viziercatalog(RA,DEC,radius,band,wvl_eff,header,catalog="NOMAD-1",frame="icrs",output='SourcesCatalog.txt',extrapolate=False):
   """ generate stars catalogue using Vizier """
   from astroquery.vizier import Vizier
   import astropy.units as u
   import astropy.coordinates as coord
   from astropy.wcs import WCS
   from scipy.interpolate import interp1d
   import scipy.optimize as op

   # Create directory if not existing
   os.makedirs(os.path.dirname(output),exist_ok=True)

   # Standard spectrum
   calib_wvl,calib_flux,catalog_wvl,catalog_F0=SED_calibration(catalog)
   calib_func = interp1d(calib_wvl,calib_flux,kind='linear')

   if catalog == 'NOMAD-1':
       v = Vizier(columns=["all", "+_r"], catalog=catalog,column_filters={"r_Bmag":"!=o","r_Rmag":"!=e"})
   else: v = Vizier(columns=["all", "+_r"], catalog=catalog)
   #v = Vizier(columns=['_RAJ2000', '_DEJ2000','B-V', 'Vmag', 'Plx'],
   #           column_filters={"Vmag":">10"}, keywords=["optical", "xry"])
   v.ROW_LIMIT =-1
   #result = v.query_region(coord.SkyCoord(ra=RA, dec=DEC, unit=(u.deg, u.deg), frame=frame),radius=radius*u.arcmin)
   result = v.query_region(coord.SkyCoord(ra=RA, dec=DEC, unit=(u.deg, u.deg), frame=frame),width=[radius*u.arcmin,radius*u.arcmin])
   #print (result)
   c = coord.SkyCoord(result[0]['RAJ2000'], result[0]['DEJ2000'], unit=(u.deg, u.deg),frame=frame)
   w = WCS(header)
   world = np.array([c.ra.deg, c.dec.deg]).T
   #print(world)
   #world = np.array([[RA,DEC ]])

   pix = w.all_world2pix(world,1) # Pixel coordinates of (RA, DEC)
   #print ("Pixel Coordinates: ", pix[0,0], pix[0,1])

   if extrapolate:
       mags=[]
       if catalog == 'NOMAD-1':
           # Calculate magnitude in the required band
           for line in result[0]['Bmag','Vmag','Rmag','Jmag','Hmag','Kmag']:
               B=line['Bmag']
               V=line['Vmag']
               R=line['Rmag']
               J=line['Jmag']
               H=line['Hmag']
               K=line['Kmag']

               input_mags=np.array([B,V,R,J,H,K])

               #Convert to Jy
               fluxes=catalog_F0*10**(-0.4*input_mags)

               mask=np.isnan(fluxes)
               #If some magnitudes are missing do interpolation with calibration spectrum
               interp1=interp1d(catalog_wvl,fluxes)
               if 1==0:
                   new_flux=interp1(wvl_eff)
               else:
                   #interpolation of the effective wvl of the catalogue
                   calib_flux_interp=calib_func(catalog_wvl)
                   f = lambda p, x, y: (p*y - x)**2

               scale,err = op.leastsq(f, [0.1], args=(fluxes[np.isfinite(fluxes)],calib_flux_interp[np.isfinite(fluxes)]))
               # Scale the missing fluxes
               new_flux=calib_func(wvl_eff)*scale[0]

               #Convert fluxes to AB magnitudes
               mags.append(-2.5*np.log10(new_flux/3631))
       elif catalog == 'USNO-A2':
           # Calculate magnitude in the required band
           for line in result[0]['Bmag','Rmag']:
               B=line['Bmag']
               R=line['Rmag']

               input_mags=np.array([B,R])

               #Convert to Jy
               fluxes=catalog_F0*10**(-0.4*input_mags)

               mask=np.isnan(fluxes)
               #If some magnitudes are missing do interpolation with calibration spectrum
               interp1=interp1d(catalog_wvl,fluxes)
               if 1==0:
                   new_flux=interp1(wvl_eff)
               else:
                   #interpolation of the effective wvl of the catalogue
                   calib_flux_interp=calib_func(catalog_wvl)
                   f = lambda p, x, y: (p*y - x)**2

               scale,err = op.leastsq(f, [0.1], args=(fluxes[np.isfinite(fluxes)],calib_flux_interp[np.isfinite(fluxes)]))
               # Scale the missing fluxes
               new_flux=calib_func(wvl_eff)*scale[0]

               #Convert fluxes to AB magnitudes
               mags.append(-2.5*np.log10(new_flux/3631))

   else:
       if catalog == 'NOMAD-1':
           if band == 'B': flux=catalog_F0[0]*10**(-0.4*result[0]['Bmag'])
           elif band == 'V': flux=catalog_F0[1]*10**(-0.4*result[0]['Vmag'])
           elif band == 'R' or 'r': flux=catalog_F0[2]*10**(-0.4*result[0]['Rmag'])
           elif band == 'J': flux=catalog_F0[3]*10**(-0.4*result[0]['Jmag'])
           elif band == 'H': flux=catalog_F0[4]*10**(-0.4*result[0]['Hmag'])
           elif band == 'K': flux=catalog_F0[5]*10**(-0.4*result[0]['Kmag'])
       elif catalog == 'USNO-A2':
           if band == 'B': flux=catalog_F0[0]*10**(-0.4*result[0]['Bmag'])
           elif band == 'R' or 'r': flux=catalog_F0[1]*10**(-0.4*result[0]['Rmag'])
       elif catalog == 'II/246':
           if band == 'J': flux=catalog_F0[0]*10**(-0.4*result[0]['Jmag'])
           elif band == 'H': flux=catalog_F0[1]*10**(-0.4*result[0]['Hmag'])
           elif band == 'K': flux=catalog_F0[2]*10**(-0.4*result[0]['Kmag'])
       mags = -2.5*np.log10(flux/3631)   # AB

   fh = open(output, 'w')
   fh.write('#   1 X                Object position along x                                    [pixel]\n')
   fh.write('#   2 Y                Object position along y                                    [pixel]\n')
   fh.write('#   3 MAG              Object magnitude                                           [AB]\n')
   fh.write('#   4 TYPE             Object type                                                [0=star, others=FITS]\n')
   fh.write('#   5 ORIENTATION      Objects orientation                                        [deg]\n')


   for i in range(len(pix[:,0])):
       if np.isfinite(mags[i]):
           #fh.write('%f %f %f %i %f \n' % (world[i,0], world[i,1], mags[i], 0, 0.0))
           fh.write('%f %f %f %i %f \n' % (pix[i,0], pix[i,1], mags[i], 0, 0.0))

def PanstarrsCatalog(RA, DEC, rad_deg, band, wvl_eff, header,mindet=1, maxsources = 50000, frame="icrs", output='SourcesCatalog.txt', extrapolate=False, server=('https://archive.stsci.edu/panstarrs/search.php')):
    """
    Query Pan-STARRS DR1 @ MAST
    parameters: RA, DEC, rad_deg: RA, Dec, field
                                          radius in degrees
                mindet: minimum number of detection (optional)
                maxsources: maximum number of sources
                server: servername
    returns: astropy.table object
    """
    import requests
    from astropy.io.votable import parse_single_table
    import astropy.units as u
    import astropy.coordinates as coord
    from astropy.wcs import WCS

    # Create directory if not existing
    os.makedirs(os.path.dirname(output),exist_ok=True)

    r = requests.get(server,
    params= {'RA': RA, 'DEC': DEC,
             'SR': rad_deg, 'max_records': maxsources,
             'outputformat': 'VOTable',
             'ndetections': ('>%d' % mindet)})

    # write query data into local file
    outf = open('panstarrs.xml', 'w')
    outf.write(r.text)
    outf.close()

    # parse local file into astropy.table object
    data = parse_single_table('panstarrs.xml')

    data_table=data.to_table(use_names_over_ids=True)

    mags = data_table['%sMeanApMag' % band]

    c = coord.SkyCoord(data_table['raMean'], data_table['decMean'], unit=(u.deg, u.deg),frame=frame)
    w = WCS(header)
    world = np.array([c.ra.deg, c.dec.deg]).T
    #world = np.array([[RA,DEC ]])

    pix = w.all_world2pix(world,1) # Pixel coordinates of (RA, DEC)
    #print ("Pixel Coordinates: ", pix[0,0], pix[0,1])

    fh = open(output, 'w')
    fh.write('#   1 X                Object position along x                                    [pixel]\n')
    fh.write('#   2 Y                Object position along y                                    [pixel]\n')
    fh.write('#   3 MAG              Object magnitude                                           [AB]\n')
    fh.write('#   4 TYPE             Object type                                                [0=star, others=FITS]\n')
    fh.write('#   5 ORIENTATION      Objects orientation                                        [deg]\n')


    for i in range(len(pix[:,0])):
        if (np.isfinite(mags[i])) & (mags[i] > 0):
            #fh.write('%f %f %f %i %f \n' % (world[i,0], world[i,1], mags[i], 0, 0.0))
            fh.write('%f %f %f %i %f \n' % (pix[i,0], pix[i,1], mags[i], 0, 0.0))


def CreateObject(RA, DEC, mag, type,header,radius=21.7,  output='SourcesAdded.txt'):
    """ Générer une source en donnant sa position et sa magnitude """
    from astroquery.vizier import Vizier
    import astropy.units as u
    import astropy.coordinates as coord
    from astropy.wcs import WCS
    from scipy.interpolate import interp1d
    import scipy.optimize as op

    # Create directory if not existing
    os.makedirs(os.path.dirname(output), exist_ok=True)

    w = WCS(header)
    world = np.array([[RA,DEC]])
    pix = w.all_world2pix(world, 1)  # Pixel coordinates of (RA, DEC)

    # print ("Pixel Coordinates: ", pix[0,0], pix[0,1])
    mag=[mag]

    fh = open(output, 'w')
    fh.write('#   1 X                Object position along x                                    [pixel]\n')
    fh.write('#   2 Y                Object position along y                                    [pixel]\n')
    fh.write('#   3 MAG              Object magnitude                                           [AB]\n')
    fh.write('#   4 TYPE             Object type                                                [0=star, 2= variable, others=FITS]\n')
    fh.write('#   5 ORIENTATION      Objects orientation                                        [deg]\n')

    for i in range(len(pix[:, 0])):
        fh.write('%f %f %f %i %f \n' % (pix[i, 0], pix[i, 1], mag[i], type, 0.0))


if __name__ == '__main__':

    starCatalog(stars=600, xmax=4096, ymax=4096, magmin=14, magmax=24)
