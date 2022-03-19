# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 14:19:13 2022

@author: Kevin
"""
#%% Funciones de utilities.py
"""
Funciones útiles para la manipulación de imágenes satelitales GOES-R, creado por Diego Souza
Referencia: https://github.com/diegormsouza/Geo-Sat-Python-Mar-2021/blob/e3984a3ddfba473d01e16d56755e1394265f9525/utilities.py
"""
def loadCPT(path):
    import numpy as np
    import colorsys
    
    try:
        f = open(path)
    except:
        print ("File ", path, "not found")
        return None

    lines = f.readlines()

    f.close()

    x = np.array([])
    r = np.array([])
    g = np.array([])
    b = np.array([])

    colorModel = 'RGB'

    for l in lines:
        ls = l.split()
        if l[0] == '#':
            if ls[-1] == 'HSV':
                colorModel = 'HSV'
                continue
            else:
                continue
        if ls[0] == 'B' or ls[0] == 'F' or ls[0] == 'N':
            pass
        else:
            x=np.append(x,float(ls[0]))
            r=np.append(r,float(ls[1]))
            g=np.append(g,float(ls[2]))
            b=np.append(b,float(ls[3]))
            xtemp = float(ls[4])
            rtemp = float(ls[5])
            gtemp = float(ls[6])
            btemp = float(ls[7])

        x=np.append(x,xtemp)
        r=np.append(r,rtemp)
        g=np.append(g,gtemp)
        b=np.append(b,btemp)

    if colorModel == 'HSV':
        for i in range(r.shape[0]):
            rr, gg, bb = colorsys.hsv_to_rgb(r[i]/360.,g[i],b[i])
        r[i] = rr ; g[i] = gg ; b[i] = bb

    if colorModel == 'RGB':
        r = r/255.0
        g = g/255.0
        b = b/255.0

    xNorm = (x - x[0])/(x[-1] - x[0])

    red   = []
    blue  = []
    green = []

    for i in range(len(x)):
        red.append([xNorm[i],r[i],r[i]])
        green.append([xNorm[i],g[i],g[i]])
        blue.append([xNorm[i],b[i],b[i]])

    colorDict = {'red': red, 'green': green, 'blue': blue}

    return colorDict

# latlon2xy: (lon, lat) => (x, y)
# geo2grid: (lon, lat) => (x, y) => (col, row). Esto último es lo que se grafica
# convertExtent2GOESProjection: list(lon, lat) => list(x, y)
# https://www.linz.govt.nz/data/geodetic-system/coordinate-conversion/geodetic-datum-conversions/equations-used-datum

def latlon2xy(lat, lon):
    import numpy as np
    # goes_imagery_projection:semi_major_axis
    req = 6378137 # meters
    #  goes_imagery_projection:inverse_flattening
    # invf = 298.257222096
    # goes_imagery_projection:semi_minor_axis
    rpol = 6356752.31414 # meters
    e = 0.0818191910435
    # goes_imagery_projection:perspective_point_height + goes_imagery_projection:semi_major_axis
    H = 35786023 + req # 42164160 meters
    # goes_imagery_projection: longitude_of_projection_origin
    lambda0 = -1.308996939 # para -75°
    #lambda0 = -1.562069680534925 # para -89.5
    
    # Convert to radians
    latRad = lat * (np.pi/180)
    lonRad = lon * (np.pi/180)

    # (1) geocentric latitude
    Phi_c = np.arctan(((rpol * rpol)/(req * req)) * np.tan(latRad))
    # (2) geocentric distance to the point on the ellipsoid
    rc = rpol/(np.sqrt(1 - ((e * e) * (np.cos(Phi_c) * np.cos(Phi_c)))))
    # (3) sx
    sx = H - (rc * np.cos(Phi_c) * np.cos(lonRad - lambda0))
    # (4) sy
    sy = -rc * np.cos(Phi_c) * np.sin(lonRad - lambda0)
    # (5)
    sz = rc * np.sin(Phi_c)

    # x,y
    x = np.arcsin((-sy)/np.sqrt((sx*sx) + (sy*sy) + (sz*sz)))
    y = np.arctan(sz/sx)

    return x, y

def geo2grid(lat, lon, nc):
    # Apply scale and offset 
    xscale, xoffset = nc.variables['x'].scale_factor, nc.variables['x'].add_offset
    yscale, yoffset = nc.variables['y'].scale_factor, nc.variables['y'].add_offset
    
    x, y = latlon2xy(lat, lon)
    col = (x - xoffset)/xscale
    lin = (y - yoffset)/yscale
    return int(lin), int(col)

def convertExtent2GOESProjection(extent):
    # GOES-16 viewing point (satellite position) height above the earth
    GOES16_HEIGHT = 35786023.0
    # GOES-16 longitude position
    #GOES16_LONGITUDE = -75.0
	
    a, b = latlon2xy(extent[1], extent[0])
    c, d = latlon2xy(extent[3], extent[2])
    return (a * GOES16_HEIGHT, c * GOES16_HEIGHT, b * GOES16_HEIGHT, d * GOES16_HEIGHT)

#%% Funciones

def Palette_goes(i):
    import numpy as np
    from matplotlib import cm
    
    if i == 1:
        minmax = [-80., 40.]
        
        # Paleta principal
        gray_cmap = cm.get_cmap('gray_r', 120)
        gray_cmap = gray_cmap(np.linspace(0,1,120))
        
        # Paleta secundaria
        jet_cmap = cm.get_cmap('jet_r', 40)
        jet_cmap = jet_cmap(np.linspace(0,1,40))
        
        # Insertar paleta secundaria en principal
        gray_cmap[:40, :] = jet_cmap
        
        cmap = cm.colors.ListedColormap(gray_cmap)
        
    elif i == 2:
        minmax = [-80., 40.]

        gray_cmap = cm.get_cmap('gray_r', 120)
        gray_cmap = gray_cmap(np.linspace(0,1,120))
        
        colors = ['#ffa0ff', '#0806ff', '#3bcfff', '#feff65', '#ff7516']
        my_colors = cm.colors.ListedColormap(colors)
        my_colors = my_colors(np.linspace(0,1,50))

        gray_cmap[:50, :] = my_colors
        
        cmap = cm.colors.ListedColormap(gray_cmap)

    elif i == 3:
        minmax = [-80., 40.]
        
        colors = ['#bc8462', '#ae656f', '#a44a79', '#962e97',
                  '#6158c5', '#2b8ffb', '#5fcdff', '#94fff0',
                  '#a5ff94', '#fff88c', '#ffbf52', '#ec7b27',
                  '#b84827', '#a1333d', '#bd5478', '#cc6a99', '#d982b8']

        cmap = cm.colors.LinearSegmentedColormap.from_list('', colors)
    
    elif i == 4:
        minmax = [-103., 84.]
        
        colors = loadCPT(r'C:\Users\Kevin\Desktop\UNALM\OctCiclo\Senso_rem\RS_GOES16\IR4AVHRR6.cpt')
        cmap = cm.colors.LinearSegmentedColormap('cpt', colors)
    
    elif i == 5:
        minmax = [200, 320]
        
        colors = loadCPT(r'C:\Users\Kevin\Desktop\UNALM\OctCiclo\Senso_rem\RS_GOES16\temperature.cpt')
        cmap = cm.colors.LinearSegmentedColormap('cpt', colors)
    
    elif i == 6:
        minmax = [175, 375]
        
        colors = loadCPT(r'C:\Users\Kevin\Desktop\UNALM\OctCiclo\Senso_rem\RS_GOES16\WVCOLOR35.cpt')
        cmap = cm.colors.LinearSegmentedColormap('cpt', colors)
        
        print('Advertencia: solo aplicable para banda 9') 

    elif i == 'Channel_10':
        minmax = [-100, 100]
        
        colors = loadCPT(r'C:\Users\Kevin\Desktop\UNALM\OctCiclo\Senso_rem\RS_GOES16\WVCOLOR35.cpt')
        cmap = cm.colors.LinearSegmentedColormap('cpt', colors)
        
        print('Advertencia: solo aplicable para banda 10') 
    
    return [cmap, minmax]

def GetAbiDate(year, month, day, hour = 00, minut = 00, unique = True):
    date_list = []

    d = [year, month, day, hour, minut]
    a = [[n] if type(n)!=list else n for n in d]
  
    for i in a[0]:
      for j in a[1]:
        for k in a[2]:
          for l in a[3]:
            if unique == True:
                print('xd')
                #date = f'{i}{j:02}{k:02}{l:02}'
                #date_list.append(date)
            else:
                for m in a[4]:
                  date = f'{i}{j:02}{k:02}{l:02}{m:02}'
                  date_list.append(date)
    date_list.sort()
    return date_list

def GetAbiData(date, band, productName = 'ABI-L2-CMIPF', M = [3, 4]):
  import os, boto3
  from botocore import UNSIGNED
  from botocore.config import Config
  from datetime import datetime 

  dataLocalPath = './'; data = []; key_l = []
  
  # Repositorio de Amazon
  # https://noaa-goes16.s3.amazonaws.com/index.html

  # Deconstrucción de la fecha
  for i in date:
    yr = datetime.strptime(i,'%Y%m%d%H%M').strftime('%Y')
    jd = datetime.strptime(i,'%Y%m%d%H%M').strftime('%j')
    hr = datetime.strptime(i,'%Y%m%d%H%M').strftime('%H')
    mn = datetime.strptime(i,'%Y%m%d%H%M').strftime('%M')

    s3Client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    dataPath = f'{productName}/{yr}/{jd}/{hr}'

    exist = []
    for j in M:
      patterName = f'OR_{productName}-M{j}C{int(band):02.0f}_G16_s{yr}{jd}{hr}{mn}'
      prefix = f'{dataPath}/{patterName}'
      # Nota: se descarga una imagen cada 2 días porque he restringido la hora a 00 min
      # Busqueda en el servidor S3 para ambos modos (3 y 4)
      s3 = s3Client.list_objects_v2(Bucket='noaa-goes16', Prefix=prefix, Delimiter = "/")
      if 'Contents' not in s3:
        exist.append(False)
      
      else:
        for img in s3['Contents']:
            key = img['Key']; key_l.append(key)
            fileName = key.split('/')[-1].split('.')[0]
            data.append(f'{dataLocalPath}/{fileName}.nc')
    
    if sum(exist) == 0:
      data.append(None)

  # Descarga de archivo netCDF
  for k, l in enumerate(data):
    if l == None:
      print(f'[-] {date[k][0:8]}: no disponible')

    elif os.path.exists(l):
      print(f'[0] {date[k][0:8]}: archivo existente')
    
    else:
      s3Client.download_file('noaa-goes16', key_l[k], l)
      print(f'[+] {date[k][0:8]}: {l}')

def Fecha(date):
    months = ("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre")
    day = date.day
    month = months[date.month - 1]
    year = date.year
    messsage = "{} de {} del {}".format(day, month, year)

    return messsage

#%% Clases
# ====================================================================
# |_____Nombre_______|__________________Descripción__________________|
# |     GOES_BT      | Cloud and moisture imagery brightness temperature |
# |      Split       | Combinaciones de diferencias                  |
# |     RGB_CDP      | Cloud Day Phase                               |
# |    Air_mass      |   Masa de aire                                |
# |      Dust        |   Arena                                       |
# |      SO2         |   Dióxido de azufre                           |
# |Night_microphysics|  Microfísica nocturna                         |
# |  Day_land_cloud  |  Contraste nubes-superficie terrestre         |
# ====================================================================


class GOES_BT:
    def __init__(self, img, extent = None, Geostationary = True):
        from netCDF4 import Dataset
        
        self.file = Dataset(img); self.Geostationary = Geostationary
        
        if  extent != None:
            lly, llx = geo2grid(extent[1], extent[0], self.file)
            ury, urx = geo2grid(extent[3], extent[2], self.file)
            
            self.data = self.file.variables['CMI'][ury:lly, llx:urx] - 273.15
            self.img_extent = convertExtent2GOESProjection(extent)
            """
            if Geostationary == True:
                self.img_extent = convertExtent2GOESProjection(extent)
            else:
                self.img_extent = tuple(extent)
            """  
        else:
            self.data = self.file.variables['CMI'][:] - 273.15
            self.img_extent = (-5434894.67527,5434894.67527,
                               -5434894.67527,5434894.67527)
        
        self.geo_proj = self.file.variables['goes_imager_projection']
        self.var = self.file.variables
        
    def plot(self, cmap = 'Greys', color_e = '#ECF653', figsize=(12,12),
             font = 14, save = 'GOES_plot.jpg', min_max = [-35, 35], 
             title_p = 'GOES-16 Banda 13', axis = 'off', 
             lines = True, gridd = True, cb = True):
        
        import cartopy, cartopy.crs as ccrs, matplotlib.pyplot as plt
        from datetime import datetime, timedelta
        
        plt.figure(figsize = figsize)
    
        # Características de la proyección
        cl = self.geo_proj.longitude_of_projection_origin
        sh = self.geo_proj.perspective_point_height
        axis_maj = self.geo_proj.semi_major_axis
        axis_min = self.geo_proj.semi_minor_axis
        
        # Definir forma del globo
        globe = ccrs.Globe(semimajor_axis = axis_maj, semiminor_axis= axis_min)
        
        # Proyección primaria de GOES
        crs = ccrs.Geostationary(central_longitude = cl, satellite_height = sh, globe = globe)
        
        # Reproyección
        if self.Geostationary == False:
            proj = ccrs.PlateCarree(central_longitude = cl)
            
            # ax = plt.axes(projection = ccrs.PlateCarree(central_longitude = cl))
            # ax.set_extent(self.img_extent, crs = ccrs.PlateCarree(central_longitude = cl))
            
        else:
            #crs = ccrs.Geostationary(central_longitude = cl, satellite_height = sh, globe = globe)
            proj = crs
            # ax = plt.axes(projection = ccrs.Geostationary(central_longitude = cl, satellite_height = sh))
            
        ax = plt.axes(projection = proj)
        
        if lines == True:
            ax.coastlines(resolution='10m', color=color_e, linewidth=1) # Líneas de costa
            ax.add_feature(cartopy.feature.BORDERS,                     # Países
                           edgecolor = color_e, linewidth=0.5) 
        if gridd == True:
            ax.gridlines(color='white', alpha=0.5, linestyle='--', linewidth=0.5) # Grillado
        
        plt.axis(axis)
        print('xd')
        img = ax.imshow(self.data, cmap = cmap,           # Paleta de colores
                   vmin = min_max[0], vmax = min_max[1],  # Máximos y mínimos, extent = self.img_extent
                   transform = crs,
                   origin = 'upper', extent = self.img_extent)
        print('xd')
        if cb == True:
            plt.colorbar(img, orientation = 'vertical',  # Orientación
                         pad = 0.01,                     # Separación con la imagen
                         fraction = 0.04)                # Escala de la barra
    
        plt.title(title_p, # Título
                  fontweight='bold', fontsize=font,   # Formato de letra
                  loc='left')                         # Locación
    
        # Fecha de la imagen
        seg = int(self.file.variables['time_bounds'][0])
        date = datetime(2000,1,1,12) + timedelta(seconds=seg)
        #date = date.strftime('%d %B %Y %H:%M UTC')
        date = Fecha(date) + ' ' + date.strftime('%H:%M UTC')
        plt.title('Full Disk\n' + date, fontweight='bold', fontsize=font, loc='right')
        #plt.title('Sureste de Brasil\n' + date, fontweight='bold', fontsize=font, loc='right')        
        #plt.title('Perú\n' + date, fontweight='bold', fontsize=font, loc='right')
        # Oeste - cuenca amazónica
        plt.tight_layout()
        plt.savefig(save, dpi = 300)   # Guardado

class GOES_B2:
    def __init__(self, img, extent = None, Geostationary = True):
        from netCDF4 import Dataset
        
        self.file = Dataset(img); self.Geostationary = Geostationary
        
        if  extent != None:
            lly, llx = geo2grid(extent[1], extent[0], self.file)
            ury, urx = geo2grid(extent[3], extent[2], self.file)
            
            self.data = self.file.variables['CMI'][ury:lly, llx:urx]
            self.img_extent = convertExtent2GOESProjection(extent)
            """
            if Geostationary == True:
                self.img_extent = convertExtent2GOESProjection(extent)
            else:
                self.img_extent = tuple(extent)
            """  
        else:
            self.data = self.file.variables['CMI'][:]
            self.img_extent = (-5434894.67527,5434894.67527,
                               -5434894.67527,5434894.67527)
        
        self.geo_proj = self.file.variables['goes_imager_projection']
        self.var = self.file.variables
        
    def plot(self, cmap = 'Greys', color_e = '#ECF653', figsize=(12,12),
             font = 14, save = 'GOES_plot.jpg', min_max = [0., 1.], 
             title_p = 'GOES-16 Banda 13', axis = 'off', 
             lines = True, gridd = True, cb = True):
        
        import cartopy, cartopy.crs as ccrs, matplotlib.pyplot as plt
        from datetime import datetime, timedelta
        
        plt.figure(figsize = figsize)
    
        # Características de la proyección
        cl = self.geo_proj.longitude_of_projection_origin
        sh = self.geo_proj.perspective_point_height
        axis_maj = self.geo_proj.semi_major_axis
        axis_min = self.geo_proj.semi_minor_axis
        
        # Definir forma del globo
        globe = ccrs.Globe(semimajor_axis = axis_maj, semiminor_axis= axis_min)
        
        # Proyección primaria de GOES
        crs = ccrs.Geostationary(central_longitude = cl, satellite_height = sh, globe = globe)
        
        # Reproyección
        if self.Geostationary == False:
            proj = ccrs.PlateCarree(central_longitude = cl)
            
            # ax = plt.axes(projection = ccrs.PlateCarree(central_longitude = cl))
            # ax.set_extent(self.img_extent, crs = ccrs.PlateCarree(central_longitude = cl))
            
        else:
            #crs = ccrs.Geostationary(central_longitude = cl, satellite_height = sh, globe = globe)
            proj = crs
            # ax = plt.axes(projection = ccrs.Geostationary(central_longitude = cl, satellite_height = sh))
            
        ax = plt.axes(projection = proj)
        
        if lines == True:
            ax.coastlines(resolution='10m', color=color_e, linewidth=1) # Líneas de costa
            ax.add_feature(cartopy.feature.BORDERS,                     # Países
                           edgecolor = color_e, linewidth=0.5) 
        if gridd == True:
            ax.gridlines(color='white', alpha=0.5, linestyle='--', linewidth=0.5) # Grillado
        
        plt.axis(axis)

        img = ax.imshow(self.data, cmap = cmap,           # Paleta de colores
                   vmin = min_max[0], vmax = min_max[1],  # Máximos y mínimos, extent = self.img_extent
                   transform = crs,
                   origin = 'upper', extent = self.img_extent)

        if cb == True:
            plt.colorbar(img, orientation = 'vertical',  # Orientación
                         pad = 0.01,                     # Separación con la imagen
                         fraction = 0.04)                # Escala de la barra
    
        plt.title(title_p, # Título
                  fontweight='bold', fontsize=font,   # Formato de letra
                  loc='left')                         # Locación
    
        # Fecha de la imagen
        seg = int(self.file.variables['time_bounds'][0])
        date = datetime(2000,1,1,12) + timedelta(seconds=seg)
        #date = date.strftime('%d %B %Y %H:%M UTC')
        date = Fecha(date) + ' ' + date.strftime('%H:%M UTC')
        #plt.title('Oeste - cuenca amazónica\n' + date, fontweight='bold', fontsize=font, loc='right')
        plt.title('Sureste de Brasil\n' + date, fontweight='bold', fontsize=font, loc='right')
        # Full Disk
        plt.tight_layout()
        plt.savefig(save, dpi = 400)   # Guardado



class Raw_veggie:
    def __init__(self, img, extent = None):
        import numpy as np
        from netCDF4 import Dataset
        
        self.file = Dataset(img[0])
        self.geo_proj = self.file.variables['goes_imager_projection']
        self.var = self.file.variables

        if  extent != None:
            lly, llx = geo2grid(extent[1], extent[0], self.file)
            ury, urx = geo2grid(extent[3], extent[2], self.file)
            
            self.data1 = self.file.variables['CMI'][ury:lly, llx:urx][::4 ,::4]
            self.data2 = Dataset(img[1]).variables['CMI'][ury:lly, llx:urx][::8 ,::8]
            self.data3 = Dataset(img[2]).variables['CMI'][ury:lly, llx:urx][::4 ,::4]

            self.img_extent = convertExtent2GOESProjection(extent)

        else:
            self.data1 = self.file.variables['CMI'][::4 ,::4]
            self.data2 = Dataset(img[1]).variables['CMI'][:][::8 ,::8]
            self.data3 = Dataset(img[2]).variables['CMI'][::4 ,::4]
            
            self.img_extent = (-5434894.67527,5434894.67527,
                               -5434894.67527,5434894.67527)
        print(self.data1)
        def Normalize(band):
            amax, amin = np.max(band), np.min(band)
            return (band - amin)/(amax - amin)
            
        R = Normalize(self.data1); G = Normalize(self.data2)
        B = Normalize(self.data3)

        self.data = np.zeros((2712,2712,3))
        self.data[:,:,0] = R; self.data[:,:,1] = G; self.data[:,:,2] = B
        
    def plot(self, color_e = '#ECF653', figsize=(12,12), 
             font = 14, save = 'GOES_plot.jpg',
             title_p = 'GOES-16 Banda 13', axis = 'off', 
             lines = True, gridd = True, cb = True):
        
        import cartopy, cartopy.crs as ccrs, matplotlib.pyplot as plt
        from datetime import datetime, timedelta
        
        plt.figure(figsize = figsize)
    
        # Reproyección geoestacionaria
        cl = self.geo_proj.longitude_of_projection_origin
        sh = self.geo_proj.perspective_point_height
        
        ax = plt.axes(projection = ccrs.Geostationary(central_longitude = cl, 
                                                      satellite_height = sh))
        ax.set_extent(self.extent)
        if lines == True:
            ax.coastlines(resolution='10m', color=color_e, linewidth=1) # Líneas de costa
            ax.add_feature(cartopy.feature.BORDERS,                     # Países
                           edgecolor = color_e, linewidth=0.5) 
        if gridd == True:
            ax.gridlines(color='white', alpha=0.5, linestyle='--', linewidth=0.5) # Grillado
        
        plt.axis(axis)

        img = ax.imshow(self.data, origin = 'upper', extent = self.img_extent)
        
        if cb == True:
            plt.colorbar(img, orientation = 'vertical',  # Orientación
                         pad = 0.01,                     # Separación con la imagen
                         fraction = 0.04)                # Escala de la barra
    
        plt.title(title_p, # Título
                  fontweight='bold', fontsize=font,   # Formato de letra
                  loc='left')                         # Locación
    
        # Fecha de la imagen
        seg = int(self.file.variables['time_bounds'][0])
        date = datetime(2000,1,1,12) + timedelta(seconds=seg)
        date = date.strftime('%d %B %Y %H:%M UTC')
        plt.title('Full Disk\n' + date, fontweight='bold', fontsize=font, loc='right')
    
        plt.savefig(save, dpi = 300)   # Guardado

class Split:
    def __init__(self, img, split = 0, extent = None):
        from netCDF4 import Dataset
        
        self.file = Dataset(img[0])
        self.geo_proj = self.file.variables['goes_imager_projection']
        self.var = self.file.variables

        if  extent != None:
            lly, llx = geo2grid(extent[1], extent[0], self.file)
            ury, urx = geo2grid(extent[3], extent[2], self.file)
            
            self.data1 = self.file.variables['CMI'][ury:lly, llx:urx][::4 ,::4]
            self.data2 = Dataset(img[1]).variables['CMI'][ury:lly, llx:urx][::8 ,::8]
            
            self.img_extent = convertExtent2GOESProjection(extent)

        else:
            self.data1 = self.file.variables['CMI'][:]
            self.data2 = Dataset(img[1]).variables['CMI'][:]

            self.img_extent = (-5434894.67527,5434894.67527,
                               -5434894.67527,5434894.67527)
            
        self.data = self.data1 - self.data2
        
        if split == 0:
            self.config = ['Split windows', 'Spectral', -6, 6]
            print(self.config[0])
        
        elif split == 1:
            self.config = ['Split ozono', 'nipy_spectral', -45, 5]
            print(self.config[0])
        
        elif split == 2:
            self.config = ['Night fog', 'gray', -5, 5]
            print(self.config[0])
        
        elif split == 3:
            self.config = ['Split cloud face', 'jet', -10, 25]
            print(self.config[0])
        
    def plot(self, color_e = '#ECF653', figsize=(12,12), 
             font = 14, save = 'GOES_plot.jpg', 
             axis = 'off', lines = True, gridd = True, cb = True):
        
        import cartopy, cartopy.crs as ccrs, matplotlib.pyplot as plt
        from datetime import datetime, timedelta
        
        plt.figure(figsize = figsize)
    
        # Reproyección geoestacionaria
        cl = self.geo_proj.longitude_of_projection_origin
        sh = self.geo_proj.perspective_point_height
        
        ax = plt.axes(projection = ccrs.Geostationary(central_longitude = cl, 
                                                      satellite_height = sh))
        
        if lines == True:
            ax.coastlines(resolution='10m', color=color_e, linewidth=1) # Líneas de costa
            ax.add_feature(cartopy.feature.BORDERS,                     # Países
                           edgecolor = color_e, linewidth=0.5) 
        if gridd == True:
            ax.gridlines(color='white', alpha=0.5, linestyle='--', linewidth=0.5) # Grillado
        
        plt.axis(axis)

        img = ax.imshow(self.data, origin = 'upper', extent = self.img_extent,
                        cmap = self.config[1], 
                        vmin = self.config[2], vmax = self.config[3])
        
        if cb == True:
            plt.colorbar(img, orientation = 'vertical',  # Orientación
                         pad = 0.01,                     # Separación con la imagen
                         fraction = 0.04)                # Escala de la barra
            
        plt.title('GOES-16 Banda 13' + ' - ' + self.config[0], # Título
                  fontweight='bold', fontsize=font,   # Formato de letra
                  loc='left')                         # Locación
    
        # Fecha de la imagen
        seg = int(self.file.variables['time_bounds'][0])
        date = datetime(2000,1,1,12) + timedelta(seconds=seg)
        date = date.strftime('%d %B %Y %H:%M UTC')
        plt.title('Full Disk\n' + date, fontweight='bold', fontsize=font, loc='right')
    
        plt.savefig(save, dpi = 300)   # Guardado

class RGB_CDP:
    def __init__(self, img, extent = None):
        from netCDF4 import Dataset
        import numpy as np
        
        self.file = Dataset(img[0])
        self.geo_proj = self.file.variables['goes_imager_projection']
        self.var = self.file.variables

        if  extent != None:
            lly, llx = geo2grid(extent[1], extent[0], self.file)
            ury, urx = geo2grid(extent[3], extent[2], self.file)
            
            self.data1 = self.file.variables['CMI'][ury:lly, llx:urx][::8 ,::8]
            self.data2 = Dataset(img[1]).variables['CMI'][ury:lly, llx:urx][::4 ,::4]
            self.data3 = Dataset(img[2]).variables['CMI'][ury:lly, llx:urx][::2 ,::2]
            
            self.img_extent = convertExtent2GOESProjection(extent)

        else:
            self.data1 = self.file.variables['CMI'][:,:][::4 ,::4]         # Channel 2
            self.data2 = Dataset(img[1]).variables['CMI'][:,:][::2 ,::2]   # Channel 5
            self.data3 = Dataset(img[2]).variables['CMI'][:,:][::1 ,::1] - 273.15  # Channel 13
            
            self.img_extent = (-5434894.67527,5434894.67527,
                               -5434894.67527,5434894.67527)
        def function(band, parameters, R = False):
            A = np.clip(band, parameters[0], parameters[1])
            
            if R == True:
                A = ((A - parameters[1])/(parameters[0]- parameters[1]))**(1/parameters[2])
            else:
                A = ((A - parameters[0])/(parameters[1]- parameters[0]))**(1/parameters[2])
            return A
        
        R = function(self.data3, parameters = [-53.5, 7.5, 1], R = True)
        G = function(self.data1, parameters = [0.0, 0.78, 1])
        B = function(self.data2, parameters = [0.01, 0.59, 1])
        R.shape, G.shape, B.shape
        
        self.data = np.stack([R,G,B], axis = 2)
                
    def plot(self, color_e = '#ECF653', figsize=(12,12), 
             font = 14, save = 'GOES_plot.jpg',
             title_p = 'GOES-16 Banda 13 - Cloud day phase', axis = 'off', 
             lines = True, gridd = True, cb = True):
        
        import cartopy, cartopy.crs as ccrs, matplotlib.pyplot as plt
        from datetime import datetime, timedelta
        
        plt.figure(figsize = figsize)
    
        # Reproyección geoestacionaria
        cl = self.geo_proj.longitude_of_projection_origin
        sh = self.geo_proj.perspective_point_height
        
        ax = plt.axes(projection = ccrs.Geostationary(central_longitude = cl, 
                                                      satellite_height = sh))
        
        if lines == True:
            ax.coastlines(resolution='10m', color=color_e, linewidth=1) # Líneas de costa
            ax.add_feature(cartopy.feature.BORDERS,                     # Países
                           edgecolor = color_e, linewidth=0.5) 
        if gridd == True:
            ax.gridlines(color='white', alpha=0.5, linestyle='--', linewidth=0.5) # Grillado
        
        plt.axis(axis)

        img = ax.imshow(self.data, origin = 'upper', extent = self.img_extent)
        
        if cb == True:
            plt.colorbar(img, orientation = 'vertical',  # Orientación
                         pad = 0.01,                     # Separación con la imagen
                         fraction = 0.04)                # Escala de la barra
    
        plt.title(title_p, # Título
                  fontweight='bold', fontsize=font,   # Formato de letra
                  loc='left')                         # Locación
    
        # Fecha de la imagen
        seg = int(self.file.variables['time_bounds'][0])
        date = datetime(2000,1,1,12) + timedelta(seconds=seg)
        date = date.strftime('%d %B %Y %H:%M UTC')
        plt.title('Full Disk\n' + date, fontweight='bold', fontsize=font, loc='right')
    
        plt.savefig(save, dpi = 300)
        
class Air_mass:
    def __init__(self, img, extent = None):
        from netCDF4 import Dataset
        import numpy as np
        
        self.file = Dataset(img[0])
        self.geo_proj = self.file.variables['goes_imager_projection']
        self.var = self.file.variables

        if  extent != None:
            lly, llx = geo2grid(extent[1], extent[0], self.file)
            ury, urx = geo2grid(extent[3], extent[2], self.file)
            
            self.data1 = self.file.variables['CMI'][ury:lly, llx:urx][::4 ,::4] - 273.15
            self.data2 = Dataset(img[1]).variables['CMI'][ury:lly, llx:urx][::4 ,::4] - 273.15
            self.data3 = Dataset(img[2]).variables['CMI'][ury:lly, llx:urx][::4 ,::4] - 273.15
            self.data4 = Dataset(img[3]).variables['CMI'][ury:lly, llx:urx][::4 ,::4] - 273.15
            
            self.img_extent = convertExtent2GOESProjection(extent)

        else:
            self.data1 = self.file.variables['CMI'][:,:][::4 ,::4] - 273.15  # Channel 8
            self.data2 = Dataset(img[1]).variables['CMI'][:,:][::4 ,::4] - 273.15  # Channel 10
            self.data3 = Dataset(img[2]).variables['CMI'][:,:][::4 ,::4] - 273.15  # Channel 12
            self.data4 = Dataset(img[3]).variables['CMI'][:,:][::4 ,::4] - 273.15  # Channel 13
            self.img_extent = (-5434894.67527,5434894.67527,
                               -5434894.67527,5434894.67527)
            
        def function(band, parameters):
            A = np.clip(band, parameters[0], parameters[1])
            A = ((A - parameters[0])/(parameters[1]- parameters[0]))**(1/parameters[2])
            return A
        
        R = function(self.data1 - self.data2, parameters = [-26.2, 0.6, 1])
        G = function(self.data3 - self.data4, parameters = [-43.2, 6.7, 1])
        B = function(self.data1, parameters = [-64.65, -29.25, 1])

        self.data = np.stack([R,G,B], axis = 2)

        
    def plot(self, color_e = '#ECF653', figsize=(12,12), 
             font = 14, save = 'GOES_plot.jpg',
             title_p = 'GOES-16 Banda 13 - Air mass', axis = 'off', 
             lines = True, gridd = True, cb = True):
        
        import cartopy, cartopy.crs as ccrs, matplotlib.pyplot as plt
        from datetime import datetime, timedelta
        
        plt.figure(figsize = figsize)
    
        # Reproyección geoestacionaria
        cl = self.geo_proj.longitude_of_projection_origin
        sh = self.geo_proj.perspective_point_height
        
        ax = plt.axes(projection = ccrs.Geostationary(central_longitude = cl, 
                                                      satellite_height = sh))
        
        if lines == True:
            ax.coastlines(resolution='10m', color=color_e, linewidth=1) # Líneas de costa
            ax.add_feature(cartopy.feature.BORDERS,                     # Países
                           edgecolor = color_e, linewidth=0.5) 
        if gridd == True:
            ax.gridlines(color='white', alpha=0.5, linestyle='--', linewidth=0.5) # Grillado
        
        plt.axis(axis)

        img = ax.imshow(self.data, origin = 'upper', extent = self.img_extent)
        
        if cb == True:
            plt.colorbar(img, orientation = 'vertical',  # Orientación
                         pad = 0.01,                     # Separación con la imagen
                         fraction = 0.04)                # Escala de la barra
    
        plt.title(title_p, # Título
                  fontweight='bold', fontsize=font,   # Formato de letra
                  loc='left')                         # Locación
    
        # Fecha de la imagen
        seg = int(self.file.variables['time_bounds'][0])
        date = datetime(2000,1,1,12) + timedelta(seconds=seg)
        date = date.strftime('%d %B %Y %H:%M UTC')
        plt.title('Full Disk\n' + date, fontweight='bold', fontsize=font, loc='right')
    
        plt.savefig(save, dpi = 300)   # Guardado
              
class Dust:
    def __init__(self, img, extent = None):
        from netCDF4 import Dataset
        import numpy as np
        
        self.file = Dataset(img[0])
        self.geo_proj = self.file.variables['goes_imager_projection']
        self.var = self.file.variables

        if  extent != None:
            lly, llx = geo2grid(extent[1], extent[0], self.file)
            ury, urx = geo2grid(extent[3], extent[2], self.file)
            
            self.data1 = self.file.variables['CMI'][ury:lly, llx:urx][::4 ,::4] - 273.15 # 11
            self.data2 = Dataset(img[1]).variables['CMI'][ury:lly, llx:urx][::4 ,::4] - 273.15 # 13
            self.data3 = Dataset(img[2]).variables['CMI'][ury:lly, llx:urx][::4 ,::4] - 273.15 # 14
            self.data3 = Dataset(img[3]).variables['CMI'][ury:lly, llx:urx][::4 ,::4] - 273.15 # 15
            
            self.img_extent = convertExtent2GOESProjection(extent)

        else:
            self.data1 = self.file.variables['CMI'][:,:][::4 ,::4] - 273.15  # Channel 11
            self.data2 = Dataset(img[1]).variables['CMI'][:,:][::4 ,::4] - 273.15  # Channel 13
            self.data3 = Dataset(img[2]).variables['CMI'][:,:][::4 ,::4] - 273.15  # Channel 14
            self.data4 = Dataset(img[3]).variables['CMI'][:,:][::4 ,::4] - 273.15  # Channel 15
            self.img_extent = (-5434894.67527,5434894.67527,
                               -5434894.67527,5434894.67527)
            
        def function(band, parameters):
            A = np.clip(band, parameters[0], parameters[1])
            A = ((A - parameters[0])/(parameters[1]- parameters[0]))**(1/parameters[2])
            return A
        
        R = function(self.data4 - self.data2, parameters = [-6.7, 2.6, 1])
        G = function(self.data3 - self.data1, parameters = [-0.5, 20., 2.5])
        B = function(self.data2, parameters = [-11.95, 15.55, 1])
        
        self.data = np.stack([R,G,B], axis = 2)
        
    def plot(self, color_e = '#ECF653', figsize=(12,12), 
             font = 14, save = 'GOES_plot.jpg',
             title_p = 'GOES-16 Banda 13 - Cloud day phase', axis = 'off', 
             lines = True, gridd = True, cb = True):
        
        import cartopy, cartopy.crs as ccrs, matplotlib.pyplot as plt
        from datetime import datetime, timedelta
        
        plt.figure(figsize = figsize)
    
        # Reproyección geoestacionaria
        cl = self.geo_proj.longitude_of_projection_origin
        sh = self.geo_proj.perspective_point_height
        
        ax = plt.axes(projection = ccrs.Geostationary(central_longitude = cl, 
                                                      satellite_height = sh))
        
        if lines == True:
            ax.coastlines(resolution='10m', color=color_e, linewidth=1) # Líneas de costa
            ax.add_feature(cartopy.feature.BORDERS,                     # Países
                           edgecolor = color_e, linewidth=0.5) 
        if gridd == True:
            ax.gridlines(color='white', alpha=0.5, linestyle='--', linewidth=0.5) # Grillado
        
        plt.axis(axis)

        img = ax.imshow(self.data, origin = 'upper', extent = self.img_extent)
        
        if cb == True:
            plt.colorbar(img, orientation = 'vertical',  # Orientación
                         pad = 0.01,                     # Separación con la imagen
                         fraction = 0.04)                # Escala de la barra
    
        plt.title(title_p, # Título
                  fontweight='bold', fontsize=font,   # Formato de letra
                  loc='left')                         # Locación
    
        # Fecha de la imagen
        seg = int(self.file.variables['time_bounds'][0])
        date = datetime(2000,1,1,12) + timedelta(seconds=seg)
        date = date.strftime('%d %B %Y %H:%M UTC')
        plt.title('Full Disk\n' + date, fontweight='bold', fontsize=font, loc='right')
    
        plt.savefig(save, dpi = 300)   # Guardado
        
class SO2:
    def __init__(self, img, extent = None):
        from netCDF4 import Dataset
        import numpy as np
        
        self.file = Dataset(img[0])
        self.geo_proj = self.file.variables['goes_imager_projection']
        self.var = self.file.variables

        if  extent != None:
            lly, llx = geo2grid(extent[1], extent[0], self.file)
            ury, urx = geo2grid(extent[3], extent[2], self.file)
            
            self.data1 = self.file.variables['CMI'][ury:lly, llx:urx][::4 ,::4] - 273.15 # 9
            self.data2 = Dataset(img[1]).variables['CMI'][ury:lly, llx:urx][::4 ,::4] - 273.15 # 10
            self.data3 = Dataset(img[2]).variables['CMI'][ury:lly, llx:urx][::4 ,::4] - 273.15 # 11
            self.data3 = Dataset(img[3]).variables['CMI'][ury:lly, llx:urx][::4 ,::4] - 273.15 # 13
            
            self.img_extent = convertExtent2GOESProjection(extent)

        else:
            self.data1 = self.file.variables['CMI'][:,:][::4 ,::4] - 273.15  # Channel 9
            self.data2 = Dataset(img[1]).variables['CMI'][:,:][::4 ,::4] - 273.15  # Channel 10
            self.data3 = Dataset(img[2]).variables['CMI'][:,:][::4 ,::4] - 273.15  # Channel 11
            self.data4 = Dataset(img[3]).variables['CMI'][:,:][::4 ,::4] - 273.15  # Channel 13
            self.img_extent = (-5434894.67527,5434894.67527,
                               -5434894.67527,5434894.67527)
            
        def function(band, parameters):
            A = np.clip(band, parameters[0], parameters[1])
            A = ((A - parameters[0])/(parameters[1]- parameters[0]))**(1/parameters[2])
            return A
        
        R = function(self.data1 - self.data2, parameters = [-4., 2., 1])
        G = function(self.data4 - self.data3, parameters = [-4., 5., 1])
        B = function(self.data4, parameters = [-30.1, 29.8, 1])
        
        self.data = np.stack([R,G,B], axis = 2)
        
    def plot(self, color_e = '#ECF653', figsize=(12,12), 
             font = 14, save = 'GOES_plot.jpg',
             title_p = 'GOES-16 Banda 13 - SO2', axis = 'off', 
             lines = True, gridd = True, cb = True):
        
        import cartopy, cartopy.crs as ccrs, matplotlib.pyplot as plt
        from datetime import datetime, timedelta
        
        plt.figure(figsize = figsize)
    
        # Reproyección geoestacionaria
        cl = self.geo_proj.longitude_of_projection_origin
        sh = self.geo_proj.perspective_point_height
        
        ax = plt.axes(projection = ccrs.Geostationary(central_longitude = cl, 
                                                      satellite_height = sh))
        
        if lines == True:
            ax.coastlines(resolution='10m', color=color_e, linewidth=1) # Líneas de costa
            ax.add_feature(cartopy.feature.BORDERS,                     # Países
                           edgecolor = color_e, linewidth=0.5) 
        if gridd == True:
            ax.gridlines(color='white', alpha=0.5, linestyle='--', linewidth=0.5) # Grillado
        
        plt.axis(axis)

        img = ax.imshow(self.data, origin = 'upper', extent = self.img_extent)
        
        if cb == True:
            plt.colorbar(img, orientation = 'vertical',  # Orientación
                         pad = 0.01,                     # Separación con la imagen
                         fraction = 0.04)                # Escala de la barra
    
        plt.title(title_p, # Título
                  fontweight='bold', fontsize=font,   # Formato de letra
                  loc='left')                         # Locación
    
        # Fecha de la imagen
        seg = int(self.file.variables['time_bounds'][0])
        date = datetime(2000,1,1,12) + timedelta(seconds=seg)
        date = date.strftime('%d %B %Y %H:%M UTC')
        plt.title('Full Disk\n' + date, fontweight='bold', fontsize=font, loc='right')
    
        plt.savefig(save, dpi = 300)   # Guardado

class Night_microphysics:
    def __init__(self, img, extent = None):
        from netCDF4 import Dataset
        import numpy as np
        
        self.file = Dataset(img[0])
        self.geo_proj = self.file.variables['goes_imager_projection']
        self.var = self.file.variables

        if  extent != None:
            lly, llx = geo2grid(extent[1], extent[0], self.file)
            ury, urx = geo2grid(extent[3], extent[2], self.file)
            
            self.data1 = self.file.variables['CMI'][ury:lly, llx:urx][::4 ,::4] - 273.15 # 7
            self.data2 = Dataset(img[1]).variables['CMI'][ury:lly, llx:urx][::4 ,::4] - 273.15 # 13
            self.data3 = Dataset(img[2]).variables['CMI'][ury:lly, llx:urx][::4 ,::4] - 273.15 # 15
            
            self.img_extent = convertExtent2GOESProjection(extent)

        else:
            self.data1 = self.file.variables['CMI'][:,:][::4 ,::4] - 273.15  # Channel 7
            self.data2 = Dataset(img[1]).variables['CMI'][:,:][::4 ,::4] - 273.15  # Channel 13
            self.data3 = Dataset(img[2]).variables['CMI'][:,:][::4 ,::4] - 273.15  # Channel 15
            self.img_extent = (-5434894.67527,5434894.67527,
                               -5434894.67527,5434894.67527)
            
        def function(band, parameters):
            A = np.clip(band, parameters[0], parameters[1])
            A = ((A - parameters[0])/(parameters[1]- parameters[0]))**(1/parameters[2])
            return A
        
        R = function(self.data3 - self.data2, parameters = [-6.7, 2.6, 1])
        G = function(self.data2 - self.data1, parameters = [-3.1, 5.2, 1])
        B = function(self.data2, parameters = [-29.6, 19.5, 1])
        
        self.data = np.stack([R,G,B], axis = 2)
        
    def plot(self, color_e = '#ECF653', figsize=(12,12), 
             font = 14, save = 'GOES_plot.jpg',
             title_p = 'GOES-16 Banda 13 - Night microphysics', axis = 'off', 
             lines = True, gridd = True, cb = True):
        
        import cartopy, cartopy.crs as ccrs, matplotlib.pyplot as plt
        from datetime import datetime, timedelta
        
        plt.figure(figsize = figsize)
    
        # Reproyección geoestacionaria
        cl = self.geo_proj.longitude_of_projection_origin
        sh = self.geo_proj.perspective_point_height
        
        ax = plt.axes(projection = ccrs.Geostationary(central_longitude = cl, 
                                                      satellite_height = sh))
        
        if lines == True:
            ax.coastlines(resolution='10m', color=color_e, linewidth=1) # Líneas de costa
            ax.add_feature(cartopy.feature.BORDERS,                     # Países
                           edgecolor = color_e, linewidth=0.5) 
        if gridd == True:
            ax.gridlines(color='white', alpha=0.5, linestyle='--', linewidth=0.5) # Grillado
        
        plt.axis(axis)

        img = ax.imshow(self.data, origin = 'upper', extent = self.img_extent)
        
        if cb == True:
            plt.colorbar(img, orientation = 'vertical',  # Orientación
                         pad = 0.01,                     # Separación con la imagen
                         fraction = 0.04)                # Escala de la barra
    
        plt.title(title_p, # Título
                  fontweight='bold', fontsize=font,   # Formato de letra
                  loc='left')                         # Locación
    
        # Fecha de la imagen
        seg = int(self.file.variables['time_bounds'][0])
        date = datetime(2000,1,1,12) + timedelta(seconds=seg)
        date = date.strftime('%d %B %Y %H:%M UTC')
        plt.title('Full Disk\n' + date, fontweight='bold', fontsize=font, loc='right')
    
        plt.savefig(save, dpi = 300)   # Guardado

class Day_land_cloud:
    def __init__(self, img, extent = None):
        import numpy as np
        from netCDF4 import Dataset
        
        self.file = Dataset(img[0])
        self.geo_proj = self.file.variables['goes_imager_projection']
        self.var = self.file.variables

        if  extent != None:
            lly, llx = geo2grid(extent[1], extent[0], self.file)
            ury, urx = geo2grid(extent[3], extent[2], self.file)
            
            self.data1 = self.file.variables['CMI'][ury:lly, llx:urx][::16 ,::16]  # 2
            self.data2 = Dataset(img[1]).variables['CMI'][ury:lly, llx:urx][::4 ,::4] # 3
            self.data3 = Dataset(img[2]).variables['CMI'][ury:lly, llx:urx][::4 ,::4] # 5

            self.img_extent = convertExtent2GOESProjection(extent)

        else:
            self.data1 = self.file.variables['CMI'][::16 ,::16]   # 2[::8 ,::8]
            self.data2 = Dataset(img[1]).variables['CMI'][::4 ,::4] # 3[::2 ,::2]
            self.data3 = Dataset(img[2]).variables['CMI'][::4 ,::4] # 5[::2 ,::2]
            
            self.img_extent = (-5434894.67527,5434894.67527,
                               -5434894.67527,5434894.67527)
            
        def function(band, parameters):
            A = np.clip(band, parameters[0], parameters[1])
            A = ((A - parameters[0])/(parameters[1]- parameters[0]))**(1/parameters[2])
            return A
        """
        R = function(self.data3, parameters = [0., 0.975, 1])
        G = function(self.data2, parameters = [0., 1.086, 1])
        B = function(self.data1, parameters = [0., 1., 1])
        """
        R = self.data3
        G = self.data2
        B = self.data1
        
        print(R.shape)
        print(G.shape)
        print(B.shape) 
        self.data = np.stack([R,G,B], axis = 2)
        
    def plot(self, color_e = '#ECF653', figsize=(12,12), 
             font = 14, save = 'GOES_plot.jpg',
             title_p = 'GOES-16 Banda 13 - Day_land_cloud', axis = 'off', 
             lines = True, gridd = True, cb = True):
        
        import cartopy, cartopy.crs as ccrs, matplotlib.pyplot as plt
        from datetime import datetime, timedelta
        
        plt.figure(figsize = figsize)
    
        # Reproyección geoestacionaria
        cl = self.geo_proj.longitude_of_projection_origin
        sh = self.geo_proj.perspective_point_height
        
        ax = plt.axes(projection = ccrs.Geostationary(central_longitude = cl, 
                                                      satellite_height = sh))
        
        if lines == True:
            ax.coastlines(resolution='10m', color=color_e, linewidth=1) # Líneas de costa
            ax.add_feature(cartopy.feature.BORDERS,                     # Países
                           edgecolor = color_e, linewidth=0.5) 
        if gridd == True:
            ax.gridlines(color='white', alpha=0.5, linestyle='--', linewidth=0.5) # Grillado
        
        plt.axis(axis)

        img = ax.imshow(self.data, origin = 'upper', extent = self.img_extent)
        
        if cb == True:
            plt.colorbar(img, orientation = 'vertical',  # Orientación
                         pad = 0.01,                     # Separación con la imagen
                         fraction = 0.04)                # Escala de la barra
    
        plt.title(title_p, # Título
                  fontweight='bold', fontsize=font,   # Formato de letra
                  loc='left')                         # Locación
    
        # Fecha de la imagen
        seg = int(self.file.variables['time_bounds'][0])
        date = datetime(2000,1,1,12) + timedelta(seconds=seg)
        date = date.strftime('%d %B %Y %H:%M UTC')
        plt.title('Full Disk\n' + date, fontweight='bold', fontsize=font, loc='right')
    
        plt.savefig(save, dpi = 300)   # Guardado

"""
dimensions(sizes): y(5424), x(5424)
img.variables['CMI']   => metadata de la variable
img.variables['goes_imager_projection']  => información del grillado


"""