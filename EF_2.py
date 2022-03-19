# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 02:45:32 2022

@author: Kevin
"""
import os
os.chdir(r'C:\\Users\\Kevin\\Desktop\\UNALM\\OctCiclo\\Senso_rem\\RS_GOES16')
from RS_GOES16 import Palette_goes, GOES_B2

# Carga de datos
os.chdir(r'C:\\Users\\Kevin\\Desktop\\UNALM\\OctCiclo\\Senso_rem')
b13_00 = r'EFinal\\OR_ABI-L2-CMIPF-M6C13_G16_s20212870000206_e20212870009525_c20212870010015.nc'
b13_07 = r'EFinal\\OR_ABI-L2-CMIPF-M6C13_G16_s20212870700208_e20212870709527_c20212870710007.nc'
b13_12 = r'EFinal\\OR_ABI-L2-CMIPF-M6C13_G16_s20212871200208_e20212871209527_c20212871210006.nc'
b13_18 = r'EFinal\\OR_ABI-L2-CMIPF-M6C13_G16_s20212871800206_e20212871809525_c20212871810016.nc'

b02_00 = r'EFinal\\OR_ABI-L2-CMIPF-M6C02_G16_s20212870000206_e20212870009514_c20212870009590.nc'
b02_07 = r'EFinal\\OR_ABI-L2-CMIPF-M6C02_G16_s20212870700208_e20212870709516_c20212870709588.nc'
b02_12 = r'EFinal\\OR_ABI-L2-CMIPF-M6C02_G16_s20212871200208_e20212871209516_c20212871209591.nc'
b02_18 = r'EFinal\\OR_ABI-L2-CMIPF-M6C02_G16_s20212871800206_e20212871809514_c20212871809581.nc'


extent = [-85., -20.,  # Lon mín, Lat mín
          -65., 2.5]   # Lon máx, Lat máx

extent2 = [-80., -25.,  # Lon mín, Lat mín
           -50., 2.5]   # Lon máx, Lat máx, extent = extent2

extent3 = [-65., -22.5,  # Lon mín, Lat mín
           -50., -12.5]   # Lon máx, Lat máx, extent = extent2

cmap_a = Palette_goes(i = 4)

d0 = GOES_B2(b02_18, extent = extent3)
d0.plot(cmap = 'gray', min_max = [0, 1], cb = False,
        title_p = 'GOES-16 Banda 2 - Temperatura de brillo\nSouth American Low-Level Jet',
        color_e = '#23FF04', save = 'EFinal\\SALLJ14_2_18Z.jpg')
