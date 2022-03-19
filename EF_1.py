# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 02:45:32 2022

@author: Kevin
"""
import os
os.chdir(r'C:\\Users\\Kevin\\Desktop\\UNALM\\OctCiclo\\Senso_rem\\RS_GOES16')
from RS_GOES16 import Palette_goes, GOES_BT

# Carga de datos
os.chdir(r'C:\\Users\\Kevin\\Desktop\\UNALM\\OctCiclo\\Senso_rem')
"""
img11 = r'EFinal\\OR_ABI-L2-CMIPF-M3C13_G16_s20172231800373_e20172231811151_c20172231811216.nc'
img13 = r'EFinal\\OR_ABI-L2-CMIPF-M3C13_G16_s20172241800370_e20172241811148_c20172241811220.nc'
img15 = r'EFinal\\OR_ABI-L2-CMIPF-M3C13_G16_s20172251800368_e20172251811146_c20172251811220.nc'
img17 = r'EFinal\\OR_ABI-L2-CMIPF-M3C13_G16_s20172261800367_e20172261811145_c20172261811219.nc'
img18 = r'EFinal\\OR_ABI-L2-CMIPF-M3C13_G16_s20172271800366_e20172271811145_c20172271811218.nc'
img21 = r'EFinal\\OR_ABI-L2-CMIPF-M3C13_G16_s20172331800379_e20172331811157_c20172331811228.nc'
"""
b08_18 = r'EFinal\\OR_ABI-L2-CMIPF-M6C08_G16_s20212871800206_e20212871809514_c20212871809593.nc'
b10_18 = r'EFinal\\OR_ABI-L2-CMIPF-M6C10_G16_s20212871800206_e20212871809525_c20212871810024.nc'
b13_18 = r'EFinal\\OR_ABI-L2-CMIPF-M6C13_G16_s20212871800206_e20212871809525_c20212871810016.nc'
b13_18n1 = r'EFinal\\OR_ABI-L2-CMIPF-M6C13_G16_s20212881800207_e20212881809526_c20212881810015.nc'

extent = [-85., -20.,  # Lon mín, Lat mín
          -65., 2.5]   # Lon máx, Lat máx

extent2 = [-80., -25.,  # Lon mín, Lat mín
           -50., 2.5]   # Lon máx, Lat máx, extent = extent2

extent3 = [-65., -22.5,  # Lon mín, Lat mín
           -50., -12.5]   # Lon máx, Lat máx, extent = extent2

cmap_a = Palette_goes(i = 4)
cmap_10 = Palette_goes(i = 'Channel_10')

d0 = GOES_BT(b10_18, extent = extent2)
d0.plot(cmap = cmap_10[0], min_max = cmap_10[1],
        title_p = 'GOES-16 Banda 10 - Vapor de agua (bajos niveles)\nSouth American Low-Level Jet',
        color_e = 'black', save = 'EFinal\\SALLJ14_10_18Z.jpg')

b8 = GOES_BT(b08_18, extent = extent2)
b8.plot(cmap = cmap_10[0], min_max = cmap_10[1],
        title_p = 'GOES-16 Banda 8 - Vapor de agua (niveles altos)\nSouth American Low-Level Jet',
        color_e = 'black', save = 'EFinal\\SALLJ14_08_18Z.jpg')

b13 = GOES_BT(b13_18, extent = extent2)
b13.plot(cmap = cmap_a[0], min_max = cmap_a[1],
        title_p = 'GOES-16 Banda 13 - Ventana infrarroja de onda larga (limpia)\nSouth American Low-Level Jet',
        color_e = 'black', save = 'EFinal\\SALLJ15_13_18Z.jpg')

b8 = GOES_BT(b08_18)
b8.plot(cmap = cmap_10[0], min_max = cmap_10[1],
        title_p = 'GOES-16 Banda 8 - Vapor de agua (niveles altos)\nSouth American Low-Level Jet',
        color_e = 'black', save = 'EFinal\\SALLJ14_08_18Z.jpg')

b13 = GOES_BT(b13_18)
b13.plot(cmap = cmap_a[0], min_max = cmap_a[1],
        title_p = 'GOES-16 Banda 13 - Ventana infrarroja de onda larga (limpia)\nSouth American Low-Level Jet',
        color_e = 'black', save = 'EFinal\\SALLJ14_13_18Z.jpg')