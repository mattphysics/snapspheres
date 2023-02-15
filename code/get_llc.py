# Get data from LLC4320 simulation

from dask.diagnostics import ProgressBar
from xmitgcm import llcreader
import xarray as xr

import numpy as np
import xgcm
import sys
# sys.path.append('/home/jovyan/ECCOv4-py/ECCOv4-py')

import ecco_v4_py as ecco

def regrid_timeslices_to_res(data,x,y,dlat=1.,dlon=1.):
    '''
    Regrid ECCO data native LLC grid to lat-lon grid
    Inputs:
        -to_full: Interpolate to full longitudes? (...-1,0,1,2,3,...)?
                    uses xgcm functionality, linear interpolation
    '''
    new_grid_delta_lat = dlat
    new_grid_delta_lon = dlon

    new_grid_min_lat = -90+new_grid_delta_lat/2
    new_grid_max_lat = 90-new_grid_delta_lat/2

    new_grid_min_lon = -180+new_grid_delta_lon/2
    new_grid_max_lon = 180-new_grid_delta_lon/2

    # datas = []

    new_grid_lon, new_grid_lat, new_grid_lon_edges, new_grid_lat_edges, field_nearest_1deg_i =\
            ecco.resample_to_latlon(x, \
                                    y, \
                                    data,\
                                    new_grid_min_lat, new_grid_max_lat, new_grid_delta_lat,\
                                    new_grid_min_lon, new_grid_max_lon, new_grid_delta_lon,\
                                    fill_value = np.NaN, \
                                    mapping_method = 'nearest_neighbor',
                                    radius_of_influence = 120000)
    datas = field_nearest_1deg_i
        
    # data_latlon = xr.DataArray(dims=['time','lat','lon'],coords=dict(
    data_latlon = xr.DataArray(dims=['lat','lon'],coords=dict(
                                                        #  time=(['time',],data.time),
                                                         lat=(['lat',], new_grid_lat[:,0]),
                                                        #  lon=(['lon',], new_grid_lon[0,:])),data=np.array(datas))
                                                         lon=(['lon',], new_grid_lon[0,:])),data=datas)
    data_latlon['lat'].attrs['long_name'] = 'latitude'
    data_latlon['lat'].attrs['units'] = 'degrees'
    data_latlon['lon'].attrs['long_name'] = 'longitude'
    data_latlon['lon'].attrs['units'] = 'degrees'
    
    return data_latlon

# define the connectivity between faces
face_connections = {'face':
                    {0: {'X':  ((12, 'Y', False), (3, 'X', False)),
                         'Y':  (None,             (1, 'Y', False))},
                     1: {'X':  ((11, 'Y', False), (4, 'X', False)),
                         'Y':  ((0, 'Y', False),  (2, 'Y', False))},
                     2: {'X':  ((10, 'Y', False), (5, 'X', False)),
                         'Y':  ((1, 'Y', False),  (6, 'X', False))},
                     3: {'X':  ((0, 'X', False),  (9, 'Y', False)),
                         'Y':  (None,             (4, 'Y', False))},
                     4: {'X':  ((1, 'X', False),  (8, 'Y', False)),
                         'Y':  ((3, 'Y', False),  (5, 'Y', False))},
                     5: {'X':  ((2, 'X', False),  (7, 'Y', False)),
                         'Y':  ((4, 'Y', False),  (6, 'Y', False))},
                     6: {'X':  ((2, 'Y', False),  (7, 'X', False)),
                         'Y':  ((5, 'Y', False),  (10, 'X', False))},
                     7: {'X':  ((6, 'X', False),  (8, 'X', False)),
                         'Y':  ((5, 'X', False),  (10, 'Y', False))},
                     8: {'X':  ((7, 'X', False),  (9, 'X', False)),
                         'Y':  ((4, 'X', False),  (11, 'Y', False))},
                     9: {'X':  ((8, 'X', False),  None),
                         'Y':  ((3, 'X', False),  (12, 'Y', False))},
                     10: {'X': ((6, 'Y', False),  (11, 'X', False)),
                          'Y': ((7, 'Y', False),  (2, 'X', False))},
                     11: {'X': ((10, 'X', False), (12, 'X', False)),
                          'Y': ((8, 'Y', False),  (1, 'X', False))},
                     12: {'X': ((11, 'X', False), None),
                          'Y': ((9, 'Y', False),  (0, 'X', False))}}}


# model = llcreader.ECCOPortalLLC2160Model()
model = llcreader.ECCOPortalLLC4320Model()
# ds = model.get_dataset(varnames=['UVEL','VVEL'], type='latlon')
ds = model.get_dataset(varnames=['U','V','oceQnet','SIarea'], type='faces')
dsday = ds[['U','V','oceQnet','SIarea']].sel(time='2012-03-04',k=0).mean('time')

print('Size of dataset:')
print(dsday.nbytes / 1e9)


# with ProgressBar():
#     dsday.load()
    
print('regrid qnet & sic')
with ProgressBar():
    qnet_latlon = regrid_timeslices_to_res(dsday['oceQnet'],dsday.XC,dsday.YC,dlat=0.025,dlon=0.025)
    sic_latlon = regrid_timeslices_to_res(dsday['SIarea'],dsday.XC,dsday.YC,dlat=0.025,dlon=0.025)
    
print('save qnet & sic')
# dsday['oceQnet'].to_netcdf('/scratch/ma5952/llc4320_20120403_sic_faces.nc')
# dsday['SIarea'].to_netcdf('/scratch/ma5952/llc4320_20120403_sic_faces.nc')

qnet_latlon.to_netcdf('/scratch/ma5952/llc4320_20120403_oceQnet_0025deg.nc')
sic_latlon.to_netcdf('/scratch/ma5952/llc4320_20120403_sic_0025deg.nc')

## use xgcm for regridding
print('regrid u,v')
# create the grid object
grid = xgcm.Grid(dsday, periodic=False, face_connections=face_connections)

with ProgressBar():
    uday = dsday['U'].load()
    vday = dsday['V'].load()
    
with ProgressBar():
    # vec = grid.interp_2d_vector({'X' : dsday['U'], 'Y' : dsday['V']}, boundary = 'fill')
    vec = grid.interp_2d_vector({'X' : uday, 'Y' : vday}, boundary = 'fill')
    vel_E = vec['X'] * ds['CS'] - vec['Y'] * ds['SN']
    vel_N = vec['X'] * ds['SN'] + vec['Y'] * ds['CS']
    vel_E_latlon = regrid_timeslices_to_res(vel_E,ds.XC,ds.YC,dlat=0.025,dlon=0.025)
    vel_N_latlon = regrid_timeslices_to_res(vel_N,ds.XC,ds.YC,dlat=0.025,dlon=0.025)

print('save u, v')
# vel_E.to_netcdf('/scratch/ma5952/llc4320_20120403_vel_E_faces_center.nc')
# vel_N.to_netcdf('/scratch/ma5952/llc4320_20120403_vel_N_faces_center.nc')

vel_E_latlon.to_netcdf('/scratch/ma5952/llc4320_20120403_velE_0025deg.nc')
vel_N_latlon.to_netcdf('/scratch/ma5952/llc4320_20120403_velN_0025deg.nc')


print('get vorticity')
with ProgressBar():
    zeta = ((-grid.diff((dsday['U'] * dsday['dxC']).load(), 'Y') + grid.diff((dsday['V'] * dsday['dyC']).load(), 'X')) / dsday['rAz']).load()
with ProgressBar():
    zeta_latlon = regrid_timeslices_to_res(zeta, dsday.XG, dsday.YG, dlat=0.02,dlon=0.02)
zeta_latlon.to_netcdf('/scratch/ma5952/llc4320_20120403_zeta_002deg.nc')

print('Done')