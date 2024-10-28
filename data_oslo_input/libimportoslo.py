
import math
import os
#import matplotlib.pyplot as plt #for plotting
#from scipy.interpolate import Rbf #for regression.
import pyvista as vtki #for dealing with vtk geometry file i.e. Utest_sample.vtki
#import numpy as np
#import os, sys
import pandas as pd #make and work with dataframes.
##import vtk
##from os.path import isdir, isfile, join
##from os.path import expanduser
##import json
import time 
##import random
#import time as timer
import pickle #
import requests
import logging



#import trame
#import vtk
#import trame_vtk

logging.basicConfig(
    filename='my_log_file.log',   # Name of the log file
    level=logging.INFO,           # Set the log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log message format
)


# %run '/media/mandart/D/AI4Hydrop_Prague/ROM/Results/data_POD/libimport.py'

def offset_dataset(deltaX,deltaY,deltaZ,df):

    # Load the dataset into a pandas DataFrame
    # Adjust the path to your dataset CSV file
    #df = pd.read_csv(dataset_path)

    # Define the offsets for each dimension
    # Change this value to the desired offset for Z

    # Apply the offsets to the X, Y, and Z columns
    df['X'] = df['X'] + deltaX
    df['Y'] = df['Y'] + deltaY
    df['Z'] = df['Z'] + deltaZ

    # Save the modified DataFrame back to a new CSV file
       # Adjust the path for the output CSV file
    #df.to_csv(output_path, index=False)

    
    return df

#***************************************************************************************************************************************************'

def save_csv_reconstructed (grid2,variable,wind_dir,fn,vectorU):     
    #print("Turbine located at around Y=1500, X=1000 in figure below")
    data = {'X': grid2.points[:, 0],    #cell_centers.points
            'Y': grid2.points[:, 1],
            'Z': grid2.points[:, 2],
        }
    df = pd.DataFrame(data)

#Suggest Time for reconstruction in arange below. 
#To reconstruct at all time-steps and save images, start loop from 0.

    if vectorU==False:        
        turbke=grid2["RECON_tke_at_WD_"+str(wind_dir)]
        data2 = {'tke':turbke}
        df=df.join(pd.DataFrame(data2))
    else:
        vel=grid2["RECON_U_at_WD_"+str(wind_dir)]  #tke_at_wind_direction.reshape(-1, 3, order='F')
        data2 = {'Velocity_X_': vel[:, 0],'Velocity_Y_': vel[:, 1],'Velocity_Z_': vel[:, 2]}
        df=df.join(pd.DataFrame(data2))            
        
       
        # saveimage(i)

# Save CSV    
    csv_file = fn+'/'+ f'{variable}'+'_'+str(wind_dir)+'_Output.csv'     
    df.to_csv(csv_file, index=False)
    print(f"CSV file saved successfully: {csv_file}")

    return df 

# #***************************************************************************************************************************************************'

def meters_to_decimal_degrees(delta_x, delta_y, reference_lat):
    import math
    # Earth's radius at the reference latitude (average of meridional and equatorial radii)
    earth_radius = 6371000  # in meters

    # Convert meters to degrees for longitude
    delta_lon = (delta_x / earth_radius) * (180 / (math.pi * math.cos(math.radians(reference_lat))))

    # Convert meters to degrees for latitude
    delta_lat = (delta_y / earth_radius) * (180 / math.pi)

    return delta_lat, delta_lon

# Example usage
#delta_x = 1000  # 1000 meters eastward
#delta_y = -500  # 500 meters southward
#reference_lat = 40.7128  # Reference latitude in decimal degrees (New York City)

#delta_lat, delta_lon = meters_to_decimal_degrees(delta_x, delta_y, reference_lat)
#print("Relative change in Latitude: {:.6f} degrees".format(delta_lat))
#print("Relative change in Longitude: {:.6f} degrees".format(delta_lon))


# #***************************************************************************************************************************************************'

# Method 2. The conversion from meters to decimal degrees is based on the following formulae: 
# 1 degree of latitude is approximately 111,139 meters.
# 1 degree of longitude is approximately 111,139 meters * cos(latitude).
# Define a function to convert meters to decimal degrees, latitude in decimal degree
def convert_meters_to_decimal_degrees(x, y, latitude):
    # Convert meters to decimal degrees
    x_deg = x / (111139 * math.cos(math.radians(latitude)))
    y_deg = y / 111139

    return x_deg, y_deg

# Test the function with some example values
#x_meters = 1000  # 1000 meters East
#y_meters = 2000  # 2000 meters North
#latitude = 45  # latitude in decimal degrees

#x_deg, y_deg = convert_meters_to_decimal_degrees(x_meters, y_meters, latitude)

#print(f"X: {x_meters} meters = {x_deg} degrees")
#print(f"Y: {y_meters} meters = {y_deg} degrees")

# #***************************************************************************************************************************************************'

def reconstruct_AI_for_winddirection(variable,wind_direc,fn,vectorU=True):     
    deltaX=-1480
    deltaY=2120
    deltaZ=229  
    
    #import time    
    start = time.process_time()
    # your code here    
    # Assume a wind direction 
    # wind_dir=np.rand270 #in degrees
    folder = 'data_POD'
    
    # Load the Rbf interpolators from a file
    print(fn+'/'+f"{variable}"+'_rbf_interpolators.pkl')
    with open(fn+'/'+f"{variable}"+'_rbf_interpolators.pkl', 'rb') as file:
        loaded_interpolators = pickle.load(file)
        
        
    #Obtain basis functions : Load saved basis functions (modes) and mean turbulent kinetic energy
    #----------------------------------------------------------------------------------------------.     
    filename = fn+'/POD_data_Oslo_' +f"{variable}"+'.npz'    
        
    PODdata=np.load(filename)

    Phit=PODdata['tbasis']
    tke_mean=PODdata['tmean']   
     
    result_sample=vtki.read('Utest_sample.vtk') 
    
    for wind_dir in [wind_direc]:     
        
        #Obtain coefficient for this wind direction
        #------------------------------------------------------------------  
        # Interpolate using the loaded interpolators (for demonstration)
        predicted_coef = []
        max_value=360
        xsin_winddir=np.sin(2 * np.pi * wind_dir / max_value)
        xcos_winddir=np.cos(2 * np.pi * wind_dir / max_value)
        for i, rbf1 in enumerate(loaded_interpolators):
            predicted_coef.append(rbf1(xsin_winddir,xcos_winddir))
            print('Coeff for mode', i+1 , 'is ', rbf1(xsin_winddir,xcos_winddir))
        print(' ')
         
       #Reconstruct flow field from the basis modes, the mean and the computed coefficients
        #----------------------------------------------------------------------------------------------------
        tke_at_wind_direction=np.dot(Phit,np.array(predicted_coef))+tke_mean

        #Visualize reconstructed flow field.
        #----------------------------------------------------------------------------------------------------
        if vectorU==False:
            result_sample.point_data["RECON_tke_at_WD_"+str(wind_dir)]=tke_at_wind_direction
            df=save_csv_reconstructed(result_sample,variable,wind_dir,fn,vectorU)
            df['tke'] = df['tke'].clip(lower=0.001) 
            #df=pd.read_csv(csv_file)
            #Make vertiport at 0,0,267. x=x-1480,y=y+2120,z=z+229.
            
        
        else:
            U_reshaped=tke_at_wind_direction.reshape(-1, 3, order='F') 
            result_sample.point_data.set_vectors(U_reshaped,"RECON_U_at_WD_"+str(wind_dir))
            df=save_csv_reconstructed(result_sample,variable,wind_dir,fn,vectorU)
            
        result_sample.save('./Results/recon_added_' +f"{variable}"+'.vtk')
        
        #Offset grid to make vertiport the center 
        df_offset=offset_dataset(deltaX,deltaY,deltaZ,df)
        df_offset.to_csv(fn+'/'+ f'{variable}'+'_'+str(wind_dir)+'_Output_Transformed.csv', index=False)

        print("Reconstruction completed. Output saved to:", fn+'/'+ f'{variable}'+'_'+str(wind_dir)+'_Output_Transformed.csv')  
        print(' ')
            
    #Print time taken
    #print(time.process_time() - start)
    print(' ')    
    return result_sample,df_offset,time.process_time() - start

# #***************************************************************************************************************************************************'

def find_nearest_location(X, Y, Z,df):
    distances = np.sqrt((df['X'] - X)**2 + (df['Y'] - Y)**2 + (df['Z'] - Z)**2)
    nearest_index = distances.idxmin()
    #print ('nearest',nearest_index,df.loc[nearest_index])
    return df.loc[nearest_index]

import math
def haversine(lat1, lon1, lat2, lon2):
    from math import sin, cos, sqrt, atan2, radians
    # Convert latitude and longitude from degrees to radians
    lat1=np.radians(lat1)
    lon1=np.radians(lon1)
    lat2=np.radians(lat2)
    lon2=np.radians(lon2)
    
    #radians(lon1), radians(lat2), radians(lon2)
    
    # Haversine formula
    dlat = lat2 - lat1
    
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    #print(a)
    
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = 6371 * c *1000 # Radius of Earth in meters
    #print(distance)
    return distance

def check_limits(X_or_longitude, Y_or_latitude, Z,):
    if df['Y'].min() <= Y_or_latitude <= df['Y'].max() and df['X'].min() <= X_or_longitude <= df['X'].max() and df['Z'].min() <= Z <= df['Z'].max():
        # Process the location
        logging.info("Latitude and longitude are within bounds. Proceeding with further processing.")
        #print("Latitude and longitude are within bounds. Proceeding with further processing.")
        # Further processing can be done here
    else:
        # Print a warning message
        print("Warning: Latitude and/or longitude are out of bounds. Please provide valid values or check input .")
        
def get_U_and_k_for_location(X_or_longitude, Y_or_latitude, Z,df,relative_distance_in_meter__or__Latlong='meter'): 
    nearest_location=[0 for _ in range(len(X_or_longitude))]
    for i in range(len(X_or_longitude)):
                       
        if relative_distance_in_meter__or__Latlong=='meter':
            if df['Y'].min() <= Y_or_latitude[i] <= df['Y'].max() and df['X'].min() <= X_or_longitude[i] <= df['X'].max() and df['Z'].min() <= Z[i] <= df['Z'].max():
                # Process the location
                logging.info("Coordinates are within bounds. Proceeding with further processing.")
                
                # Further processing can be done here
            else:
            # Print a warning message
                print("Warning: Coordinates are out of bounds. Please provide valid values or check input .")
                print(f"Relative Distance from Vertiport in North-south Y direction should be in between the range: {df['Y'].min()}and {df['Y'].max()}")
                print(f"Relative Distance from Vertiport in East-West X direction should be in between the range: {df['Y'].min()} and {df['Y'].max()}")
                return []
            location_data= df.index[(df['X'] == X_or_longitude[i]) & (df['Y'] == Y_or_latitude[i]) & (df['Z'] == Z[i])]
            #print(location_data,'type')
        else:
            if df['latitude'].min() <= Y_or_latitude[i] <= df['latitude'].max() and df['longitude'].min() <= X_or_longitude[i] <= df['longitude'].max() and df['Z'].min() <= Z[i] <= df['Z'].max():
                # Process the location
                logging.info("Latitude and longitude are within bounds. Proceeding with further processing.")
                # Further processing can be done here
            else:
            # Print a warning message
                print("Warning: Latitude and/or longitude are out of bounds. Please provide valid values or check input .")
                print(f"Relative Distance from Vertiport in Latitude Y direction should be in between the range :{df['latitude'].min()} and {df['latitude'].max()}")
                print(f"Relative Distance from Vertiport in Longitude X direction should be in between the range:{df['longitude'].min()} and {df['longitude'].max()}")
                return []
            location_data = df.index[(df['latitude'] == Y_or_latitude[i]) & (df['longitude'] == X_or_longitude[i]) & (df['Z'] == Z[i])]
            # print('index',location_data)      

        if len(location_data) == 0:                 
            if relative_distance_in_meter__or__Latlong=='meter':
                distances = np.sqrt((df['X'] - X_or_longitude[i])**2 + (df['Y'] - Y_or_latitude[i])**2 + (df['Z'] - Z[i])**2)
                print("using nearest based on distance") 
            else:
                distances = np.sqrt((df['longitude'] - X_or_longitude[i])**2 + (df['latitude'] - Y_or_latitude[i])**2 + (df['Z'] - Z[i])**2)
                #distances=haversine(df['latitude'],df['longitude'],Y,X) + np.sqrt(df['Z'] - Z)**2
                #print("using nearest based on lat-long. Haversten  not used.") 
                
            nearest_index = distances.idxmin()             
            # nearest_location[i] = df.loc[nearest_index] #find_nearest_location(X, Y, Z,df)
            nearest_location[i] = nearest_index #df.loc[nearest_index]
            
            #print('nearest',i,nearest_location[i],nearest_index)
            logging.info('nearest',i,nearest_location[i],nearest_index)
            
            #vx=nearest_location['Velocity_X_']
            #vy=nearest_location['Velocity_Y_']
            #vz=nearest_location['Velocity_Z_']
            #Ucomponent=[vx,vy,vz]
            
            #U_values = math.sqrt(vx**2 + vy**2 + vz**2) 
            #k_values = nearest_location['tke']
            #return nearest_location #U_values, k_values, Ucomponent,
        else:
            #U_values = location_data['Velocity_X_']
            #k_values = location_data['k'].tolist()
            nearest_location[i] = location_data #df.loc[nearest_index] #find_nearest_location(X, Y, Z,df)
            logging.info('nearest',i,nearest_location[i],nearest_index)
        del location_data
    return df.loc[nearest_location]   
        
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import mplcursors


def reconstruct_AI_for_winddirection_windspeed(variable,wind_direc,windspeed,fn,vectorU=True):  
    
    deltaX=-1480
    deltaY=2120
    deltaZ=229  
    
    start = time.process_time()
    # your code here    
    # Assume a wind direction 
    # wind_dir=np.rand270 #in degrees
    folder = 'data_POD'
    
    # Load the Rbf interpolators from a file
    logging.info(fn+'/'+f"{variable}"+'_rbf_interpolators_all.pkl')
    with open(fn+'/'+f"{variable}"+'_rbf_interpolators_all.pkl', 'rb') as file:
        loaded_interpolators = pickle.load(file)
        
        
    #Obtain basis functions : Load saved basis functions (modes) and mean turbulent kinetic energy
    #----------------------------------------------------------------------------------------------.     
    filename = fn+'/POD_data_all' +f"{variable}"+'.npz'    
        
    PODdata=np.load(filename)

    Phit=PODdata['tbasis']
    tke_mean=PODdata['tmean']    
    
    result_sample=vtki.read('Utest_sample.vtk')

    for wind_dir in [wind_direc]:
        for wind_spee in [windspeed]:     
            #print (f"wind speed {wind_spee}")
            #print (f"wind direction {wind_dir}")
            #Obtain coefficient for this wind direction
            #------------------------------------------------------------------  
            # Interpolate using the loaded interpolators (for demonstration)
            predicted_coef = []
            max_value=360
            xsin_winddir=np.sin(2 * np.pi * wind_dir / max_value)
            xcos_winddir=np.cos(2 * np.pi * wind_dir / max_value)
            for i, rbf1 in enumerate(loaded_interpolators):
                predicted_coef.append(rbf1(xsin_winddir,xcos_winddir,wind_spee))
                logging.info('Coeff for mode', i+1 , 'is ', rbf1(xsin_winddir,xcos_winddir,wind_spee))

        #Reconstruct flow field from the basis modes, the mean and the computed coefficients
            #----------------------------------------------------------------------------------------------------
            tke_at_wind_direction=np.dot(Phit,np.array(predicted_coef))+tke_mean

            #Visualize reconstructed flow field.
            #----------------------------------------------------------------------------------------------------
            if vectorU==False:
                result_sample.point_data["RECON_tke_at_WD_WS"+str(wind_dir)+"_"+str(wind_spee)]=tke_at_wind_direction
                df=save_csv_reconstructed_all(result_sample,variable,wind_dir,wind_spee,fn,vectorU) #(grid2,variable,wind_dir,wind_speed,fn,vectorU)
                #df=pd.read_csv(csv_file)
                #Make vertiport at 0,0,267. x=x-1480,y=y+2120,z=z+229.               
            
            else:
                U_reshaped=tke_at_wind_direction.reshape(-1, 3, order='F') 
                result_sample.point_data.set_vectors(U_reshaped,"RECON_U_at_WD_WS"+str(wind_dir)+"_"+str(wind_spee))
                df=save_csv_reconstructed_all(result_sample,variable,wind_dir,wind_spee,fn,vectorU)
                
            path1=os.path.join(fn,'VTK_Database')
            if not os.path.exists(path1):
                
                # If it doesn't exist, create the directory
                os.makedirs(path1)
                print(f"Directory '{path1}' created successfully.")
            else:
                logging.info(f"Directory '{path1}' already exists.") 
                   
            result_sample.save(path1+'/Recon_VTK_added_all_ws_wd' +f"{variable}"+'.vtk')
            print(" ")
            print("VTK reconstruction saved at:", path1+'/Recon_VTK_added_all_ws_wd' +f"{variable}"+'.vtk')      
            #Offset grid to make vertiport the center 
            df_offset=offset_dataset(deltaX,deltaY,deltaZ,df)
            path1=os.path.join(fn,'CSV_Database')
            if not os.path.exists(path1):
                
                # If it doesn't exist, create the directory
                os.makedirs(path1)
                logging.info(f"Directory '{path1}' created successfully.")
            #else:
                #print(f"Directory '{path1}' already exists.")
            df_offset.to_csv(path1+'/'+ f'{variable}'+'_'+str(wind_dir)+'_Output_Transformed_all.csv', index=False)
            print(" ")
            print("Transformation completed. Output saved to:", path1+'/'+ f'{variable}'+'_'+str(wind_dir)+'_Output_Transformed_all.csv')      
    #Print time taken
        print(time.process_time() - start)
        
    return result_sample,df_offset,time.process_time() - start

def save_csv_reconstructed_all(grid2,variable,wind_dir,wind_speed,fn,vectorU):
    
    #print("Turbine located at around Y=1500, X=1000 in figure below")
    data = {'X': grid2.points[:, 0],    #cell_centers.points
            'Y': grid2.points[:, 1],
            'Z': grid2.points[:, 2],
        }
    df = pd.DataFrame(data)

#Suggest Time for reconstruction in arange below. 
#To reconstruct at all time-steps and save images, start loop from 0.

    if vectorU==False:        
        turbke=grid2["RECON_tke_at_WD_WS"+str(wind_dir)+"_"+str(wind_speed)]
        data2 = {'tke':turbke}
        df=df.join(pd.DataFrame(data2))
        df['tke'] = df['tke'].clip(lower=0.001) 
        
    else:
        vel=grid2["RECON_U_at_WD_WS"+str(wind_dir)+"_"+str(wind_speed)]  #tke_at_wind_direction.reshape(-1, 3, order='F')
        data2 = {'Velocity_X_': vel[:, 0],'Velocity_Y_': vel[:, 1],'Velocity_Z_': vel[:, 2]}
        df=df.join(pd.DataFrame(data2))            
        
       
        # saveimage(i)

# Save CSV  
    path1=os.path.join(fn,'CSV_Database')
    if not os.path.exists(path1):
        
        # If it doesn't exist, create the directory
        os.makedirs(path1)
        print(f"Directory '{path1}' created successfully.")
    else:
        logging.info(f"Directory '{path1}' already exists.")
        #print(f"Directory '{path1}' already exists.")

       
         
    csv_file = path1+'/'+ f'{variable}'+'_'+str(wind_dir)+'_'+str(wind_speed)+'_Output_all.csv'     
    df.to_csv(csv_file, index=False)
    print(f"CSV file saved successfully: {csv_file}")

    return df 
#Save Grid
#grid2.save(fn+'/Velocity_reconstruction_stored.vtk')
#grid2.save(fn+'/Velocity_reconstruction_stored.vtu') 
    


def visualize_plot(df,df_3D,wind_direc,wind_speed,grid): # arguement : wind_direc for angles in quiver.
    #df=df.to_frame()     
# Plot 3D terrain map
    print(wind_direc)
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X-Relative Distance from vertiport,m')
    ax.set_ylabel('Y-Relative Distance from vertiport,m')
    ax.set_zlabel('Z-Altitude,m')
    ax.set_xlim(-450, 450)  # Set limits for x direction
    ax.set_ylim(-450, 450)  # Set limits for y direction
    ax.set_zlim(267,500)  # Set limits for z direction
    #ax.plot(x, y, z, 'gray')  # Plot terrain map

# Plot drone's trajectory with colored line representing turbulence
    # for i in range(df.shape[0]):
    #    ax.scatter(0, 0, 267)
    #    ax.scatter(df['X'].iloc[i], df['Y'].iloc[i], df['Z'].iloc[i], c=df['tke'].iloc[i],cmap='viridis') #  plt.cm.jet(df['tke'].iloc[i]) , y is latitude, X changes with longitude.
    ax.scatter(0, 0, 267,color='grey',marker='^')
    ax.text(10, 10, 275, 'vertiport', color='red',ha='center', va='center')  
    
    df['Velocity_Magnitude']=np.sqrt(df['Velocity_X_']**2+  df['Velocity_Y_']**2+df['Velocity_Z_']**2)
    scatter = ax.scatter(df['X'],df['Y'],df['Z'], c=df['Velocity_Magnitude'], cmap='viridis')
    cbar = fig.colorbar(scatter)
    cbar.set_label('velocity, m/s')  
    tooltip = mplcursors.cursor(scatter, hover=True)
    tooltip.connect("add", lambda sel: sel.annotation.set_text(labels[sel.target.index]))
    
    
    
    
# Plot vectors at specified points using quiver
    # ax.quiver(df_3D['X'],df_3D['Y'],df_3D['Z'],df_3D['Velocity_X_'],df_3D['Velocity_Y_'],df_3D['Velocity_Z_'],length=15, normalize=True) #ax.quiver(x, y, z, u, v, w)
    
   
# Set plot title and labels
    #ax.set_title('Vector Plot using Quiver')
    #ax.set_xlabel('X')
    #ax.set_ylabel('Y')

    plt.show()
    plt.savefig('vel.png')
    
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X-Relative Distance from vertiport,m')
    ax.set_ylabel('Y-Relative Distance from vertiport,m')
    ax.set_zlabel('Z-Altitude,m')
    ax.set_xlim(-450, 450)  # Set limits for x direction
    ax.set_ylim(-450, 450)  # Set limits for y direction
    ax.set_zlim(267,500)  # Set limits for z direction
    #ax.plot(x, y, z, 'gray')  # Plot terrain map

# Plot drone's trajectory with colored line representing turbulence
    # for i in range(df.shape[0]):
    #    ax.scatter(0, 0, 267)
    #    ax.scatter(df['X'].iloc[i], df['Y'].iloc[i], df['Z'].iloc[i], c=df['tke'].iloc[i],cmap='viridis') #  plt.cm.jet(df['tke'].iloc[i]) , y is latitude, X changes with longitude.
    ax.scatter(0, 0, 267,color='grey',marker='^')
    ax.text(10, 10, 275, 'vertiport', color='red',ha='center', va='center') 
    # ax.text(0, 0, 267, 'vertiport', color='red')    
    scatter = ax.scatter(df['X'],df['Y'],df['Z'], c=df['tke'], cmap='viridis')
    cbar = fig.colorbar(scatter)
    cbar.set_label('tke')     
    
    

    # Add color bar
    #cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='jet'))
    #cbar.set_label('Turbulence')

    plt.show()
    plt.savefig('tke.png')
    
    #vel=grid2["RECON_U_at_WD_WS"+str(wind_dir)+"_"+str(wind_speed)]
    #fig = plt.figure(figsize=(12,12))
    print(" Dataframe : min and Max Altitude")
    print(df['Z'].iloc[0])
    print(df_3D['Z'].min())
    print(df_3D['Z'].max())
    print(" ")
    
    altitude = grid.points[:, 2] 
    print(" Grid : min and Max Altitude")
    print(altitude.min())
    print(altitude.max())
    
    grid.point_data["Alt"]=altitude
    
    print(df.describe())
    
    print(grid)
    
    slice_z = grid.slice(normal='z',origin=(df['X'].iloc[0], df['Y'].iloc[0], 268))
    #clip_plane = vtki.Plane(normal=['Z'], origin=[df['X'].iloc[0], df['Y'].iloc[0], df['Z'].iloc[0]])
    #clipped_surface = grid.clip(clip_plane)
    clipped_surface=slice_z
    #print(clipped_surface)
    glyph = clipped_surface.glyph(scale='RECON_U_at_WD_WS'+str(wind_direc)+"_"+str(wind_speed), orient='RECON_U_at_WD_WS'+str(wind_direc)+"_"+str(wind_speed),factor=10)
    print(glyph)
# Plot the result
    p = vtki.Plotter(notebook=True)
    #p.add_mesh(clipped_surface, color='lightblue', show_edges=True)
    p.add_mesh(glyph,cmap='coolwarm')
    
    
    #p.add_mesh(
    #grid,
    #scalars="Alt",
    #cmap="terrain",
    #show_scalar_bar=False,)
    p.show()
    
    return grid
    

import math
import os
#import matplotlib.pyplot as plt #for plotting
#from scipy.interpolate import Rbf #for regression.
import pyvista as vtki #for dealing with vtk geometry file i.e. Utest_sample.vtki
#import numpy as np
#import os, sys
import pandas as pd #make and work with dataframes.
##import vtk
##from os.path import isdir, isfile, join
##from os.path import expanduser
##import json
import time 
##import random
#import time as timer
import pickle #
#import trame
#import vtk
#import trame_vtk



# %run '/media/mandart/D/AI4Hydrop_Prague/ROM/Results/data_POD/libimport.py'

def offset_dataset(deltaX,deltaY,deltaZ,df):

    # Load the dataset into a pandas DataFrame
    # Adjust the path to your dataset CSV file
    #df = pd.read_csv(dataset_path)

    # Define the offsets for each dimension
    # Change this value to the desired offset for Z

    # Apply the offsets to the X, Y, and Z columns
    df['X'] = df['X'] + deltaX
    df['Y'] = df['Y'] + deltaY
    df['Z'] = df['Z'] + deltaZ

    # Save the modified DataFrame back to a new CSV file
       # Adjust the path for the output CSV file
    #df.to_csv(output_path, index=False)

    
    return df

#***************************************************************************************************************************************************'

def save_csv_reconstructed (grid2,variable,wind_dir,fn,vectorU):     
    #print("Turbine located at around Y=1500, X=1000 in figure below")
    data = {'X': grid2.points[:, 0],    #cell_centers.points
            'Y': grid2.points[:, 1],
            'Z': grid2.points[:, 2],
        }
    df = pd.DataFrame(data)

#Suggest Time for reconstruction in arange below. 
#To reconstruct at all time-steps and save images, start loop from 0.

    if vectorU==False:        
        turbke=grid2["RECON_tke_at_WD_"+str(wind_dir)]
        data2 = {'tke':turbke}
        df=df.join(pd.DataFrame(data2))
    else:
        vel=grid2["RECON_U_at_WD_"+str(wind_dir)]  #tke_at_wind_direction.reshape(-1, 3, order='F')
        data2 = {'Velocity_X_': vel[:, 0],'Velocity_Y_': vel[:, 1],'Velocity_Z_': vel[:, 2]}
        df=df.join(pd.DataFrame(data2))            
        
       
        # saveimage(i)

# Save CSV    
    csv_file = fn+'/'+ f'{variable}'+'_'+str(wind_dir)+'_Output.csv'     
    df.to_csv(csv_file, index=False)
    print(f"CSV file saved successfully: {csv_file}")

    return df 

# #***************************************************************************************************************************************************'

def meters_to_decimal_degrees(delta_x, delta_y, reference_lat):
    import math
    # Earth's radius at the reference latitude (average of meridional and equatorial radii)
    earth_radius = 6371000  # in meters

    # Convert meters to degrees for longitude
    delta_lon = (delta_x / earth_radius) * (180 / (math.pi * math.cos(math.radians(reference_lat))))

    # Convert meters to degrees for latitude
    delta_lat = (delta_y / earth_radius) * (180 / math.pi)

    return delta_lat, delta_lon

# Example usage
#delta_x = 1000  # 1000 meters eastward
#delta_y = -500  # 500 meters southward
#reference_lat = 40.7128  # Reference latitude in decimal degrees (New York City)

#delta_lat, delta_lon = meters_to_decimal_degrees(delta_x, delta_y, reference_lat)
#print("Relative change in Latitude: {:.6f} degrees".format(delta_lat))
#print("Relative change in Longitude: {:.6f} degrees".format(delta_lon))


# #***************************************************************************************************************************************************'

# Method 2. The conversion from meters to decimal degrees is based on the following formulae: 
# 1 degree of latitude is approximately 111,139 meters.
# 1 degree of longitude is approximately 111,139 meters * cos(latitude).
# Define a function to convert meters to decimal degrees, latitude in decimal degree
def convert_meters_to_decimal_degrees(x, y, latitude):
    # Convert meters to decimal degrees
    x_deg = x / (111139 * math.cos(math.radians(latitude)))
    y_deg = y / 111139

    return x_deg, y_deg

# Test the function with some example values
#x_meters = 1000  # 1000 meters East
#y_meters = 2000  # 2000 meters North
#latitude = 45  # latitude in decimal degrees

#x_deg, y_deg = convert_meters_to_decimal_degrees(x_meters, y_meters, latitude)

#print(f"X: {x_meters} meters = {x_deg} degrees")
#print(f"Y: {y_meters} meters = {y_deg} degrees")

# #***************************************************************************************************************************************************'

def reconstruct_AI_for_winddirection(variable,wind_direc,fn,vectorU=True):     
    deltaX=-1480
    deltaY=2120
    deltaZ=229  
    
    #import time    
    start = time.process_time()
    # your code here    
    # Assume a wind direction 
    # wind_dir=np.rand270 #in degrees
    folder = 'data_POD'
    
    # Load the Rbf interpolators from a file
    print(fn+'/'+f"{variable}"+'_rbf_interpolators.pkl')
    with open(fn+'/'+f"{variable}"+'_rbf_interpolators.pkl', 'rb') as file:
        loaded_interpolators = pickle.load(file)
        
        
    #Obtain basis functions : Load saved basis functions (modes) and mean turbulent kinetic energy
    #----------------------------------------------------------------------------------------------.     
    filename = fn+'/POD_data_' +f"{variable}"+'.npz'    
        
    PODdata=np.load(filename)

    Phit=PODdata['tbasis']
    tke_mean=PODdata['tmean']   
     
    result_sample=vtki.read('Utest_sample.vtk') 
    
    for wind_dir in [wind_direc]:     
        
        #Obtain coefficient for this wind direction
        #------------------------------------------------------------------  
        # Interpolate using the loaded interpolators (for demonstration)
        predicted_coef = []
        max_value=360
        xsin_winddir=np.sin(2 * np.pi * wind_dir / max_value)
        xcos_winddir=np.cos(2 * np.pi * wind_dir / max_value)
        for i, rbf1 in enumerate(loaded_interpolators):
            predicted_coef.append(rbf1(xsin_winddir,xcos_winddir))
            print('Coeff for mode', i+1 , 'is ', rbf1(xsin_winddir,xcos_winddir))
        print(' ')
         
       #Reconstruct flow field from the basis modes, the mean and the computed coefficients
        #----------------------------------------------------------------------------------------------------
        tke_at_wind_direction=np.dot(Phit,np.array(predicted_coef))+tke_mean

        #Visualize reconstructed flow field.
        #----------------------------------------------------------------------------------------------------
        if vectorU==False:
            result_sample.point_data["RECON_tke_at_WD_"+str(wind_dir)]=tke_at_wind_direction
            df=save_csv_reconstructed(result_sample,variable,wind_dir,fn,vectorU)
            df['tke'] = df['tke'].clip(lower=0.001) 
            #df=pd.read_csv(csv_file)
            #Make vertiport at 0,0,267. x=x-1480,y=y+2120,z=z+229.
            
        
        else:
            U_reshaped=tke_at_wind_direction.reshape(-1, 3, order='F') 
            result_sample.point_data.set_vectors(U_reshaped,"RECON_U_at_WD_"+str(wind_dir))
            df=save_csv_reconstructed(result_sample,variable,wind_dir,fn,vectorU)
            
        result_sample.save('./Results/recon_added_' +f"{variable}"+'.vtk')
        
        #Offset grid to make vertiport the center 
        df_offset=offset_dataset(deltaX,deltaY,deltaZ,df)
        df_offset.to_csv(fn+'/'+ f'{variable}'+'_'+str(wind_dir)+'_Output_Transformed.csv', index=False)

        print("Reconstruction completed. Output saved to:", fn+'/'+ f'{variable}'+'_'+str(wind_dir)+'_Output_Transformed.csv')  
        print(' ')
            
    #Print time taken
    #print(time.process_time() - start)
    print(' ')    
    return result_sample,df_offset,time.process_time() - start

# #***************************************************************************************************************************************************'

def find_nearest_location(X, Y, Z,df):
    distances = np.sqrt((df['X'] - X)**2 + (df['Y'] - Y)**2 + (df['Z'] - Z)**2)
    nearest_index = distances.idxmin()
    print ('nearest',nearest_index,df.loc[nearest_index])
    return df.loc[nearest_index]

import math
def haversine(lat1, lon1, lat2, lon2):
    from math import sin, cos, sqrt, atan2, radians
    # Convert latitude and longitude from degrees to radians
    lat1=np.radians(lat1)
    lon1=np.radians(lon1)
    lat2=np.radians(lat2)
    lon2=np.radians(lon2)
    
    #radians(lon1), radians(lat2), radians(lon2)
    
    # Haversine formula
    dlat = lat2 - lat1
    
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    #print(a)
    
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = 6371 * c *1000 # Radius of Earth in meters
    #print(distance)
    return distance

def check_limits(X_or_longitude, Y_or_latitude, Z,):
    if df['Y'].min() <= Y_or_latitude <= df['Y'].max() and df['X'].min() <= X_or_longitude <= df['X'].max() and df['Z'].min() <= Z <= df['Z'].max():
        # Process the location
        logging.info("Latitude and longitude are within bounds. Proceeding with further processing.")
        # Further processing can be done here
    else:
        # Print a warning message
        print("Warning: Latitude and/or longitude are out of bounds. Please provide valid values or check input .")
        
def get_U_and_k_for_location(X_or_longitude, Y_or_latitude, Z,df,relative_distance_in_meter__or__Latlong='meter'): 
    nearest_location=[0 for _ in range(len(X_or_longitude))]
    for i in range(len(X_or_longitude)):
                       
        if relative_distance_in_meter__or__Latlong=='meter':
            if df['Y'].min() <= Y_or_latitude[i] <= df['Y'].max() and df['X'].min() <= X_or_longitude[i] <= df['X'].max() and df['Z'].min() <= Z[i] <= df['Z'].max():
                # Process the location
                logging.info("Coordinates are within bounds. Proceeding with further processing.")
                
                # Further processing can be done here
            else:
            # Print a warning message
                print("Warning: Coordinates are out of bounds. Please provide valid values or check input .")
                print(f"Relative Distance from Vertiport in North-south Y direction should be in between the range: {df['Y'].min()}and {df['Y'].max()}")
                print(f"Relative Distance from Vertiport in East-West X direction should be in between the range: {df['Y'].min()} and {df['Y'].max()}")
                return []
            location_data= df.index[(df['X'] == X_or_longitude[i]) & (df['Y'] == Y_or_latitude[i]) & (df['Z'] == Z[i])]
            #print(location_data,'type')
        else:
            if df['latitude'].min() <= Y_or_latitude[i] <= df['latitude'].max() and df['longitude'].min() <= X_or_longitude[i] <= df['longitude'].max() and df['Z'].min() <= Z[i] <= df['Z'].max():
                # Process the location
                logging.info("Latitude and longitude are within bounds. Proceeding with further processing.")
                # Further processing can be done here
            else:
            # Print a warning message
                print("Warning: Latitude and/or longitude are out of bounds. Please provide valid values or check input .")
                print(f"Relative Distance from Vertiport in Latitude Y direction should be in between the range :{df['latitude'].min()} and {df['latitude'].max()}")
                print(f"Relative Distance from Vertiport in Longitude X direction should be in between the range:{df['longitude'].min()} and {df['longitude'].max()}")
                return []
            location_data = df.index[(df['latitude'] == Y_or_latitude[i]) & (df['longitude'] == X_or_longitude[i]) & (df['Z'] == Z[i])]
            #print('index',location_data)      

        if len(location_data) == 0:                 
            if relative_distance_in_meter__or__Latlong=='meter':
                distances = np.sqrt((df['X'] - X_or_longitude[i])**2 + (df['Y'] - Y_or_latitude[i])**2 + (df['Z'] - Z[i])**2)
                print("using nearest based on distance") 
            else:
                distances = np.sqrt((df['longitude'] - X_or_longitude[i])**2 + (df['latitude'] - Y_or_latitude[i])**2 + (df['Z'] - Z[i])**2)
                #distances=haversine(df['latitude'],df['longitude'],Y,X) + np.sqrt(df['Z'] - Z)**2
                logging.info("using nearest based on lat-long. Haversten  not used.") 
                
            nearest_index = distances.idxmin()             
            # nearest_location[i] = df.loc[nearest_index] #find_nearest_location(X, Y, Z,df)
            nearest_location[i] = nearest_index #df.loc[nearest_index]
            #print('nearest',i,nearest_location[i],nearest_index)
            
            #vx=nearest_location['Velocity_X_']
            #vy=nearest_location['Velocity_Y_']
            #vz=nearest_location['Velocity_Z_']
            #Ucomponent=[vx,vy,vz]
            
            #U_values = math.sqrt(vx**2 + vy**2 + vz**2) 
            #k_values = nearest_location['tke']
            #return nearest_location #U_values, k_values, Ucomponent,
        else:
            #U_values = location_data['Velocity_X_']
            #k_values = location_data['k'].tolist()
            nearest_location[i] = location_data #df.loc[nearest_index] #find_nearest_location(X, Y, Z,df)
            #print('nearest',i,nearest_location[i],nearest_index)
        del location_data
    return df.loc[nearest_location]   
        
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import mplcursors


def reconstruct_AI_for_winddirection_windspeed(variable,wind_direc,windspeed,fn,vectorU=True):  
    
    deltaX=-1480
    deltaY=2120
    deltaZ=229  
    
    start = time.process_time()
    # your code here    
    # Assume a wind direction 
    # wind_dir=np.rand270 #in degrees
    folder = 'data_POD'
    
    # Load the Rbf interpolators from a file
    # print(fn+'/'+f"{variable}"+'_rbf_interpolators_all.pkl')
    with open(fn+'/'+f"{variable}"+'_rbf_interpolators_oslo_all.pkl', 'rb') as file:
        loaded_interpolators = pickle.load(file)
        
        
    #Obtain basis functions : Load saved basis functions (modes) and mean turbulent kinetic energy
    #----------------------------------------------------------------------------------------------.     
    filename = fn+'/POD_data_oslo_all' +f"{variable}"+'.npz'    
        
    PODdata=np.load(filename)

    Phit=PODdata['tbasis']
    tke_mean=PODdata['tmean']    
    
    result_sample=vtki.read('Utest_sample.vtk')

    for wind_dir in [wind_direc]:
        for wind_spee in [windspeed]:     
            #print (f"wind speed {wind_spee} ")
            #print (f"wind direction {wind_dir}")
            #Obtain coefficient for this wind direction
            #------------------------------------------------------------------  
            # Interpolate using the loaded interpolators (for demonstration)
            predicted_coef = []
            max_value=360
            xsin_winddir=np.sin(2 * np.pi * wind_dir / max_value)
            xcos_winddir=np.cos(2 * np.pi * wind_dir / max_value)
            for i, rbf1 in enumerate(loaded_interpolators):
                predicted_coef.append(rbf1(xsin_winddir,xcos_winddir,wind_spee))
                # print('Coeff for mode', i+1 , 'is ', rbf1(xsin_winddir,xcos_winddir,wind_spee))

        #Reconstruct flow field from the basis modes, the mean and the computed coefficients
            #----------------------------------------------------------------------------------------------------
            tke_at_wind_direction=np.dot(Phit,np.array(predicted_coef))+tke_mean

            #Visualize reconstructed flow field.
            #----------------------------------------------------------------------------------------------------
            if vectorU==False:
                result_sample.point_data["RECON_tke_at_WD_WS"+str(wind_dir)+"_"+str(wind_spee)]=tke_at_wind_direction
                df=save_csv_reconstructed_all(result_sample,variable,wind_dir,wind_spee,fn,vectorU) #(grid2,variable,wind_dir,wind_speed,fn,vectorU)
                #df=pd.read_csv(csv_file)
                #Make vertiport at 0,0,267. x=x-1480,y=y+2120,z=z+229.               
            
            else:
                U_reshaped=tke_at_wind_direction.reshape(-1, 3, order='F') 
                result_sample.point_data.set_vectors(U_reshaped,"RECON_U_at_WD_WS"+str(wind_dir)+"_"+str(wind_spee))
                df=save_csv_reconstructed_all(result_sample,variable,wind_dir,wind_spee,fn,vectorU)
                
            path1=os.path.join(fn,'VTK_Database')
            if not os.path.exists(path1):
                
                # If it doesn't exist, create the directory
                os.makedirs(path1)
                logging.info(f"Directory '{path1}' created successfully.")
            else:
                logging.info(f"Directory '{path1}' already exists.")
                # print(f"Directory '{path1}' already exists.") 
                   
            result_sample.save(path1+'/Recon_VTK_added_all_ws_wd' +f"{variable}"+'.vtk')
            print(" ")
            print("Done. VTK reconstruction saved at:", path1+'/Recon_VTK_added_all_ws_wd' +f"{variable}"+'.vtk')      
            #Offset grid to make vertiport the center 
            df_offset=offset_dataset(deltaX,deltaY,deltaZ,df)
            path1=os.path.join(fn,'CSV_Database')
            if not os.path.exists(path1):
                
                # If it doesn't exist, create the directory
                os.makedirs(path1)
                logging.info(f"Directory '{path1}' created successfully.")
            else:
                logging.info(f"Directory '{path1}' already exists.")
                # print(f"Directory '{path1}' already exists.")
            df_offset.to_csv(path1+'/'+ f'{variable}'+'_'+str(wind_dir)+'_Output_Transformed_all.csv', index=False)
            print(" ")
            print("Transformation completed. Output saved to:", path1+'/'+ f'{variable}'+'_'+str(wind_dir)+'_Output_Transformed_all.csv')      
    #Print time taken
        print(time.process_time() - start)
        
    return result_sample,df_offset,time.process_time() - start

def save_csv_reconstructed_all(grid2,variable,wind_dir,wind_speed,fn,vectorU):
    
    #print("Turbine located at around Y=1500, X=1000 in figure below")
    data = {'X': grid2.points[:, 0],    #cell_centers.points
            'Y': grid2.points[:, 1],
            'Z': grid2.points[:, 2],
        }
    df = pd.DataFrame(data)

#Suggest Time for reconstruction in arange below. 
#To reconstruct at all time-steps and save images, start loop from 0.

    if vectorU==False:        
        turbke=grid2["RECON_tke_at_WD_WS"+str(wind_dir)+"_"+str(wind_speed)]
        data2 = {'tke':turbke}
        df=df.join(pd.DataFrame(data2))
        df['tke'] = df['tke'].clip(lower=0.001) 
        
    else:
        vel=grid2["RECON_U_at_WD_WS"+str(wind_dir)+"_"+str(wind_speed)]  #tke_at_wind_direction.reshape(-1, 3, order='F')
        data2 = {'Velocity_X_': vel[:, 0],'Velocity_Y_': vel[:, 1],'Velocity_Z_': vel[:, 2]}
        df=df.join(pd.DataFrame(data2))            
        
       
        # saveimage(i)

# Save CSV  
    path1=os.path.join(fn,'CSV_Database')
    if not os.path.exists(path1):
        
        # If it doesn't exist, create the directory
        os.makedirs(path1)
        print(f"Directory '{path1}' created successfully.")
    else:
        logging.info(f"Directory '{path1}' already exists.")

       
         
    csv_file = path1+'/'+ f'{variable}'+'_'+str(wind_dir)+'_'+str(wind_speed)+'_Output_all.csv'     
    df.to_csv(csv_file, index=False)
    print(f"CSV file saved successfully: {csv_file}")

    return df 
#Save Grid
#grid2.save(fn+'/Velocity_reconstruction_stored.vtk')
#grid2.save(fn+'/Velocity_reconstruction_stored.vtu') 

def visualize_slice(df,df_3D,wind_direc,wind_speed,grid,sliceloc): # arguement : wind_direc for angles in quiver.
   
    
    df['Velocity_Magnitude']=np.sqrt(df['Velocity_X_']**2+  df['Velocity_Y_']**2+df['Velocity_Z_']**2)
    
    #print(" Dataframe : min and Max Altitude")
    #print(df['Z'].iloc[0])
    #print(df_3D['Z'].min())
    #print(df_3D['Z'].max())
    #print(" ")
    
    altitude = grid.points[:, 2] 
    print(" Grid : min and Max Altitude")
    print(altitude.min())
    print(altitude.max())
    
    grid.point_data["Alt"]=altitude
    
    #print(df.describe())
    
    #print(grid)
    
    slice_z = grid.slice(normal='z',origin=(df['X'].iloc[0], df['Y'].iloc[0],sliceloc))
    #clip_plane = vtki.Plane(normal=['Z'], origin=[df['X'].iloc[0], df['Y'].iloc[0], df['Z'].iloc[0]])
    #clipped_surface = grid.clip(clip_plane)
    clipped_surface=slice_z
    #print(clipped_surface)
    glyph = clipped_surface.glyph(scale='RECON_U_at_WD_WS'+str(wind_direc)+"_"+str(wind_speed), orient='RECON_U_at_WD_WS'+str(wind_direc)+"_"+str(wind_speed),factor=10)
    #print(glyph)
# Plot the result
    p = vtki.Plotter(notebook=True)
    #p.add_mesh(clipped_surface, color='lightblue', show_edges=True)
    p.add_mesh(glyph,cmap='coolwarm')
    
    
    #p.add_mesh(
    #grid,
    #scalars="Alt",
    #cmap="terrain",
    #show_scalar_bar=False,)
    p.show()
    
    return grid
    
def demo(date_and_hrs_only,random_coordinates,fn2,API_KEY = 'fa3e3882b6508618a835169fb753d745',visualization=False): 
    Macroscale_ws_wd_prediction= pd.DataFrame(columns=['lat', 'lon','ws','wd']) 
    
    # random_coordinates=[(59.9494,10.7156)]   
    for idx, (lat, lon) in enumerate(random_coordinates):
        # print(f"Coordinate {idx}: Latitude = {lat}, Longitude = {lon}")
        url=f'https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}'
        # Make API request
        #url=f'http://api.openweathermap.org/data/2.5/forecast?id=524901&appid={API_KEY}'
        #url = f'https://api.openweathermap.org/data/2.5/onecall/timemachine?lat=50.0755&lon=14.4378&dt={date}&appid={API_KEY}'
        response = requests.get(url)

        # url=f'https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&exclude={part}&appid={API_KEY}'
        if response.status_code == 200:
            data = response.json()
            # print(data)
            #for forecast in data['list']:  
                # print(forecast)    #forecast is a list within data[list].  
                #if forecast['dt_txt'].split()[0] == dateonly:
                    #print(forecast['dt_txt'])

                #print(forecast['dt_txt']) #outputs date_time data
                #if forecast['dt_txt']==date_and_hrs_only:
                    #print(forecast['dt_txt'])


            #For date only
            #target_forecast = [forecast for forecast in data['list'] if forecast['dt_txt'].split()[0] == dateonly]

            #For date and hrs.
            target_forecast = [forecast for forecast in data['list'] if forecast['dt_txt'] == date_and_hrs_only]
            #print("target forecast is a new list of lists.")
            #print(target_forecast)

            if target_forecast:
                #print("out of " , len(target_forecast),f" hourly forecasts on the date : {dateonly}" )
                #print(f"Selecting wind speed and wind direction for time : {target_forecast[3]['dt_txt']}")
                ws = [forecast['wind']['speed'] for forecast in target_forecast]
                deg = [forecast['wind']['deg'] for forecast in target_forecast]
                #print(ws)
                #print(deg)
                Macroscale_ws_wd_prediction=pd.concat([Macroscale_ws_wd_prediction,pd.DataFrame({'lat': lat, 'lon': lon, 'ws': ws, 'wd':deg})],ignore_index=True)
                #average_ws = sum(ws) / len(ws)
                #print(f"Average ws on {dateonly}: {average_ws} mps")  

            else:
                print(f"No forecast available for {date_and_hrs_only}")
                print(f"Select NEW DATES in future and run again")
                return None, None, None,None,None
            # Extract wind speed and direction
            #wind_speedx = data['list'][0] #['wind']['speed']
            #print(wind_speedx) #['wind']['speed']))
            #wind_direction = data['current']['wind_deg']
            #print(f"Wind Speed on {date}: {wind_speed} m/s")
            #print(f"Wind Direction on {date}: {wind_direction} degrees")
        else:
            print("Error fetching data:", response.status_code)
            print("Check and Get new API KEY. Run again")
        
    if not os.path.exists(fn2):
        
    # If it doesn't exist, create the directory
        os.makedirs(directory_path)
        print(f"Directory '{fn2}' created successfully.")
        print(" ")
    else:
        logging.info(f"Directory '{fn2}' already exists.")
        print(" ")

    #path1=os.path.join(fn2,'libimportoslo.py')   
    #path2=os.path.join(fn2,'funcimport.py')

# Step . Ask for user inputs - wind direction in degrees, locations in latitude/longitude.
# Ask for user inputs

    user_input_at_command=False

    if user_input_at_command==True:
        
        wind_direc = float(input("Enter single wind direction in degree: "))
        
        wind_speed = float(input("Enter  inlet wind speed in mps in region between 0.5 to 8 mps:"))
        
        latitude_input = input("Enter list of latitudes (Y) in Decimal Degree(in comma-separated in range between 50.04585949932427 and 50.053957709673476) : example - 50.05,50.047 ")

        longitude_input = input("Enter list of longitudes (X) in Decimal Degree (in comma-separated in range between 14.430968123655077 and 14.444980676344922): 14.44,14.431")

        altitude_input = input("Enter list of altitudes in m 267m>Z>500m (comma-separated): 270,300")
        
        longitude_list = [float(lon) for lon in longitude_input.split(',')]
        altitude_list = [float(alt) for alt in altitude_input.split(',')]
        latitude_list = [float(lat) for lat in latitude_input.split(',')]
        
    else:
        wind_direc=deg[0] #90
        wind_speed=np.max([np.min([ws[0],8]),0.5]) #np.min(ws[0],8) #np.max([np.min(ws[0],8),0.5])
        print(f"Inputs obtained from Meso-scale open_weather_map for {date_and_hrs_only} is :")
        print(f"wind direction is {wind_direc} degrees and wind speed is {wind_speed} mps")
        print(" ")
        
        lat_long_needed=True
        if lat_long_needed==True:
            latitude_list=Macroscale_ws_wd_prediction['lat'].values #[50.05,50.047]
            longitude_list=Macroscale_ws_wd_prediction['lon'].values #[14.44,14.431]
            altitude_list=[300]*Macroscale_ws_wd_prediction['lat'].shape[0] #[270,300] 
        #Check if all points are within the range . Keep only those that are in the range.count
    

            # Print the inputs for verification
            if len(latitude_list) != len(longitude_list) or len(latitude_list) != len(altitude_list):
                print("Error: The lists of latitude, longitude, and altitude must have the same length.")
            #else:
                #print("Wind direction:", wind_direc)
                #print("Latitude list:", latitude_list)
                #print("Longitude list:", longitude_list)
                #print("Altitude list:", altitude_list)


    # Step . Reconstruct data using wind direction.
    print("  ")
    print("COMPUTING: AI model is obtaining Wind Velocity and Turbulence in the city region (micro-scale predictions) using the input meso-scale wind direction and wind-speed for selected data and time.")
    print("  ")
    result_samples_k,dataframe_k,timetaken_k=reconstruct_AI_for_winddirection_windspeed("k",wind_direc,wind_speed,fn2,vectorU=False) #nr not needed.
    result_samples_U,dataframe_U,timetaken_U=reconstruct_AI_for_winddirection_windspeed("U",wind_direc,wind_speed,fn2,vectorU=True)  #nr not neeeded. 
    merged_dataframe_U_k_Relative_2_vertiport_reconstructed = pd.merge(dataframe_k,dataframe_U, on=['X','Y','Z'], how='outer')
    print(f'Done. Total time taken for reconstruction - {timetaken_U+timetaken_k} s')
    print("  ")
    lat_long_needed=True
    if lat_long_needed==True:      
        ref_long_at_vertiport=10.7156
        ref_latitude_at_vertiport=59.9494
        #Latitude is the Y axis, longitude is the X axis. Belo since X,Y are relative to vertiport. Hence, we use this to obtain relative change in latitude/long from vertiport. then add reference to this change to get actual latitude and longitude.
        merged_dataframe_U_k_Relative_2_vertiport_reconstructed['latitude']=(merged_dataframe_U_k_Relative_2_vertiport_reconstructed['Y'] / 111139)+ref_latitude_at_vertiport
        lat_rad= math.radians(ref_latitude_at_vertiport) #merged_df_U_k_Relative_2_vertiport_reconstructed['latitude'].apply(math.radians)
        #lat_rad.apply(math.cos)
        cos_lat_rad=math.cos(lat_rad)
        merged_dataframe_U_k_Relative_2_vertiport_reconstructed['longitude']=(merged_dataframe_U_k_Relative_2_vertiport_reconstructed['X'] / (111139 * cos_lat_rad))+ref_long_at_vertiport
        

    merged_dataframe_U_k_Relative_2_vertiport_reconstructed["Velocity_mag"]=np.sqrt(merged_dataframe_U_k_Relative_2_vertiport_reconstructed["Velocity_X_"]**2+\
                                                                                    merged_dataframe_U_k_Relative_2_vertiport_reconstructed["Velocity_Y_"]**2 + \
                                                                                        merged_dataframe_U_k_Relative_2_vertiport_reconstructed["Velocity_Z_"]**2)

    merged_dataframe_U_k_Relative_2_vertiport_reconstructed['Normalized_Velocity']=(merged_dataframe_U_k_Relative_2_vertiport_reconstructed["Velocity_mag"]-merged_dataframe_U_k_Relative_2_vertiport_reconstructed["Velocity_mag"].min())/\
        (merged_dataframe_U_k_Relative_2_vertiport_reconstructed["Velocity_mag"].max()-merged_dataframe_U_k_Relative_2_vertiport_reconstructed["Velocity_mag"].min())


    merged_dataframe_U_k_Relative_2_vertiport_reconstructed['Normalized_turbulence']=(merged_dataframe_U_k_Relative_2_vertiport_reconstructed["tke"]-merged_dataframe_U_k_Relative_2_vertiport_reconstructed["tke"].min())/\
        (merged_dataframe_U_k_Relative_2_vertiport_reconstructed["tke"].max()-merged_dataframe_U_k_Relative_2_vertiport_reconstructed["tke"].min())
    

    #merged_dataframe_U_k_Relative_2_vertiport_reconstructed['normalized_U']=merged_dataframe_U_k_Relative_2_vertiport_reconstructed

    # Step. Obtain values at specific points.
    lat_long_needed=True
    if lat_long_needed==True: 
        print("  ")
        print("Obtaining values at drone trajectory points from the reconstructed field.")
        print("  ")
        print(merged_dataframe_U_k_Relative_2_vertiport_reconstructed.columns)
        wind_data_near_drone_trajectory =get_U_and_k_for_location(longitude_list,latitude_list,altitude_list,merged_dataframe_U_k_Relative_2_vertiport_reconstructed,relative_distance_in_meter__or__Latlong='relat')
        #print(wind_data_near_drone_trajectory)
        
        print(" ")
        print("1. Done. In python, see DataFrame variable - wind_data_near_drone_trajectory . It shows  wind velocity and turbulence at chosen drone trajectory locations.")
        print("2. For more information: See Saved CSV files for  wind velocity and turbulence at chosen drone trajectory locations.")
        print(" ")
        print("2. For more information: See Saved VTK file for in Paraview or pyVista to visualize the reconstructed flow field")
        print(" ")


    print("Obtaining locations in Oslo with high TURBULENCE and high WIND using a threhold of 0.9 times normalized values.")
    threshold=0.9
    locations_at_high_turbulence=merged_dataframe_U_k_Relative_2_vertiport_reconstructed[merged_dataframe_U_k_Relative_2_vertiport_reconstructed['Normalized_Velocity']>threshold]
    locations_at_high_wind_magnitude=merged_dataframe_U_k_Relative_2_vertiport_reconstructed[merged_dataframe_U_k_Relative_2_vertiport_reconstructed['Normalized_turbulence']>threshold]
    print("High_turbulence locations obtained....")
    print(" ")
    #print(locations_at_high_turbulence)
    print("For more information: See contents of the variable: locations_at_high_turbulence ")
    print(" ")
    print("High_WIND locations obtained....")
    print(" ")
    #print(locations_at_high_wind_magnitude)
    print("or more information: See contents of the variable: locations_at_high_wind_magnitude ")
    print(" ")
    
    print("obtain locations near max turbulence limit")
    value=merged_dataframe_U_k_Relative_2_vertiport_reconstructed['tke'].max()
    diff = (merged_dataframe_U_k_Relative_2_vertiport_reconstructed['tke']- value).abs() # Find the column-wise absolute difference from the provided value
    nearest_index = diff.idxmin() # Find the index (row label) where the minimum absolute difference occurs 
    nearest_values = merged_dataframe_U_k_Relative_2_vertiport_reconstructed.iloc[nearest_index] # # Get the values at these indices

    #print("Nearest location for", value, "is", nearest_values)  

    #visualization='True'
    if visualization=='True':
        print("  ")
        print("Ignore notebook backend error and :")
        print("Visualize the micro-scale flow vectors in city centre as generated by AI in image.")
        print("  ")
        stride=10
        slice_location=49
        visualize_slice(wind_data_near_drone_trajectory,merged_dataframe_U_k_Relative_2_vertiport_reconstructed[::stride],wind_direc,wind_speed,result_samples_U,slice_location)

    
    if lat_long_needed==True:
        
        print(wind_data_near_drone_trajectory.columns)
        
        columnsTitles=['X', 'Y', 'Z','latitude','longitude','Velocity_mag','Normalized_Velocity','tke','Normalized_turbulence','Velocity_X_', 'Velocity_Y_','Velocity_Z_']

        wind_data_near_drone_trajectory = wind_data_near_drone_trajectory.reindex(columns=columnsTitles)            
    
    else:
        wind_data_near_drone_trajectory=[]
        
        #
    
    return Macroscale_ws_wd_prediction, wind_data_near_drone_trajectory,locations_at_high_turbulence,locations_at_high_wind_magnitude,merged_dataframe_U_k_Relative_2_vertiport_reconstructed
    

