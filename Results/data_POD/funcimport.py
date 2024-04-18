
#%run '/media/mandart/D/AI4Hydrop_Prague/ROM/Results/data_POD/libimport.py'

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
    import time    
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

    for wind_dir in wind_direc:     
        
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

       #Reconstruct flow field from the basis modes, the mean and the computed coefficients
        #----------------------------------------------------------------------------------------------------
        tke_at_wind_direction=np.dot(Phit,np.array(predicted_coef))+tke_mean

        #Visualize reconstructed flow field.
        #----------------------------------------------------------------------------------------------------
        if vectorU==False:
            result_sample.point_data["RECON_tke_at_WD_"+str(wind_dir)]=tke_at_wind_direction
            df=save_csv_reconstructed(result_sample,variable,wind_dir,fn,vectorU)
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

        print("Transformation completed. Output saved to:", fn+'/'+ f'{variable}'+'_'+str(wind_dir)+'_Output_Transformed.csv')      
    #Print time taken
    print(time.process_time() - start)
        
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
        print("Latitude and longitude are within bounds. Proceeding with further processing.")
        # Further processing can be done here
    else:
        # Print a warning message
        print("Warning: Latitude and/or longitude are out of bounds. Please provide valid values or check input .")
        
def get_U_and_k_for_location(X_or_longitude, Y_or_latitude, Z,df,relative_distance_in_meter__or__Latlong='meter'):                
    if relative_distance_in_meter__or__Latlong=='meter':
        if df['Y'].min() <= Y_or_latitude <= df['Y'].max() and df['X'].min() <= X_or_longitude <= df['X'].max() and df['Z'].min() <= Z <= df['Z'].max():
            # Process the location
            print("Coordinates are within bounds. Proceeding with further processing.")
            
            # Further processing can be done here
        else:
        # Print a warning message
            print("Warning: Coordinates are out of bounds. Please provide valid values or check input .")
            print(f"Relative Distance from Vertiport in North-south Y direction should be in between the range: {df['Y'].min()}and {df['Y'].max()}")
            print(f"Relative Distance from Vertiport in East-West X direction should be in between the range: {df['Y'].min()} and {df['Y'].max()}")
            return []
        location_data = df[(df['X'] == X_or_longitude) & (df['Y'] == Y_or_latitude) & (df['Z'] == Z)]
    else:
        if df['latitude'].min() <= Y_or_latitude <= df['latitude'].max() and df['longitude'].min() <= X_or_longitude <= df['longitude'].max() and df['Z'].min() <= Z <= df['Z'].max():
            # Process the location
            print("Latitude and longitude are within bounds. Proceeding with further processing.")
            # Further processing can be done here
        else:
        # Print a warning message
            print("Warning: Latitude and/or longitude are out of bounds. Please provide valid values or check input .")
            print(f"Relative Distance from Vertiport in Latitude Y direction should be in between the range :{df['latitude'].min()} and {df['latitude'].max()}")
            print(f"Relative Distance from Vertiport in Longitude X direction should be in between the range:{df['longitude'].min()} and {df['longitude'].max()}")
            return []
        location_data = df[(df['latitude'] == Y_or_latitude) & (df['longitude'] == X_or_longitude) & (df['Z'] == Z)]
        print(location_data)      

    if len(location_data) == 0:                 
        if relative_distance_in_meter__or__Latlong=='meter':
            distances = np.sqrt((df['X'] - X_or_longitude)**2 + (df['Y'] - Y_or_latitude)**2 + (df['Z'] - Z)**2)
            print("using nearest based on distance") 
        else:
            distances = np.sqrt((df['longitude'] - X_or_longitude)**2 + (df['latitude'] - Y_or_latitude)**2 + (df['Z'] - Z)**2)
            #distances=haversine(df['latitude'],df['longitude'],Y,X) + np.sqrt(df['Z'] - Z)**2
            print("using nearest based on lat-long. Haversten  not used.") 
            
        nearest_index = distances.idxmin()             
        nearest_location = df.loc[nearest_index] #find_nearest_location(X, Y, Z,df)
        print('nearest',nearest_location,nearest_index)
        #vx=nearest_location['Velocity_X_']
        #vy=nearest_location['Velocity_Y_']
        #vz=nearest_location['Velocity_Z_']
        #Ucomponent=[vx,vy,vz]
        
        #U_values = math.sqrt(vx**2 + vy**2 + vz**2) 
        #k_values = nearest_location['tke']
        return nearest_location #U_values, k_values, Ucomponent,
    else:
        #U_values = location_data['Velocity_X_']
        #k_values = location_data['k'].tolist()
        return location_data   
        

