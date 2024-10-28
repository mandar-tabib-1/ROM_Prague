In order to run the AI wind model to obtain wind and turbulence in urban city segment as a result of building-induced turbulence for drone operations.
Read all the points below and then follow it:
 1. For Prague City model - Open the following file on your google colab-
   https://colab.research.google.com/github/mandar-tabib-1/ROM_Prague/blob/main/Reconstruc_AI_colab_windspeed_googlecolab.ipynb
2. The user input to be provided is "Date and time" (see Step 1 in code) in YYYY-MM-DD HR:min:sec format as shown here - date_and_hrs_only = '2024-10-29 09:00:00'.
   Check if any error message comes on running the above cell. The data provided must lie in a future time upto 3 day from now. So, past dates and 3 days after the
   future IS NOT accepted. The code obtain the meso-scale wind direction and wind speed for this date and time from an external website, and uses
   this as input for the AI model. 
4. A OPTIONAL input AI model can take is drone path trajectory in "latitude" and "longitude". But it is NOT MANDATORY needed.
5. The output of AI code is : wind and turbulence at points within segment, and maximum wind and turbulence location.
6. Methodology : First a training data is generated using computational fluid dynamics simulation for use in training the AI model.
7. The AI model during the traiing uses an unsupervised machine learning model to decompose the training data and find the most dominant spatial patterns.
8. (called basis, which are function of spatial locations) and the accompanying coefficients (which are function of meso-scale wind speed and meso-scale wind direction) for each of the basis.
9. Then, A radial basis function (RBF) is trained to regress the coefficients to the meso-scale wind speed and meso-scale wind direction using the training data.During inference,
10. for any new meso-scale wind direction and wind speed at the user-provided hour of the day, the AI model then uses the trained RBF function and obtained basis function to
11. reconstruct the flow field (wind and turbulence) in entire city segments within few seconds (5-10 s) which is orders of magnitude faster than CFD runs (which can take upto 5 hrs). 
12. 
       
