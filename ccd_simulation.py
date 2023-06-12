# READING THE IMAGE AND GENERATING 3 DIFFERENT MATRICES FOR RED, GREEN AND BLUE
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# Function to extract pixel data of input image
def getPixels(filename):
    im = Image.open(filename, 'r')
    im.load()
    #Converting the colorspace of the image to RGB
    img = Image.new('RGB', im.size, (255, 255, 255))
    img.paste(im, None)

    w, h = img.size
    pix = list(img.getdata())
    return np.array([pix[n: n + w] for n in range(0, w * h, w)])

# Function to take the RGB wxhx3 matrix and creating 3 different matrices
# for Red, Green and Blue values
def RGB(pixels):
    red = []
    green = []
    blue = []
    for i in range(len(pixels)):
        rtemp, gtemp, btemp = [], [], []
        for j in range(len(pixels[0])):
            rtemp.append(pixels[i][j][0])
            gtemp.append(pixels[i][j][1])
            btemp.append(pixels[i][j][2])
        red.append(rtemp)
        green.append(gtemp)
        blue.append(btemp)
    return red, green, blue

pixels = getPixels("img.jpg")
plt.imshow(pixels)
plt.show()
red, green, blue = RGB(pixels)
print(red[200])

#Noises to be taken into account for charge to signal 
"""
Dark Current Shot noise, 
    I(dc shot e-) = Poisson Process (mean = I(dc e-))
    I(dc e-) = Integration Time (t1) * Average Dark Current(Dr)
    Dr = Pixel Area (Pa) * Figure-of-Merit (Dfm) * (Temperature)^3/2 * exp (-Egap/2kT)
"""
dim1=len(red)
dim2=len(red[0])

PRNU = 0.01 #PRNU factor is generally 0.01 , 0.02
t_int = 1 #integration time in seconds
T = 300 #Temperature in Kelvin
P_area = 1e-6 #Pixel Area in mm^2
Dfm = 1 #figure-of-merit, constant for a ccd, kept random as of now, please change later
Egap = 1 #Energy Band Gap in eV
k = 1.38e-23 #Boltzmann Constant in SI units
Dn = 1 #Dark current FPN factor, constant for a ccd, kept random as of now, please change later


mean = t_int * P_area * Dfm * (T**3/2) * np.exp(-Egap/(2*k*T))

# using the above mean dark current, we generate a poisson distribution
red += np.random.poisson(mean, size = (dim1, dim2))
green += np.random.poisson(mean, size = (dim1, dim2))
blue += np.random.poisson(mean, size = (dim1, dim2))

"""
Dark Current Fixed Pattern Noise, 
   I(dc FPN e-) = I(dc shot e-) + I(dc shot e-)*Log-Normal-distribution(0,variance(dc FPN e-))
   standard deviation(dc FPN e-) = integration time(t1) * Average Dark Current (Dr) * Dark Current FPN factor(Dn)
"""

dcFPN = t_int * mean * Dn * 1000
red = red + red*np.random.lognormal(0,dcFPN, size = (dim1, dim2))
green = green + green*np.random.lognormal(0,dcFPN, size = (dim1, dim2))
blue = blue + blue*np.random.lognormal(0,dcFPN, size = (dim1, dim2))

"""
Photon response non uniformity
    I(PRNU e-) = I(e-) + I(e-)*Normal-Distribution(0,variance(PRNU))
    standard deviation(PRNU) = PRNU factor value
"""
red += red*np.random.normal(0,PRNU**2, size = (dim1, dim2))
blue += blue*np.random.normal(0,PRNU**2, size = (dim1, dim2))
green += green*np.random.normal(0,PRNU**2, size = (dim1, dim2))

red *= 0.5
green *= 0.5
blue *= 0.5
#Size of the image

#Incident photon wavelengths: 632 nm(red),532 nm(green),465 nm(blue)
#Generate random electron counts
#Band Gap of p-type Si: 1.128eV. All wavelengths capable of promoting electrons.
#r is the photon influx count, Quantum efficiency=60%

red=0.6*red
blue = 0.6*blue
green = 0.6*green

#Function to consider blooming


def blooming(red):
  """
  This function simulates blooming in CCDs. Blooming is considered to occur only along the column due to channel stops.
  """
#Pixel saturation occurs at value=245(typical)
  saturated=245
  k=0
  factor=0.5
  for row in range(len(red)):
    for column in range(len(red[0])):
      while(red[row-k][column]>saturated and row-k-1>=0 and row+k+1<=dim1):
        red[row-k-1][column]=factor*(red[row-k][column]-saturated)
        factor = 1
        k=k+1
  
blooming(red)
blooming(green)
blooming(blue)
'''
Voltage conversion and amplification
#Capacitance = 100 uF
#Amplification factor =
Cap = 1e-4
Amp_fac = 1e10

voltage = arr*Cap
voltage = Amp_fac*voltage

#Noise
k = 1.38 * 1e-23  #boltzmann constant
T = 298   #room temp
var = k*T*1e4
kt_noise = np.random.lognormal(0,var,size = [length,width])

voltage += kt_noise
'''
# take n_cross n arraay as input. outputs a 2 cross n_sqare, containing the two signal values array to next group that processes the image
from IPython.core.display import json
import numpy as np

#Ref = np.array([120]*(n*n))
#Capacitance = 100 uF
#Amplification factor =
Cap = 1e-4
Amp_fac = 1e10

'''voltage = arr*Cap
voltage = Amp_fac*voltage'''

#Noise
k = 1.38 * 1e-23  #boltzmann constant
T = 298   #room temp


def shift(single_colour_array):
    n = len(single_colour_array)
    noise = 0.123
    ampli_factor = 0.146
    register = np.zeros(n)

    var = k*T*1e4
    kt_noise = np.random.lognormal(0,var)

    output_array = np.zeros([2,n**2])

    # arbitory value of reference == 1 V
    # capacitance = 0.01 uF 
    capacitor = 0.01
    single_colour_array = capacitor*single_colour_array
    
    for i in range(n):
        # the register1 is carrying the the column to register 1       
        register = single_colour_array[0:n,i] + (i+1)*noise +kt_noise
        
        # register voltage is amplified
        register = register*ampli_factor
        
        for j in range(n):
            # upper value is the reference
            output_array[0 , (n*j)+i] = 1+ noise*((j*n)+i+1) + kt_noise

            # below value is the actual voltage value
            output_array[1 , (n*j) + i] = register[j]+ noise*((j*n)+i+1) +kt_noise

    return output_array


'''
sample_Arr = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
print(len(sample_Arr))
print(shift(sample_Arr))
'''
r_shifted,g_shifted,b_shifted = shift(red),shift(green),shift(blue)
print(r_shifted)
# Code for noise
# **Last Part**
import numpy as np
n = dim1
l = (n ** 2)
r0 = 1
Cap = 15

def process(array):
    global n, r0
    # r0arr = np.array([r0] * l)
    array = np.transpose(array)

    diff1 = array[:, 0] - np.ones((l,), dtype=int)
    varr1 = array[:, 1] - diff1
    w = n
    h = n    
    return (varr1.reshape(w, h))

red_m = process(r_shifted)
green_m = process(g_shifted)
blue_m = process(b_shifted)


Charge_Matrix_red = red_m * Cap
Charge_Matrix_green = green_m * Cap
Charge_Matrix_blue = blue_m * Cap
print(Charge_Matrix_red[700])
print(np.max(Charge_Matrix_red), np.max(Charge_Matrix_green), np.max(Charge_Matrix_blue))

# Displaying image
import matplotlib.pyplot as plt
from PIL import Image

def rgb_concat(r, g, b):
    res = []
    for i in range(len(r)):
        row = []
        for j in range(len(r[0])):
            temp = [r[i][j], g[i][j], b[i][j]]
            row.append(temp)
        res.append(row)
    return res

No_Photons_Output_Matrix_ = np.floor(rgb_concat(Charge_Matrix_red, Charge_Matrix_green, Charge_Matrix_blue))

print(No_Photons_Output_Matrix[200])