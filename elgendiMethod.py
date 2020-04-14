#!/usr/bin/env python3

#
#--------------------------------------------------------------------------------
#--                                                                            --
#--                 Universidade Federal de Santa Maria                        --
#--                        Centro de Tecnologia                                --
#--                 Curso de Engenharia de Computação                          --
#--                 Santa Maria - Rio Grande do Sul/BR                         --
#--                                                                            --
#--------------------------------------------------------------------------------
#--                                                                            --
#-- Design      : Elgendi Methods v1.0                                         --
#-- File		: elgendiMethod.py      	                              	   --
#-- Authors     : Luis Felipe de Deus                                          --
#--             : Leonardo Ferreira                                            --
#--             : Tiago Knorst                                                 --
#--             : Cesar Abascal                                                --
#--                                                                            --
# --Mentors     : Cesar Augusto Prior and Cesar Rodrigues                      -- 
#--                                                                            -- 
#--------------------------------------------------------------------------------
#--                                                                            --
#-- Created     : 10 Abr 2019                                                  --
#-- Update      : 29 Jun 2019                                                  --
#--------------------------------------------------------------------------------
#--                              Overview                                      --
#--                                                                            --
#-- This code performs data processing. It reads the .cdv file with the        --
#-- patients ECG signal, applies the elgendi method to detect the QRS complex, --
#-- obtains the HRV of this data by removing outliers and finally executes the --
#-- extraction of features. The extraction of features is done in moving       --
#-- windows of 10 HRV samplesThis code perform data analysis of features       --
#-- extracted from HRV data                                                    --
#-- obtained and processed by the code elgendiMethod.py                        --
#--                                                                            --
#-- Code executed in python3                                                   --
#--------------------------------------------------------------------------------
#

#Import the libraries we need
from processing.readandfilter import *
from processing.mathprocessing import *
from processing.makegraphics import *
import pandas as pd
import numpy as np
import math
import re
import pyhrv as ph
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from hrvanalysis import *
from sklearn.decomposition import PCA

# @brief:   This Function execute aquisition of RR-Time
# @param:   peaks is a value in time of happened the R wave
# @return:  RRTime is a list the values of difference in time between the R wave
def getRRTime(peakx):
    RRTime = []
    last = 0
    for i in peakx:
        RRTime.append(i - last)
        last = i
        
    return RRTime       
#end

# @brief:   This Function save in .csv file the R-R Time, acumulator R time and a index
# @param:   patientN is a name of patient that we need for access your folder
# @param:   RRTime is a list of R-R Time values
# @return:  void
def saveRR(patientN, RRTime):
    path = "exg-data/raw-exg/" + patientN
    dataFile = open(path+"/RRTime.csv", "w")
    dataFile.write("Diff Between Pulses ;  ACC\n")
    acc = 0
    idx = 0
    for valor in RRTime:
        acc += valor
        dataFile.write(str(valor)+";" + \
                        str(acc)+";" + \
                        str(idx)+"\n"
                        )
        idx+=1
    dataFile.close()
#End

# @brief:   This Function execute the features about the move window in the RRTime data 
# @param:   patientN is a name of patient that we need for access your folder
# @return:  void
def getAndExportFeatures(patientName):
    #Open the file of features that we will write
    pathW = "exg-data/raw-exg/" + patientName
    dataFileW = open(pathW+"/Features.csv", "w")
    dataFileW.write("Features collected of individuals \n")
    dataFileW.write("Mean ; SDNN ; RMSSD ; SDSD ; Median ; Var ; Range ; CVSD ; CVNNI  ; Activity\n")

    lista = []
    #Open the file of RRTime that we will read
    path = "exg-data/raw-exg/" + patientName+ "/RRTime.csv"
    with open(path) as dataFile:
        next(dataFile)
        for line in dataFile:
            aux = line.split(';')
            RRTime = (float(aux[0]))
            acc = (float(aux[1]))
            idx = (int(aux[2]))
            if(idx < 20):   #Se não houveram 10 amostras ainda so add na lista
                lista.append(RRTime)
            else:
                timeFeatures = get_time_domain_features(lista) #Ja tem 10 amostras entao calcula a features
                #print(timeFeatures)
                frame = pd.Series([*lista])
                variance = frame.var()
                if(acc <= 60):   #Menor que 60s é rest
                    activity = 0
                elif (acc>60 and acc <= 180): #Maior que 60s e Menor que 180s(3min) é running
                    activity = 1
                else:
                    activity = 1    #Se nao é recovery

                dataFileW.write( str(timeFeatures['mean_nni']) + ";" + 
                                str(timeFeatures['sdnn']) + ";" +
                                str(timeFeatures['rmssd']) + ";" +
                                str(timeFeatures['sdsd']) + ";" +
                                str(timeFeatures['median_nni']) + ";" +
                                str(variance) + ";" +
                                str(timeFeatures['range_nni']) + ";" +
                                str(timeFeatures['cvsd']) + ";" +
                                str(timeFeatures['cvnni']) + ";" +
                                str(activity) + "\n"
                             ) #Escreve no arquivo Features.csv

                lista.pop(0)    #Remove o valor de indice 0
                lista.append(RRTime)    #Adiciona no fim da lista
                #print("Instante: ", idx)   #Descomente para printar
                #print(lista)

#end

# @brief:   This function remove outliers (if the difference for one R-R in another is greater 0.3, so is outlier)
# @param:   RRTime is a list of differenc betwen R wave
# @return:  RRTime_NO is a list the RRTime without the outliers
def removeOutliers(RRTime):
    delta = 0
    oldDelta = 0
    last = 0
    RRTime_NO = []
    for i in RRTime:
        #print(i)
        delta = i - last
        if(delta > (oldDelta+0.3) and last != 0):
            #print('Found Outlier - Removed: ', i)
             i=i
        else:
            RRTime_NO.append(i)
            last = i    
            oldDelta = delta
    return RRTime_NO
#end    
 

# ----------------------------------------- MAIN ----------------------------------------------------------------------------------
# Reading uPPG signals
ecgPSG, pletPSG, annMarksPSG, sps, patientName = getSignals(psg=True)
ECG = ecgPSG # The ecg data

# Calculating x axis
nsamples = len(ECG)
xAxis = np.linspace(0, nsamples/sps, nsamples, endpoint=True)

# Filtering PSG ECG signal
lowcut = 8 # 8 From Elgendi
highcut = 20 # 20 From Elgendi
order = 2 # 2 From Elgendi
ECGf = butter_bandpass_filter_zi(ECG, lowcut, highcut, sps, order)
#ECGf = ECG

# Squaring PSG ECG signal
ECGfs = squaringValues(ECGf)

# W1 = 97ms (19pts @ 200Hz)
W1 = 19 # Nearest odd integer
MAqrs = expMovingAverage_abascal(ECGfs, W1)

# W2 = 611ms (123pts @ 200Hz)
W2 = 123 # Nearest odd integer
MAbeat = expMovingAverage_abascal(ECGfs, W2)

# Statiscal mean of the signal
ECGfsa = average(ECGfs)

# Alpha will be the multiplication of ECGfsa by beta plus MAbeat
beta = 0.08 # Provide by Elgendi
alpha = (beta * ECGfsa) + MAbeat

# Threshold1 will be the sum of each point in MAbeat by alpha
THR1 = MAbeat + alpha # array

# Threshold2 will be the same as W1
THR2 = W1 # scalar

# Getting blocks of interest with rejected noise and peaks.
# realBlocksOfInterest = blocks of interest with rejected noise
realBlocksOfInterest, peakx, peaky = elgendiRealBOIandPeaks(xAxis, ECGf, MAqrs, THR1, THR2)

# Obtain R-R Time and print de HeartRate
RRTime = getRRTime(peakx[1:len(peakx)])
#Remove outliers
RRTime_NO = removeOutliers(RRTime)

#Save de RR Time in .csv file
saveRR(patientName, RRTime_NO)
#Get features and save in .csv file
getAndExportFeatures(patientName) 

#if( __debug__):
#    ph.frequency_domain.welch_psd(RRTime_NO, rpeaks=peakx, nfft=2**12, detrend=True, window='hamming', show=True, show_param=True, legend=True)
#    plot_psd(RRTime_NO, method="welch")

print('Heart Rate: {:.2f}' .format(60/(sum(RRTime_NO)/len(RRTime_NO))))

# Ploting ECG filtered signal, MApeak and MAbeat
#plot_signal_movingAvarages(xAxis, ECGf, MAqrs, MAbeat, True)

# Ploting ECF filtered signal and Blocks Of Interest
#plot_signal_realBlocksOfInterest(xAxis, ECGf, realBlocksOfInterest, True)

# Ploting ECF filtered signal and peaks
if( __debug__):
    plot_signal_peaks(xAxis, ECGf, peakx, peaky, RRTime_NO, show=True)