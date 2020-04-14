#!/usr/bin/env python3

#
#--------------------------------------------------------------------------------
#--                                                                            --
#--                 Federal University of Santa Maria                          --
#--                        Technology Center                                   --
#--                     Computer Engineering                                   --
#--                 Santa Maria - Rio Grande do Sul/BR                         --
#--                                                                            --
#--------------------------------------------------------------------------------
#--                                                                            --
#-- Design      : detecting Activity v1.0                                      --
#-- File		: detectingActivity.py      	                               --
#-- Authors     : Luis Felipe de Deus                                          --
#--             : Leonardo Ferreira                                            --
#--             : Tiago Knorst                                                 --
#--             : Cesar Abascal                                                --
#--                                                                            --
# --Mentors     : Cesar Augusto Prior and Cesar Rodrigues                      -- 
#--                                                                            -- 
#--------------------------------------------------------------------------------
#--                                                                            --
#-- Created     : 10 Jun 2019                                                  --
#-- Update      : 09 Jul 2019                                                  --
#--------------------------------------------------------------------------------
#--                              Overview                                      --
#--                                                                            --
#--                                                                            --
#-- This code perform data analysis of features extracted from HRV data        --
#-- obtained and processed by the code elgendiMethod.py                        --
#-- *Run PCA for data visualization                                            --
#-- *Run Random Forest classifier                                              --
#-- *Run KNN classifier                                                        --
#--                                                                            --
#-- Code executed in python3 (3.6)                                             --
#--------------------------------------------------------------------------------
#

#Import the libraries we need
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

#--------------------------------------------------------------------------------

# @brief:   This Function execute KNeighors classifier
# @param:   the data is a dataFrame of pandas library
# @param:   the testData is a dataFrame of pandas library
# @param:   the features is array of labels for features
# @param:   the test_strategy is a boolean indicator from witch test
# @return:  void
def KNNeighbors(data, testData, features, test_strategy):
    print('#\n\n\n### RUNNING KNN ####')
    #Print five init data
    print(data.head())
    
    #Separate the data in X- Features | Y- Labels
    X=data[[*features]]  # Features
    y=data['activity']  # labels

    xT = testData[[*features]] 
    yT = testData['activity'] 
    #print('--- Print data in Features ----')
    #print(xT)

    #print('-----Print data in Labels ------')
    #print(yT)  

    # Split dataset into training set and test set in case use the same data for trainning and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% 
    
    #print('------ Trainning Data --------') 
    #print(X_train.head())

    #print('------- Test Data ------------') 
    #print(X_test)

    ##Create a KNN Classifier
    clf = KNeighborsClassifier(n_neighbors = 3)

    ##Train the model using the training sets 
    if(test_strategy):     
        clf.fit(X,y)
    else:
        clf.fit(X_train,y_train)

    if(test_strategy):
        #print("## TEST STRATEGY 1 ###")
        ##Predict the response for test dataset (testData)
        y_pred=clf.predict(xT)

        accuracy = metrics.accuracy_score(yT, y_pred)
        # Model Accuracy: how often is the classifier correct?
        print(" ---- Accuracy:",accuracy)

        # Create and print the matrix of confusion
        print(confusion_matrix(yT, y_pred))  
        tn, fp, fn, tp = confusion_matrix(yT, y_pred).ravel()
        print("True Positive: ", tp)
        print("False Positive: ", fp)
        print("True Negative: ", tn)
        print("False Negative: ", fn)

        print(classification_report(yT, y_pred)) 

        probs = clf.predict_proba(xT)
        # keep probabilities for the positive outcome only
        probs = probs[:, 1]
        # calculate AUC
        auc = roc_auc_score(yT, probs)
        print('AUC: %.3f' % auc)
        # calculate roc curve
        fpr, tpr, thresholds = roc_curve(yT, probs)

    else:
        ##Predict the response for test dataset
        #print("## TEST STRATEGY 0 ###")
        y_pred=clf.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        # Model Accuracy: how often is the classifier correct?
        print("\n ---- Accuracy:",accuracy)

        # Create and print the matrix of confusion
        print(confusion_matrix(y_test, y_pred))  
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        print("True Positive: ", tp)
        print("False Positive: ", fp)
        print("True Negative: ", tn)
        print("False Negative: ", fn)

        print(classification_report(y_test, y_pred)) 

        probs = clf.predict_proba(X_test)
        # keep probabilities for the positive outcome only
        probs = probs[:, 1]
        # calculate AUC
        auc = roc_auc_score(y_test, probs)
        print('AUC: %.3f' % auc)
        # calculate roc curve
        fpr, tpr, thresholds = roc_curve(y_test, probs)

    # plot no skill
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('ROC Curve')
    # plot the roc curve for the model
    plt.plot(fpr, tpr, marker='.')
    # show the plot
    plt.show()

#End KNN


# @brief:   This Function execute random forest classifier
# @param:   the data is a dataFrame of pandas library
# @param:   the testData is a dataFrame of pandas library
# @param:   the features is array of labels for features
# @param:   the test_strategy is a boolean indicator from witch test
# @return:  void
def randomForest_(data, testData, features, test_strategy):
    print('#\n\n\n### RUNNING RANDOM FOREST ####')
    #Print five init data
    print(data.head())
    
    #Separate the data in X- Features | Y- Labels
    X=data[[*features]]  # Features
    y=data['activity']  # labels

    xT = testData[[*features]] 
    yT = testData['activity'] 
    #print('--- Print data in Features ----')
    #print(xT)

    #print('-----Print data in Labels ------')
    #print(yT)  

    # Split dataset into training set and test set in case use the same data for trainning and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% 
    
    #print('------ Trainning Data --------') 
    #print(X_train)

    #print('------- Test Data ------------') 
    #print(X_test)

    ##Create a Gaussian Classifier
    #clf=RandomForestClassifier(n_estimators=100)
    clf =  RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
    
    ##Train the model using the training sets 
    if(test_strategy):     
        clf.fit(X,y)
    else:
        clf.fit(X_train,y_train)

    if(test_strategy):
        #print("## TEST STRATEGY 1 ###")
        ##Predict the response for test dataset (testData)
        y_pred=clf.predict(xT)

        accuracy = metrics.accuracy_score(yT, y_pred)
        # Model Accuracy: how often is the classifier correct?
        print(" ---- Accuracy:",accuracy)

        # Create and print the matrix of confusion
        print(confusion_matrix(yT, y_pred))  
        tn, fp, fn, tp = confusion_matrix(yT, y_pred).ravel()
        print("True Positive: ", tp)
        print("False Positive: ", fp)
        print("True Negative: ", tn)
        print("False Negative: ", fn)

        print(classification_report(yT, y_pred)) 

        probs = clf.predict_proba(xT)
        # keep probabilities for the positive outcome only
        probs = probs[:, 1]
        # calculate AUC
        auc = roc_auc_score(yT, probs)
        print('AUC: %.3f' % auc)
        # calculate roc curve
        fpr, tpr, thresholds = roc_curve(yT, probs)

    else:
        ##Predict the response for test dataset
        #print("## TEST STRATEGY 0 ###")
        y_pred=clf.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        # Model Accuracy: how often is the classifier correct?
        print(" ---- Accuracy:",accuracy)

        # Create and print the matrix of confusion
        print(confusion_matrix(y_test, y_pred))  
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        print("True Positive: ", tp)
        print("False Positive: ", fp)
        print("True Negative: ", tn)
        print("False Negative: ", fn)

        print(classification_report(y_test, y_pred)) 

        probs = clf.predict_proba(X_test)
        # keep probabilities for the positive outcome only
        probs = probs[:, 1]
        # calculate AUC
        auc = roc_auc_score(y_test, probs)
        print('AUC: %.3f' % auc)
        # calculate roc curve
        fpr, tpr, thresholds = roc_curve(y_test, probs)

    # plot no skill
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('AUC-ROC Curve')
    # plot the roc curve for the model
    plt.plot(fpr, tpr, marker='.')
    # show the plot
    plt.show()


    # Print the importance of features in the model
    importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(clf.feature_importances_,3)})
    importances = importances.sort_values('importance',ascending=False).set_index('feature')
    print(importances.head(15))
    
    ft_imp = []
    for i in range(0,len(features)):
        ft_imp.append(importances.iloc[i]['importance']*100)
        #print(ft_imp)
        
    if(__debug__):
        fig1, ax1 = plt.subplots()
        ax1.pie(ft_imp, autopct ='%1.1f%%', shadow=False, startangle=90)
        ax1.axis('equal')
        ax1.set_title("Importance of Features")
        ax1.legend(features, title="Features", loc="right")
        plt.show()

    return accuracy
#End random forest

# @brief:   This Function execute principal component analysis
# @param:   an "mean" is a list of all means the data - Feature
# @param:   an "sdnn" is a list of all sdnn the data - Feature
# @param:   an "rmssd" is a list of all rmssd the data - Feature
# @param:   an "sdsd" is a list of all sdsd the data - Feature
# @param:   an "median" is a list of all median the data - Feature
# @param:   an "activity" is a list of all activity the data - Label
# @return:  data.T is a transposed of dataFrame
def doPCA(values,ft, mean):
    
    #Create a indicator of sample named 's'. s1,s2,s3...sn
    sample = ['s' + str(i) for i in range(1,(len(mean)+1))]
    
    #Create the dataFrame of all data (basically is an array 2d)
    data = pd.DataFrame(values, index=[*ft], columns=[*sample], dtype = float)
    
    #print(" ------- MATRIZ  ORIGINAL --------")
    print(data)
    #print("Linha X Coluna: ", data.shape)

    # First center and scale the data
    scaled_data = preprocessing.scale(data.T)
   
    #Crate a object the Principal Component Analysis
    pca = PCA()
    #Ajust the model and execute
    pca.fit(scaled_data)
    pca_data = pca.transform(scaled_data)
    
    #Create a bar graph of PC's and yours variance
    per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
    
    if( __debug__):
        plt.bar(x=range(1,len(per_var)+1),height=per_var, tick_label=labels)
        plt.ylabel('Percentage of Explained Variance')
        plt.xlabel('Principal component')
        plt.title('Scree Plot')
    
        plt.show()
    
        #Create a graph witch represent the data in all PCs
        pca_df = pd.DataFrame(pca_data, index = [*sample], columns=labels)
        plt.scatter(pca_df.PC1, pca_df.PC2)
        plt.title('My PCA Graph')
        plt.xlabel('PC1 - {0}%'.format(per_var[0]))
        plt.ylabel('PC2 - {0}%'.format(per_var[1]))

        for sample in pca_df.index:
            plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))
    
        plt.show()

    # Determine which the biggest influence on PC1
    ## get the name of the top 10 measurements that contribute
    ## most to pc1.
    ## first, get the loading scores
    loading_scores = pd.Series(pca.components_[0], index=ft)
    ## now sort the loading scores based on their magnitude
    sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
    
    # get the names of the top 5 features
    top_5 = sorted_loading_scores[0:5].index.values
    
    ## print the names and their scores (and +/- sign)
    #print(loading_scores[top_5])
    ft.remove('activity')
    #print(ft)
    return data.T
#end#


# @brief:   This Function open the .csv archive in pacient folder and read the features 
# @param:   "patients" is a list of names of pacients
# @return:   an "mean" is a list of all means the data - Feature
# @return:   an "sdnn" is a list of all sdnn the data - Feature
# @return:   an "rmssd" is a list of all rmssd the data - Feature
# @return:   an "sdsd" is a list of all sdsd the data - Feature
# @return:   an "median" is a list of all median the data - Feature
# @return:   an "activity" is a list of all activity the data - Label
def readFeatures(patients):
    #Create lists for storage the data
    mean, sdnn, rmssd, median,sdsd, var = [], [], [], [], [], []
    activity = []
    range_ , cvsd , cvnni = [],[],[]
    i = 0
    #Access the folder of patients and read the data
    while(i < len(patients)):
        patientN = patients[i]
        path = "exg-data/raw-exg/" + patientN+ "/Features.csv"
        with open(path) as dataFile:
            next(dataFile)
            next(dataFile)
            for line in dataFile:
                aux = line.split(';')
                mean.append(float(aux[0]))
                sdnn.append(float(aux[1]))
                rmssd.append(float(aux[2]))
                sdsd.append(float(aux[3]))
                median.append(float(aux[4]))
                var.append(float(aux[5]))
                range_.append(float(aux[6]))
                cvsd.append(float(aux[7]))
                cvnni.append(float(aux[8]))
                activity.append(float(aux[9]))
            #end-for
        #end-with
        i+=1
     #end while
    return mean,sdnn,rmssd,sdsd,median,var,range_,cvsd,cvnni, activity
#end 

# ---------------------------------- MAIN  ----------------------------------------------------------------------------------

# Expects a number of patient in argument
nPatient = int(sys.argv[1])

i = 0
patients = []
#Read all patient names of argument and append of a list
while(i < nPatient):
    #print(sys.argv[i+2])
    patients.append(sys.argv[i+2])
    i+=1
    
testName = []
testName.append('husm')
#testName.append(str(input(('Patient name for testing the model: '))))
#print(testName)

####### PROCESSING FIRST DATA SET
#Effect te read of features in patient folder
mean,sdnn,rmssd,sdsd,median,var,range_,cvsd,cvnni, activity = readFeatures(patients)

#Create the dataFrame of all data (basically is an array 2d)
#dataInit = [mean,sdnn,rmssd,sdsd,median,var,range_,cvsd, activity]
dataInit = [mean,sdnn,rmssd, activity]
 #Create indicators for features and labels
#ft = ['mean','sdnn','rmssd','sdsd','median','var','range_','cvsd', 'activity']
ft = ['mean','sdnn','rmssd', 'activity']

#Effect the PCA of the data
data = doPCA(dataInit,ft, mean)

####### PROCESSING SECOND DATA SET WHICH IS A TEST PATIENT 
#Effect the read of features in patient folder Testing pacient
mean,sdnn,rmssd,sdsd,median,var,range_,cvsd,cvnni, activity = readFeatures(testName)
ft = ['mean','sdnn','rmssd', 'activity']
dataInit = [mean,sdnn,rmssd, activity]
#Effect the PCA of the testing data
testData = doPCA(dataInit,ft, mean)

############ RUN RANDOM FOREST  ##########
randomForest_(data,testData,ft, True)

############ RUN KNN ##########
KNNeighbors(data,testData, ft, True)
