import tensorflow as tf
import numpy as np
import pandas as pd
import math
import random
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import  StandardScaler, LabelEncoder, OneHotEncoder, LabelBinarizer, MinMaxScaler, MultiLabelBinarizer



from initial import IN
from SurvNet import FN



#----------------------------------------------------------------------------------------------------
# Generate/load the data
# (here we take dataset 1 as a demo, you can substitute it with your own data)
#----------------------------------------------------------------------------------------------------

seed=1 # set a seed

#data function for reading and processing the train and test sets
#necessary as an input for the optimisation algorithm
def data():
    #define input processing function
    def process_attributes(df, train, val, test):
        
        #define and fit the scaler to the full dataset
        cs = MinMaxScaler()
        cs.fit(df_inputs.select_dtypes(np.number))
        
        #scale the numerical input variables
        trainContinuous = cs.transform(train.select_dtypes(np.number))
        valContinuous = cs.transform(val.select_dtypes(np.number))
        testContinuous = cs.transform(test.select_dtypes(np.number))
        
        #uncomment the code below to accommodate for any categorical columns
        zipBinarizer = LabelBinarizer().fit(df["Gender"])
        trainCategorical = zipBinarizer.transform(train["Gender"])
        valCategorical = zipBinarizer.transform(val["Gender"])
        testCategorical = zipBinarizer.transform(test["Gender"])
        
        # construct our training and testing data points by concatenating
        # the categorical features with the continuous features
        trainX = np.hstack([trainContinuous, trainCategorical])
        valX = np.hstack([valContinuous, valCategorical])
        testX = np.hstack([testContinuous, testCategorical])
        
        
        # return the concatenated training and testing data
        return (trainX, valX, testX)
    
    #read the excel datasets
    df = pd.read_excel('Cleaned_Dataframe.xlsx')
    df.set_index('Sample',inplace=True)

    df_cancer = df.loc[df['Status'] == 'Cancer']
    df_control = df.loc[df['Status'] == 'Control']

    #randomly seelct 538 samples from the cancer population to create an equal sample size 
    df_cancer_small = df_cancer.sample(n=538, random_state = 100)

    df1 = pd.concat([df_cancer_small, df_control])
    #separate cancer markers and input data
    df_outputs= df1['Status']
    df_inputs = df1.drop(['Status'],axis=1)

    
    X_train, X_test_val, y_train, y_test_val = train_test_split(df_inputs, df_outputs, random_state=100, stratify=df_outputs, test_size=0.4)
    X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, random_state=100, stratify=y_test_val, test_size=0.5)
    #process the input sets
    (X_train_sc, X_val_sc, X_test_sc) = process_attributes(df_inputs, X_train, X_val, X_test)
    
    #encode the categorical output variables
    #encode categorical outputs
    lb = LabelBinarizer()
    lb.fit(y_train)
    train_outputs= lb.transform(y_train)
    val_outputs= lb.transform(y_val)
    test_outputs= lb.transform(y_test)
    
    lb2=MultiLabelBinarizer()
    lb2.fit(train_outputs)
    Y_train = lb2.transform(train_outputs)
    Y_val = lb2.transform(val_outputs)
    Y_test = lb2.transform(test_outputs)
    #Y_train = to_categorical(train_outputs)
    #Y_val = to_categorical(val_outputs)
    #Y_test = to_categorical(test_outputs)

    return X_train_sc, Y_train, X_val_sc, Y_val, X_test_sc, Y_test, lb

train_X, train_Y, val_X, val_Y, test_X, test_Y, lb = data()


#np.random.seed(seed)
#whole_X=np.random.uniform(0,1,(19570,260))
#n=whole_X.shape[0]
p0=train_X.shape[1] # the number of original variables

#random.seed(seed)
#art=np.array(random.sample(range(p0),24)) # the indexes of significant variables

# make the variables significant by shifting their values at half of the samples
#np.random.seed(seed)
#sign=np.random.choice([-1,1],len(art))
#u=np.random.uniform(0.1,0.3,len(art))
#u=u*sign
#mat=np.reshape(np.tile(u,int(0.5*n)),(int(0.5*n),len(art)))
#whole_X[:int(0.5*n),art]=whole_X[:int(0.5*n),art]+mat

# define labels
#whole_Y=np.zeros((n,2))
#whole_Y[:int(0.5*n),0]=1
#whole_Y[int(0.5*n):,1]=1

# permute (shuffle) and standardize the data
#t=np.random.RandomState(seed).permutation(n)
#whole_X=whole_X[t]
#whole_Y=whole_Y[t]
#whole_X=preprocessing.scale(whole_X)

# data splitting
#train_X=whole_X[:int(0.8*0.7*n)]
#train_Y=whole_Y[:int(0.8*0.7*n)]
#val_X=whole_X[int(0.8*0.7*n):int(0.8*n)]
#val_Y=whole_Y[int(0.8*0.7*n):int(0.8*n)]
#test_X=whole_X[int(0.8*n):]
#test_Y=whole_Y[int(0.8*n):]



#----------------------------------------------------------------------------------------------------
# Define the network struture and parameters
#----------------------------------------------------------------------------------------------------

n_classes=2
n_hidden1=32
n_hidden2=24
learning_rate=0.001
epochs=100
batch_size=32
num_batches=train_X.shape[0]/batch_size
dropout=0.25886
alpha=0.03 # used for a GL_alpha stopping criterion

print('train_X shape', train_X.shape[0], train_X.shape[1], 'num_batches', num_batches )

#----------------------------------------------------------------------------------------------------
# Define the number of surrogate variables, a prespecified threshold for FDR, an elimination rate, 
# and salience: "abs" or "squ", which means using either absolute values or squares of partial derivatives 
# to measure variable importance
#----------------------------------------------------------------------------------------------------
 
q0=p0
eta=0.3
elimination_rate=0.5

salience="squ"



#----------------------------------------------------------------------------------------------------
# Print some important parameters
# Note: 'art' is not applicable to real datasets with unknown significant variables
#----------------------------------------------------------------------------------------------------

print('seed:',seed)
print('FDR cutoff:',eta,'elimination rate:',elimination_rate,'salience:',salience,'\n')




#----------------------------------------------------------------------------------------------------
# Run initial results without variable selection
#----------------------------------------------------------------------------------------------------

initial=IN(seed, 
           train_X,train_Y,val_X,val_Y,test_X,test_Y, 
           n_classes,n_hidden1,n_hidden2, 
           learning_rate,epochs,batch_size,num_batches,dropout,alpha)



#----------------------------------------------------------------------------------------------------
# Run SurvNet
# Note: 'PP' and 'AFDR' are not applicable to real datasets with unknown significant variables
#----------------------------------------------------------------------------------------------------

final,No_org,sval_org,P,Q,train_Loss,train_Acc,val_Loss,val_Acc,EFDR=FN(seed, 
                                                                                train_X,train_Y,val_X,val_Y,test_X,test_Y, 
                                                                                n_classes,n_hidden1,n_hidden2, learning_rate,epochs,batch_size,num_batches,dropout,alpha, 
                                                                                p0,q0,eta,elimination_rate,salience)



#----------------------------------------------------------------------------------------------------
# Output results
#----------------------------------------------------------------------------------------------------
initnfinal=np.concatenate((initial,final))
np.savetxt('initnfinal.txt',initnfinal)

P,Q,train_Loss,train_Acc,val_Loss,val_Acc,EFDR=np.array(P),np.array(Q),np.array(train_Loss),np.array(train_Acc),np.array(val_Loss),np.array(val_Acc),np.array(EFDR)
step=np.column_stack((P,Q,train_Loss,train_Acc,val_Loss,val_Acc,EFDR))
np.savetxt('step.txt',step)

result=np.column_stack((No_org,sval_org))
np.savetxt('result.txt',result, delimiter=',' )