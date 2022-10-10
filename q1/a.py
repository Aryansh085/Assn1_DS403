from turtle import color
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

train1=pd.read_csv('Segment2_STEnergy.csv')
train2=pd.read_csv('Segment2_MelEnergy.csv')

train=pd.read_csv('Segment2_VAD_GT.csv')

test1=pd.read_csv('Segment3_STEnergy.csv')
test2=pd.read_csv('Segment3_MelEnergy.csv')

test=pd.read_csv('Segment3_VAD_GT.csv')
# print(train1.info())
# print(train.info())
# print(test.info())

df1=pd.DataFrame({'Energy':train1.iloc[:,0],'Value':train.iloc[:,0]})
df2=pd.DataFrame({'Energy':train2.iloc[:,0],'Value':train.iloc[:,0]})
dft=pd.DataFrame({'Value':test.iloc[:,0]})
# print(df1.info())
# print(df2.info())

dft1=pd.DataFrame({'Energy':test1.iloc[:,0],'Value':test.iloc[:,0]})
dft2=pd.DataFrame({'Energy':test2.iloc[:,0],'Value':test.iloc[:,0]})

dft1=list(dft1['Energy'])
dft2=list(dft2['Energy'])

dft=list(dft['Value'])
print(len(dft))
# plt.scatter(df1['Value'],df1['Energy'],color="r")

# plt.scatter(df2['Value'],df2['Energy'],color="green")


def fn(df,df1,dft,c):
    #df is the energy value which we will predict
    #df1 is train data set
    #dft is the actual class 
    m1=0
    m2=0
    c1=0
    c2=0
    for i in range(len(df1['Value'])):
        if(df1['Value'][i]==0):
            c1=c1+1
            m1=m1+df1['Energy'][i]
        else:
            c2=c2+1
            m2=m2+df1['Energy'][i]
    # plt.show()

    # print(c1,c2,m1,m2)
    # print(x)
    m1=m1/c1
    m2=m2/c2
    s1=0
    s2=0
# ____________________________Part to be reseen-------

    for i in range(len(df1['Value'])):
        if(df1['Value'][i]==0):
            s1=s1+((df1['Energy'][i]-m1)*(df1['Energy'][i]-m1))
        else:
            s2=s2+((df1['Energy'][i]-m2)*(df1['Energy'][i]-m2))
    s1=s1/(c1-1)
    s2=s2/(c2-1)
    print(m1,m2)
    print(c1,c2)
    print(s1,s2)
    # m1 and m2 are means of the two classes 
    # s1 and s2 are the variances of the two classes

    # lis=list(df[0])
    res=[]
    xd=[]
    print("len of df is", len(df))
    for i in range(len(df)):
        y=1/(math.sqrt(s1*2*3.1415))
        k=float(df[i])
        k=k-m1
        y=y*(pow(2.718,(-1.0)*(((k)*(k))/(2*s1))))
        x=y
        y=1
        k=float(df[i])
        k=k-m2
        y=1/(math.sqrt(s2*2*3.1415))
        xd.append(y*(pow(2.718,(-1.0)*(((k)*(k))/(2*s2))))*(c2/(c1+c2)))
        a=y*(pow(2.718,(-1.0)*(((k)*(k))/(2*s2))))
        
        # a = np.random.random(1)[0]
        res.append(a)
        # print(x,y)
    # p/=len(df1['Value'])
    # print("sum is ",p )

    print("res is")
    print(xd)
    # x=np.arange(-15,30,2)
    x=np.arange(-0.2,4,0.1)
    X=[]
    Y=[]
    for i in range(len(x)):
        tp=0
        tn=0
        fp=0
        fn=0
        # Segregating based on 
        for j in range(len(res)):
            if(res[j]>=x[i]):
                if(dft[j]==1):
                    tp=tp+1
                else:
                    fp=fp+1
            else:
                if(dft[j]==0):
                    tn=tn+1  
                else:
                    fn=fn+1     
        Y.append(tp/(tp+fn))
        X.append(fp/(fp+tn))
    # print(X)
    # print(Y)
    plt.plot([0,1],[0,1])
    plt.plot(X,Y)
    stri=str(c)
    plt.legend(stri)
    
print("len of dft1",len(dft))

fn(dft1,df1,dft,1)

fn(dft2,df2,dft,2)
plt.show()

# _________________Part to be reseen-----------


