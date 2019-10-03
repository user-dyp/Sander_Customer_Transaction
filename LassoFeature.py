# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 17:05:56 2019

@author: dipen
"""
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LassoFeature:
    '''
    how to use it:
        from LassoFeature import LassoFeature
        lassoFeature= LassoFeature(dataX,dataY)
        lf=lassoFeature.FindBestAlphaForLasso()
        lf=lassoFeature.LassoRegularization(alpha=0.01)
        
    : description
        # Here if you notice, we come across an extra term, which is known 
        # as the penalty term. λ given here, is actually denoted by alpha
        # parameter in the ridge function. So by changing the values of 
        # alpha, we are basically controlling the penalty term. Higher 
        # the values of alpha, bigger is the penalty and therefore the 
        # magnitude of coefficients are reduced.
        
        #Important Points:
        #
        #    It shrinks the parameters, therefore it is mostly used to prevent multicollinearity.
        #    It reduces the model complexity by coefficient shrinkage.
        #    It uses L2 regularization technique. (which I will discussed later in this article)
            
    :summary    
        # model summary
        #R-squared score: 0.57
        #Mean Average error: 5.499042007361856
        #Mean squared error: 65.25147959606764
        #Root Mean squared error: 8.077838794879955
        
    '''
   
    def __init__(self, data_X,data_Y):
        self.dataX = data_X
        self.dataY=data_Y
        #self.list_of_columns_having_1_category=[]
        #self.RemoveColumnHavingOneCategoryOnly()
    
    # return rmse and error dataframe,coeff dataframe
    # df_metrics,df_coeff= LassoRegularization(0.1)
    
    def LassoRegularization(self,alpha, order_by_coeff=False,detail_result=False,normalize=False):
        '''
        :param self:
        :param alpha: number
        :return: df_metrics, df_coeff
        
        :return: return dataframe of metrics containing score and rmse
                    return list of lass coefficient
        '''
        X_train, X_Test, y_train, y_Test = train_test_split(self.dataX, self.dataY, test_size=0.2, random_state=987)
        
        model_lasso = Lasso(alpha=alpha,normalize=normalize)
        model_lasso.fit(X_train, y_train)
        #iscategroical = category
        #predict value
        pred = model_lasso.predict(X_Test) 
        print('detail_result', detail_result)
        if detail_result==True:
            print('Coefficient:', model_lasso.coef_.reshape(-1,1))
            print("Intercept: %.2f" %model_lasso.intercept_)
            # Explained variance score: 1 is perfect prediction
            print('-------------Summary-----------------')
            print('R-squared score: %.2f' % model_lasso.score(X_Test, y_Test))
            print('Mean Average error:', metrics.mean_absolute_error(y_Test, pred))
            print('Mean squared error:', metrics.mean_squared_error(y_Test, pred))
#        if iscategroical:
#            print('Accuracy score: ', metrics.confusion_matrix(y_Test,pred>0.5))
        print('Root Mean squared error:', np.sqrt(metrics.mean_squared_error(y_Test, pred)))
        coeff =pd.DataFrame(model_lasso.coef_.reshape(-1,1)[:,0],index=self.dataX.columns.values, columns=['Value'])
        if order_by_coeff:
            coeff.sort_values(['Value'], ascending=False, inplace=True)
#        if iscategroical: 	
#            return [alpha,metrics.mean_squared_error(y_Test, pred), metrics.confusion_matrix(y_Test,if pred>0.5),model_lasso.score(X_Test, y_Test)], coeff
#        else: 	
        return [alpha,metrics.mean_squared_error(y_Test, pred),model_lasso.score(X_Test, y_Test)], coeff

    def FindBestAlphaForLasso(self, alphas=[], roundof=True,order_by_coeff=False,detail_result=False, normalize=False):
        '''
        :param alphas: array of number; by default it will pick some defined value for execute
        :return : return array of metrics dataframe
        '''
        data =[]
        if len(alphas)<=0:
            alphas = [1e-4, 1e-3,1e-2, 1, 5, 10, 20,50,100,200,1000]
        count=0    
        for alpha in alphas:
            print('')
            print('------------ Iteration {} : alpha: {} ------------------'.format(count,alpha))
            #print('alpha: ', alpha)
            d_metrics, coeff = self.LassoRegularization(alpha,order_by_coeff,detail_result,normalize)
            #print(d_metrics)
            data.append(d_metrics)
            count = count+1
        #columns=['alpha','rmse','r_score']    
        df_metrics = pd.DataFrame()
#        if iscategroical:
#            df_metrics=pd.DataFrame(data,columns=['alpha','rmse','accuracy','r_score'])
#        else:
        df_metrics=pd.DataFrame(data,columns=['alpha','rmse','r_score'])
        if roundof:
            df_metrics = df_metrics.round(4)
        print('Summary Matrics')
        print(df_metrics)
        return df_metrics
    
    def RemoveColumnHavingOneCategoryOnly(self):
        '''
        Due to incorrect sample sometime some column only have only one cateogory 
        which mean nothing and cannot consume by Lasso
        
        : param: dataX as DataFrame
        : return : new DataFrame dataX, list_of_columns_having_1_category
        '''
        list_col_uniq= []
        list_col_0categeory =[]
        for x in list(self.dataX.columns):
            if self.dataX[x].nunique()>1:
                list_col_uniq.append(x)
                #print(x,self.dataX[x].nunique())  
            else:
                list_col_0categeory.append(x)
                
        self.dataX=self.dataX[list_col_uniq]   
        self.list_of_columns_having_1_category=list_col_0categeory
        return self.dataX,  self.list_of_columns_having_1_category