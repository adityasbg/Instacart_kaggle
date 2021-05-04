
import pandas as pd 
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
import xgboost as xgb
import pickle
import joblib
import os 
import numpy as np
from multipledispatch import dispatch
from flask import Flask ,request,render_template

def read_pickle_dictionary(filename):
    '''
    Load serialized dictionary
    '''
    with open(os.path.join(filename), 'rb') as handle:
        file_dict = pickle.load(handle)
        
    return file_dict

def deserialize_model( filename):
    """
    DeSerialize trained model 
    """
    return joblib.load(filename)


def standardize(X_train, X_test ,test_preprocessing_object , flag ='train'):
    """
    This function standardize test and train columns if flag is train standization will fit and trainsform 
    if test it will just standardize.
    returns train ,test and  test_preprocessing_object if flag is train else test data and test_preprocessing_object
    X_train : train data
    X_test  :test data 
    test_preprocessing_object : dictionary containing object of columns transformer for different column 
    flag : string either train or test 
    """
    std_columns =['ur_pr_reordered','order_number','ttl_cnt_product_user','Avg_no_prod_perOrder','days_since_prior_order','usr_ro_ratio','product_name_length']
    scaler_objects =[]

    if(flag == 'train'):
        for col in std_columns:

            scaler = StandardScaler()
            scaler.fit(X_train.loc[:,col].values.reshape(-1,1))
            X_train.loc[:,col] =scaler.transform(X_train.loc[:,col].values.reshape(-1,1))
            X_test.loc[:,col] =scaler.transform(X_test.loc[:,col].values.reshape(-1,1))
            scaler_objects.append({col:scaler})
            del scaler
            gc.collect()
        test_preprocessing_object['std']=scaler_objects
    elif(flag =='test'):
        for item in test_preprocessing_object['std']:
            for col_item in item.items():
                col =col_item[0]
                scaler=col_item[1]

                X_test.loc[:,col] =scaler.transform(X_test.loc[:,col].values.reshape(-1,1))
    if(flag=='train'):
        return X_train.copy() , X_test.copy() ,test_preprocessing_object
    else:
        return X_test.copy() ,test_preprocessing_object
            
def response_code_test( X_test ,response_dict):
    """
    This function takes data and does fit transform based on column wise,
    according to column specific encoder stored in reponse_dctionary 
    X_test : test data 
    response_dict : dictionary containing encoder object 
    return transformed test data 
    """

    response_column =['max_hour_of_day' ,'reordered_last','max_dow']
    for col in response_column:
        encoder =response_dict[col]
        X_test.loc[:,col] =encoder.transform(X_test.loc[:,col])


    
    return X_test.copy()


def merge_products(x):
    """
    x : string input
    This function merge group to a list 
    returns list of strings 
    """
    return " ".join(list(x.astype('str')))

            



def suggestProduct(test_sub, pred , thresh):
    """
    This suggests products based on if Prediction is reordered =1
    
    returns Dataframe containing order_id and group of product_id seperated by space 
    
    test_sub : dataframe containing order_id and product_id
    
    pred:prediction probality of positive class
    
    thresh : optimal threshold to convert probality to class 
    """
    
    # if pobality is greter than threshold predict 1 else 0
    test_sub["Pred"] = np.where(pred>=thresh ,1,0)
    # select all cases where prediction is 1
    test_sub = test_sub.loc[test_sub["Pred"].astype('int')==1]
    #group by order_id and create lsit of products
    test_sub = test_sub.groupby("order_id")["product_id"].aggregate(merge_products).reset_index()
    test_sub.columns = ["order_id", "products"]
    

    return test_sub['products'].values

@dispatch(list,int)
def validate_order_id( orderIds,id_):
    """
    This function checks if querried order id is there in list of order_id
    returns True is order id is present and false if not present
    """

    flag =False
    for order_id in orderIds:
        if id_ == order_id:
            flag=True
            break       
    return flag


@dispatch(list,list)
def validate_order_id( orderIds,id_list):

    """
    This function returns list of valid order_id  querried by user
    """

    valid_order_ids =[]
    for querry_id in id_list:
        for order_id in orderIds:
            if order_id ==querry_id:
                valid_order_ids.append(querry_id)
                break
      
    return valid_order_ids
            


@dispatch(int)
def final(orderNumber):
    """
    This function takes orderNumber and suggest Product user is most Likely to buy
    orderNumber :Integer
    retruns : None or string of product Id seperated by space
    
    """
    # Read data set 
    print("read dataset ")
    test =pd.read_parquet('data/test.gzip')
    # get all the order_id in dataset
    orderIds =list(test.order_id.values)
    product_suggestion ='None'
    print("validate dataset ")
    # check if order If order Id querred is valid 
    if(validate_order_id( orderIds,orderNumber)):
        print("filtter dataset ")
        # filter dataset based on orderId
        test=test[test.order_id ==orderNumber]
        
        # store all the product user bought for particular order id
        test_temp = test[["order_id", "product_id"]]
        print("drop column ")
        # drop unnecessarory columns  
        test.drop(columns =["order_id",'ur_pr_count' ,"user_id" ,'product_id' ,'department_id' ,'aisle_id' ,'ur_pr_count'] , inplace =True)
        # standardizing test data 
        print("std")
        test_preprocessing_object= read_pickle_dictionary(os.path.join('final_model_pkl' ,'test_preprocessing_object_dict.pkl' ))
        test , test_preprocessing_object=standardize(None, test ,test_preprocessing_object , flag ='test')
        # Target code encodding
        print("target encoding")
        reponse_dict= read_pickle_dictionary(os.path.join('final_model_pkl' ,'reponse_dict.pkl' ))
        test=response_code_test( test ,reponse_dict)
        # read pickled  model for prediction 
        xgboost=deserialize_model('final_model_pkl/xgboost.pkl')
        # predict probality
        print("prediction")
        predict_xg_test =xgboost.predict_proba(test)[:,1]
        # threshold for prediction
        threshold=0.692886
        product_name =suggestProduct(test_temp ,predict_xg_test ,threshold)
        print(product_name)
        # if suggested product is not empty take that as suggested product else None
        if(product_name.shape[0]>0):
            product_suggestion =product_name[0]
                  
    return product_suggestion
    
    

def predict_product(orderNumber):
    data = pd.read_csv('data/xg_boost_850_.csv')
    orderIds =list(data.order_id.values)
    product_out=None
    if(validate_order_id( orderIds,orderNumber)):
        data=data[data.order_id ==orderNumber]
        data=data['products']
        if data.shape[0]:
            product_out=data.values[0]
    return product_out


app = Flask(__name__)

@app.route('/' ,methods = ['get'])
def home():
    return render_template('Instacart.html')

@app.route('/predict' ,methods = ['POST'])
def predict():
    if request.method =='POST':

        order_id =int(request.form['order_id'])
        print(order_id)
        # product =final(order_id)
        product=predict_product(order_id)
        return render_template('result.html' ,  order_id=order_id , product_id = product)


if __name__ == '__main__':
   app.run(host='0.0.0.0',port =8080, debug=True)