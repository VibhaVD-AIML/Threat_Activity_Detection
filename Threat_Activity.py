
# coding: utf-8

# In[1]:

# Import the packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:

#Load the dataset of cybersecurity
dataset = pd.read_csv("CYBERSECURITY_DATASET_INCIDENT_COMPLETE.csv")


# In[3]:

dataset.head()


# In[4]:

dataset["os"] = dataset["os"].astype('category').cat.codes


# In[5]:

dataset["version"] = dataset["version"].astype('category').cat.codes


# In[6]:

dataset["location"] = dataset["location"].astype('category').cat.codes


# In[7]:

dataset["device_name"] = dataset["device_name"].astype('category').cat.codes


# In[8]:

dataset["application"] = dataset["application"].astype('category').cat.codes


# In[9]:

dataset["service"] = dataset["service"].astype('category').cat.codes


# In[10]:

dataset["listed_ip_address"] = dataset["listed_ip_address"].astype('category').cat.codes


# In[11]:

dataset["unlisted_ip_address"] = dataset["unlisted_ip_address"].astype('category').cat.codes


# In[12]:

dataset.head()


# In[13]:

#Divide the dataset into train and test for identifying it is a compromised device or not
X = dataset.iloc[:,[0,2,3,4,5,6,13,14]]
y = dataset.iloc[:,[15]]


# In[14]:

X.head()


# In[15]:

y.head()


# In[16]:

#Splitting the dataset into train and test
from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[17]:

# Applying XG Boost Algorithm  
import xgboost
from xgboost import XGBClassifier
model3 = XGBClassifier()
model3.fit(X_train, y_train)
print(model3.score(X_test,y_test))


# In[18]:

y_pred1 =pd.DataFrame(model3.predict(X_test))
y_pred1.head()


# In[19]:

# Predicting the output when the user enter the first value
y_pred1 =pd.DataFrame(model3.predict(X_test.iloc[0:1,:]))
y_pred1.columns = ['compromised_devices']
y_pred1.head()


# In[20]:

#Meging the id column with the respective output
compromised_devices=pd.concat([X_test.iloc[0:1,0:1].reset_index(drop=True),y_pred1],axis=1)
compromised_devices


# In[21]:

#Divide the dataset into train and test for identifying it is a risk devices or not 
X1 = dataset.iloc[:,[0,2,3,4,5,6,13,14]]
y1 = dataset.iloc[:,[16]]


# In[22]:

from sklearn.model_selection import train_test_split

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=0)


# In[23]:

# Applying XG Boost Algorithm  -
import xgboost
from xgboost import XGBClassifier
model7 = XGBClassifier()
model7.fit(X1_train, y1_train)
print(model7.score(X1_test,y1_test))


# In[24]:

y1_pred2 =pd.DataFrame( model7.predict(X1_test))
y1_pred2.head()


# In[25]:

# Predicting the output when the user enter the first value
y1_pred2 =pd.DataFrame(model7.predict(X_test.iloc[0:1,:]))
y1_pred2.columns = ['risk_devices']
y1_pred2


# In[26]:

#Meging the id column with the respective output
risk_devices=pd.concat([X_test.iloc[0:1,0:1].reset_index(drop=True),y1_pred2],axis=1)
risk_devices


# In[27]:

#Divide the dataset into train and test for identifying it is a threat activity or not
X2 = dataset.iloc[:,[0,2,3,4,5,6,13,14]]
y2 = dataset.iloc[:,[17]]


# In[28]:

from sklearn.model_selection import train_test_split

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=0)


# In[29]:

# Applying Random Forest Algorithm
from sklearn.ensemble import RandomForestClassifier
model11 = RandomForestClassifier(n_estimators=10)
model11.fit(X2_train, y2_train)
print(model11.score(X2_test,y2_test))


# In[30]:

y2_pred3 = pd.DataFrame(model11.predict(X2_test))
y2_pred3.head()


# In[31]:

# Predicting the output when the user enter the first value
y2_pred3 =pd.DataFrame(model11.predict(X_test.iloc[0:1,:]))
y2_pred3.columns = ['threat_activity']
y2_pred3


# In[32]:

#Meging the id column with the respective output
threat_activity=pd.concat([X_test.iloc[0:1,0:1].reset_index(drop=True),y2_pred3],axis=1)
threat_activity


# In[33]:

# Applying Gradient Boosting Algorithm
from sklearn.ensemble import GradientBoostingClassifier
model13 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
model13.fit(X2_train, y2_train)
print(model13.score(X2_test,y2_test))


# In[34]:

#Divide the dataset into train and test for identifying it is a healthy device or not
X3 = dataset.iloc[:,[0,2,3,4,5,6,13,14]]
y3 = dataset.iloc[:,[18]]


# In[35]:

from sklearn.model_selection import train_test_split
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.3, random_state=0)


# In[36]:

# Applying Random Forest Algorithm
from sklearn.ensemble import RandomForestClassifier
model15 = RandomForestClassifier(n_estimators=10)
model15.fit(X3_train, y3_train)
print(model15.score(X3_test,y3_test))


# In[37]:

y3_pred4 = pd.DataFrame(model15.predict(X3_test))
y3_pred4.head()


# In[38]:

# Predicting the output when the user enter the first value
y3_pred4 =pd.DataFrame(model15.predict(X_test.iloc[0:1,:]))
y3_pred4.columns = ['healthy_devices']
y3_pred4


# In[39]:

#Meging the id column with the respective output
healthy_devices=pd.concat([X_test.iloc[0:1,0:1].reset_index(drop=True),y3_pred4],axis=1)
healthy_devices


# In[40]:

#Divide the dataset into train and test for identifying it is infected device or not
X4 = dataset.iloc[:,[0,2,3,4,5,6,13,14]]
y4 = dataset.iloc[:,[19]]


# In[41]:

from sklearn.model_selection import train_test_split
X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.3, random_state=0)


# In[42]:

# Applying Random Forest Algorithm
from sklearn.ensemble import RandomForestClassifier
model17 = RandomForestClassifier(n_estimators=10)
model17.fit(X4_train, y4_train)
print(model17.score(X4_test,y4_test))


# In[43]:

y4_pred4 = pd.DataFrame(model17.predict(X4_test))
y4_pred4.head()


# In[44]:

# Predicting the output when the user enter the first value
y4_pred4 =pd.DataFrame(model17.predict(X_test.iloc[0:1,:]))
y4_pred4.columns = ['infected_devices']
y4_pred4


# In[45]:

#Meging the id column with the respective output
infected_devices=pd.concat([X_test.iloc[0:1,0:1].reset_index(drop=True),y4_pred4],axis=1)
infected_devices


# In[46]:

#Divide the dataset into train and test for identifying it is_listed device or not 
X5 = dataset.iloc[:,[0,2,3,4,5,6,13,14]]
y5 = dataset.iloc[:,[20]]


# In[47]:

from sklearn.model_selection import train_test_split
X5_train, X5_test, y5_train, y5_test = train_test_split(X5, y5, test_size=0.3, random_state=0)


# In[48]:

# Applying XG Boost Algorithm  -
import xgboost
from xgboost import XGBClassifier
model19 = XGBClassifier()
model19.fit(X5_train, y5_train)
print(model19.score(X5_test,y5_test))


# In[49]:

y5_pred5 = pd.DataFrame(model19.predict(X5_test))
y5_pred5.head()


# In[50]:

# Predicting the output when the user enter the first value
y5_pred5 =pd.DataFrame(model19.predict(X_test.iloc[0:1,:]))
y5_pred5.columns = ['is_listed_device']
y5_pred5


# In[51]:

#Meging the id column with the respective output
is_listed_device=pd.concat([X_test.iloc[0:1,0:1].reset_index(drop=True),y5_pred5],axis=1)
is_listed_device


# In[52]:

#Divide the dataset into train and test for identifying it is a un listed device or not
X6 = dataset.iloc[:,[0,2,3,4,5,6,13,14]]
y6 = dataset.iloc[:,[21]]
y6.head()


# In[53]:

from sklearn.model_selection import train_test_split
X6_train, X6_test, y6_train, y6_test = train_test_split(X6, y6, test_size=0.3, random_state=0)


# In[54]:

# Applying Random Forest Algorithm
from sklearn.ensemble import RandomForestClassifier
model21 = RandomForestClassifier(n_estimators=10)
model21.fit(X6_train, y6_train)
print(model21.score(X6_test,y6_test))


# In[55]:

y6_pred6 =pd.DataFrame(model21.predict(X6_test))
y6_pred6.head()


# In[56]:

# Predicting the output when the user enter the first value
y6_pred6 =pd.DataFrame(model21.predict(X_test.iloc[0:1,:]))
y6_pred6.columns = ['is_unlisted_device']
y6_pred6


# In[57]:

#Meging the id column with the respective output
is_unlisted_device=pd.concat([X_test.iloc[0:1,0:1].reset_index(drop=True),y6_pred6],axis=1)
is_unlisted_device


# In[58]:

#Divide the dataset into train and test for identifying it is a alert type or not
X7 = dataset.iloc[:,[0,2,3,4,5,6,13,14]]
y7 = dataset.iloc[:,[22]]


# In[59]:

from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X7_train, X7_test, y7_train, y7_test = train_test_split(X7, y7, test_size=0.3, random_state=0)


# In[60]:

#Applying Gradient Boosting Algorithm
from sklearn.ensemble import GradientBoostingClassifier
model25 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
model25.fit(X7_train, y7_train)
print(model25.score(X7_test,y7_test))


# In[61]:

y7_pred7 = pd.DataFrame(model25.predict(X7_test))
y7_pred7.head()


# In[62]:

# Predicting the output when the user enter the first value
y7_pred7 =pd.DataFrame(model25.predict(X_test.iloc[0:1,:]))
y7_pred7.columns = ['alert_type']
y7_pred7


# In[63]:

#Meging the id column with the respective output
alert_type=pd.concat([X_test.iloc[0:1,0:1].reset_index(drop=True),y7_pred7],axis=1)
alert_type


# In[64]:

compromised_devices.set_index('id',inplace=True)
risk_devices.set_index('id',inplace=True)
threat_activity.set_index('id',inplace=True)
healthy_devices.set_index('id',inplace=True)
infected_devices.set_index('id',inplace=True)
is_listed_device.set_index('id',inplace=True)
is_unlisted_device.set_index('id',inplace=True)
alert_type.set_index('id',inplace=True)
df = pd.concat([compromised_devices,risk_devices,threat_activity,healthy_devices,infected_devices,is_listed_device,is_unlisted_device,alert_type],axis=1,sort=False).reset_index()
df.rename(columns = {'index':'id'})


# In[65]:

X_test.iloc[0:1,:]


# In[66]:

x= {'id':709,'listed_ip_address':227,'unlisted_ip_address':0,'os':1,'version':2,'location':15,'application':6,'service':7}


# In[67]:

x


# In[68]:

pd.DataFrame([x],columns=['id', 'listed_ip_address','unlisted_ip_address','os','version','location','application','service'])


# In[69]:

input=pd.DataFrame([x],columns=['id', 'listed_ip_address','unlisted_ip_address','os','version','location','application','service'])


# In[70]:

y_pred1 =pd.DataFrame(model3.predict(input))
y_pred1.columns = ['compromised_devices']
#Meging the id column with the respective output
compromised_devices=pd.concat([input.iloc[0:1,0:1].reset_index(drop=True),y_pred1],axis=1)

# Predicting the output when the user enter the first value
y1_pred2 =pd.DataFrame(model7.predict(input.iloc[0:1,:]))
y1_pred2.columns = ['risk_devices']

#Meging the id column with the respective output
risk_devices=pd.concat([input.iloc[0:1,0:1].reset_index(drop=True),y1_pred2],axis=1)

# Predicting the output when the user enter the first value
y2_pred3 =pd.DataFrame(model11.predict(input))
y2_pred3.columns = ['threat_activity']
#Meging the id column with the respective output
threat_activity=pd.concat([input.iloc[0:1,0:1].reset_index(drop=True),y2_pred3],axis=1)


# Predicting the output when the user enter the first value
y3_pred4 =pd.DataFrame(model15.predict(input))
y3_pred4.columns = ['healthy_devices']
#Meging the id column with the respective output
healthy_devices=pd.concat([input.iloc[0:1,0:1].reset_index(drop=True),y3_pred4],axis=1)


# Predicting the output when the user enter the first value
y4_pred4 =pd.DataFrame(model17.predict(input))
y4_pred4.columns = ['infected_devices']

#Meging the id column with the respective output
infected_devices=pd.concat([input.iloc[0:1,0:1].reset_index(drop=True),y4_pred4],axis=1)


# 5 Predicting the output when the user enter the first value
y5_pred5 =pd.DataFrame(model19.predict(input))
y5_pred5.columns = ['is_listed_device']
#Meging the id column with the respective output
is_listed_device=pd.concat([input.iloc[0:1,0:1].reset_index(drop=True),y5_pred5],axis=1)

# 6 Predicting the output when the user enter the first value
y6_pred6 =pd.DataFrame(model21.predict(input))
y6_pred6.columns = ['is_unlisted_device']
#Meging the id column with the respective output
is_unlisted_device=pd.concat([input.iloc[0:1,0:1].reset_index(drop=True),y6_pred6],axis=1)

# 7 Predicting the output when the user enter the first value
y7_pred7 =pd.DataFrame(model25.predict(input))
y7_pred7.columns = ['alert_type']
#Meging the id column with the respective output
alert_type=pd.concat([input.iloc[0:1,0:1].reset_index(drop=True),y7_pred7],axis=1)


compromised_devices.set_index('id',inplace=True)
risk_devices.set_index('id',inplace=True)
threat_activity.set_index('id',inplace=True)
healthy_devices.set_index('id',inplace=True)
infected_devices.set_index('id',inplace=True)
is_listed_device.set_index('id',inplace=True)
is_unlisted_device.set_index('id',inplace=True)
alert_type.set_index('id',inplace=True)
df = pd.concat([compromised_devices,risk_devices,threat_activity,healthy_devices,infected_devices,is_listed_device,is_unlisted_device,alert_type],axis=1,sort=False).reset_index()
df.rename(columns = {'index':'id'})


# In[71]:

str(df['id'].values[0])


# In[ ]:




# In[72]:

from flask import Flask, request , jsonify
from flask_restful import Resource, Api
#from flask import Blueprint
from json import dumps
#api_bp = Blueprint('api', __name__)
#from flask.ext.jsonpify import jsonify
from flask_cors import CORS
import traceback
app = Flask(__name__)
CORS(app)
api = Api(app)


# In[73]:

input=pd.DataFrame([x],columns=['id', 'listed_ip_address','unlisted_ip_address','os','version','location','application','service'])

y_pred1 = pd.DataFrame(model3.predict(input))
y_pred1.columns = ['compromised_devices']
pd.concat([input.reset_index(drop=True),y_pred1],axis=1)

y1_pred2=pd.DataFrame(model7.predict(input))
y1_pred2.columns = ['risk_devices']
#Meging the id column with the respective output
risk_devices=pd.concat([input.reset_index(drop=True),y1_pred2],axis=1)

y2_pred3 =pd.DataFrame(model11.predict(input))
y2_pred3.columns = ['threat_activity']
threat_activity=pd.concat([X_test.iloc[0:1,0:1].reset_index(drop=True),y2_pred3],axis=1)

y3_pred4 =pd.DataFrame(model15.predict(input))
y3_pred4.columns = ['healthy_devices']
#Meging the id column with the respective output
healthy_devices=pd.concat([input.reset_index(drop=True),y3_pred4],axis=1)

y4_pred4 = pd.DataFrame(model17.predict(input))
y4_pred4.columns = ['infected_devices']
#Meging the id column with the respective output
infected_devices=pd.concat([input.reset_index(drop=True),y4_pred4],axis=1)

y5_pred5 =pd.DataFrame(model19.predict(input))
y5_pred5.columns = ['is_listed_device']

#Meging the id column with the respective output
y6_pred6 =pd.DataFrame(model21.predict(input))
y6_pred6.columns = ['is_unlisted_device']
#Meging the id column with the respective output
is_unlisted_device=pd.concat([input.reset_index(drop=True),y6_pred6],axis=1)



y7_pred7 =pd.DataFrame(model25.predict(input))
y7_pred7.columns = ['alert_type']
#Meging the id column with the respective output
alert_type=pd.concat([input.reset_index(drop=True),y7_pred7],axis=1)


# In[ ]:

from flask import Flask, request , jsonify
from flask_restful import Resource, Api
#from flask import Blueprint
#from json import dumps
import json
#api_bp = Blueprint('api', _name_)
#from flask.ext.jsonpify import jsonify
from flask_cors import CORS
import traceback

app = Flask(__name__)
CORS(app)
api = Api(app)
class threatDetection(Resource):
    def post(self):
        try:
            x = request.get_json()
            input=pd.DataFrame.from_records([s for s in x],columns=['id', 'listed_ip_address','unlisted_ip_address','os','version','location','application','service'])
            y_pred1 =pd.DataFrame(model3.predict(input))
            y_pred1.columns = ['compromised_devices']
            #Meging the id column with the respective output
            compromised_devices=pd.concat([input.iloc[:,0:1].reset_index(drop=True),y_pred1],axis=1)

            # Predicting the output when the user enter the first value
            y1_pred2 =pd.DataFrame(model7.predict(input))
            y1_pred2.columns = ['risk_devices']

            #Meging the id column with the respective output
            risk_devices=pd.concat([input.iloc[:,0:1].reset_index(drop=True),y1_pred2],axis=1)

            # Predicting the output when the user enter the first value
            y2_pred3 =pd.DataFrame(model11.predict(input))
            y2_pred3.columns = ['threat_activity']
            #Meging the id column with the respective output
            threat_activity=pd.concat([input.iloc[:,0:1].reset_index(drop=True),y2_pred3],axis=1)


            # Predicting the output when the user enter the first value
            y3_pred4 =pd.DataFrame(model15.predict(input))
            y3_pred4.columns = ['healthy_devices']
            #Meging the id column with the respective output
            healthy_devices=pd.concat([input.iloc[:,0:1].reset_index(drop=True),y3_pred4],axis=1)


            # Predicting the output when the user enter the first value
            y4_pred4 =pd.DataFrame(model17.predict(input))
            y4_pred4.columns = ['infected_devices']

            #Meging the id column with the respective output
            infected_devices=pd.concat([input.iloc[:,0:1].reset_index(drop=True),y4_pred4],axis=1)


            # 5 Predicting the output when the user enter the first value
            y5_pred5 =pd.DataFrame(model19.predict(input))
            y5_pred5.columns = ['is_listed_device']
            #Meging the id column with the respective output
            is_listed_device=pd.concat([input.iloc[:,0:1].reset_index(drop=True),y5_pred5],axis=1)

            # 6 Predicting the output when the user enter the first value
            y6_pred6 =pd.DataFrame(model21.predict(input))
            y6_pred6.columns = ['is_unlisted_device']
            #Meging the id column with the respective output
            is_unlisted_device=pd.concat([input.iloc[:,0:1].reset_index(drop=True),y6_pred6],axis=1)

            # 7 Predicting the output when the user enter the first value
            y7_pred7 =pd.DataFrame(model25.predict(input))
            y7_pred7.columns = ['alert_type']
            #Meging the id column with the respective output
            alert_type=pd.concat([input.iloc[:,0:1].reset_index(drop=True),y7_pred7],axis=1)


            compromised_devices.set_index('id',inplace=True)
            risk_devices.set_index('id',inplace=True)
            threat_activity.set_index('id',inplace=True)
            healthy_devices.set_index('id',inplace=True)
            infected_devices.set_index('id',inplace=True)
            is_listed_device.set_index('id',inplace=True)
            is_unlisted_device.set_index('id',inplace=True)
            alert_type.set_index('id',inplace=True)
            df = pd.concat([compromised_devices,risk_devices,threat_activity,healthy_devices,infected_devices,is_listed_device,is_unlisted_device,alert_type],axis=1,sort=False).reset_index()
            df.rename(columns = {'index':'id'})
            print(df)
            result=df.to_json(orient='records')
            return {'status': 'success', 'response': json.loads(result)}, 200
        except:
            return jsonify({'trace': traceback.format_exc()})
api.add_resource(threatDetection, '/api/threat-detection')
#api.add_resource(saveTrainData, '/api/retrainmodel')  

if __name__ == '__main__':
    app.run(host='0.0.0.0',port='3002')
           


# In[ ]:




# In[ ]:



