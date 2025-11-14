# Travel_insurance_modelling_catboost_rus  
-Projek ini membuat model klasisfikasi untuk memprediksi apakah kustomer akan melakukan claim atau tidak.  
-Dataset sangat imbalanced(≈98.5% non-claim, 1.5% claim) dan mempertimbangkan businiss problem, model ini berfokus untuk memaksimalkan recall.  
-Final model yang digunakan adalah Catboost menggunakan Random Undersampling (RUS)

# Goals  
Perusahaan memerlukan pengecekan resiko yang berbasis data-driven untuk membantu early detection atas potential claimants.

Tujuan pembuatan model klasifikasi ini adalah untuk:  
1. Melakukan identifikasi dini untuk kustomer yang berpotensi untuk (likely to) membuat klaim;  
2. Meminimalisir tidak terdeteksinya kustomer yang secara potensial melakukan klaim (undetected claimants)(meminimalisir false negative);  
3. Menyediakan latar belakang pengetahuan bagi tim strategis dan tim terkait untuk melakukan langkah-langkah proaktif dalam mengelola resiko klaim asuransi perjalanan;  

Model ini akan membantu tim manajemen resiko untuk mengambil keputusan dan mengalokasi sumber daya perusahaan dengan lebih efisien dan akurat.  

# 5 Points Business ML Goals  
1. Business Problem:  
Perusahaan kehilangan keuntungan (profit optimal) ketika kustomer yang berpotensi melakukan klaim tidak terdeteksi sejak awal.

2. Data:
Data berasal dari perusahaan asuransi yang memuat (durasi, produk yang dibeli, net values, destination, dll) dari kustomer yang membeli asuransi perjalanan. Data highly imbalanced dengan hanya 1.5% kustomer melakukan klaim.  

3. ML Objective:  
Membuat model klasifikasi yang mendapat nilai recall tinggi, tujuannya mengidentifikasi potential claimants secara lebih akurat.  

4. Action:  
Model klasifikasi akan digunakan oleh Managemen, Risk Management Team, dan Underwriting Team untuk:  
a) Melakukan identifikasi dini untuk kustomer yang berpotensi untuk membuat klaim;  
b) Melakukan readjustment premi asuransi untuk produk asuransi yang teridentifikasi beresiko melakukan klaim;  
c) Melakukan review ulang terhadap agen yang teridentifikasi melakukan banyak klaim berdasarkan data historis.  

5. Business Value:  
a) Mengurangi kerugian finansial akibat tidak terdeteksinya nasabah yang berpotensi melakukan klaim;  
b) Meningkatkan akurasi penilaian resiko dan strategi penetapan premi asuransi.;  
c) Meningkatkan efisiensi operasional melalui pengambilan keputusan berbasis data.  
d) Meningkatkan transparansi dalam pemantauan kinerja agen.  

# Data Dictionary:  
Agensi: Nama agensi (pake kode)  
Agency Type: jenis agenis  
Distribution Channel: saluran distribus  
Product Name: nama produk asuransi  
Gender: jenis kelamin
Duration: Lama perjalanan  
Destination: Destinasi  
Net Sales: jumlalah penjualan atas satu produk asuransi perjalanan  
Commision (in value): nilai komisi yg diterima agen asuransi  
Age: Usia  
Claim: Status klaim (1 = klaim, 0 = tidak klaim)  


# Library  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  

#pipa  
from sklearn.compose import ColumnTransformer  
from sklearn.pipeline import Pipeline  

#prepro  
from sklearn.preprocessing import OneHotEncoder  
from sklearn.preprocessing import StandardScaler  
from sklearn.preprocessing import RobustScaler  
#model_select  
from sklearn.model_selection import train_test_split  
from sklearn.model_selection import StratifiedKFold  
from sklearn.model_selection import cross_val_score  
from sklearn.model_selection import cross_val_predict  
from sklearn.model_selection import RandomizedSearchCV  
#model  
from sklearn.linear_model import LogisticRegression  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.ensemble import AdaBoostClassifier  
from sklearn.ensemble import GradientBoostingClassifier  

from xgboost import XGBClassifier  
from lightgbm import LGBMClassifier  
from catboost import CatBoostClassifier  
#mblearn&sampling  
from imblearn.pipeline import Pipeline as ImbPipeline  
from imblearn.over_sampling import SMOTE  
from imblearn.over_sampling import RandomOverSampler  
from imblearn.under_sampling import RandomUnderSampler  
from imblearn.under_sampling import NearMiss  
from imblearn.combine import SMOTETomek  
from imblearn.combine import SMOTEENN  
from imblearn.ensemble import EasyEnsembleClassifier  
#metric  
from sklearn.metrics import precision_recall_curve  
from sklearn.metrics import auc  
from sklearn.metrics import precision_score  
from sklearn.metrics import recall_score  
from sklearn.metrics import f1_score  
from sklearn.metrics import average_precision_score  
from sklearn.metrics import classification_report  
from sklearn.metrics import confusion_matrix  


from sklearn.base import clone
from scipy.stats import randint
from scipy.stats import uniform
import pickle
import shap



