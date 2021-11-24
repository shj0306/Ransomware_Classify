import os
import pickle

from data_process import DataProcess
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import accuracy_score
import calc_auc as ca
import warnings
from calc_auc import Calc_Auc
warnings.filterwarnings(action='ignore')
os.chdir('C:/sharedfolder/testmodel')#작업할 디렉토리 설정
train_software_path = '../software/behavior_api_order.csv'
train_malware_path = '../behavior_api_order.csv'
test_software_path = './test_software_behavior_api_order.csv'
test_malware_path = './test_behavior_api_order.csv'
model_name = "../Doc2vec_model_vector30_window15_dm0"

def get_clf_eval(y_test, pred):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)

    return accuracy, precision, recall, confusion

dp = DataProcess()
ca = Calc_Auc()
train_software_sentences, train_malware_sentences = dp.extract_data(train_software_path, train_malware_path)
test_software_sentences, test_malware_sentences = dp.extract_data(test_software_path, test_malware_path)

x_train, y_train = dp.make_vectordata(model_name, train_software_sentences, train_malware_sentences)

x_test, y_test = dp.make_vectordata(model_name, test_software_sentences, test_malware_sentences)

rf_clf = RandomForestClassifier()
rf_model = rf_clf.fit(x_train, y_train)
pickle.dump(rf_model,open("rf_model.dat","wb")) #학습 모델 저장
loaded_model = pickle.load(open("rf_model.dat","rb")) #학습 모델 로드
y_pred = loaded_model.predict(x_test)
y_pred  = y_pred > 0.5
y_pred = y_pred.astype(int)

acc, precision, recall, _ = get_clf_eval(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy: %.2f%%" % (acc * 100.0))  # 예측률
print("Precision : %.2f%%" % (precision * 100.0))  # 정밀도
print("Recall : %.2f%%" % (recall * 100.0))  # 재현율
print("f1 score : %.2f%%" % (f1 * 100.0))  # f1_score

ca.roc_curve_plot(y_test, y_pred)



