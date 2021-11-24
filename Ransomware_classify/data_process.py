import csv
import numpy as np
from gensim.models.doc2vec import TaggedDocument, Doc2Vec


def featureset_remove_fun(sentence, remove_features):
    '''
    sentence : 문장, remove_features : 제거할 특성들. 제거 된 상태로 나가게됨
    '''
    return [e for e in sentence if e not in (remove_features)]

class DataProcess:

    def extract_data(self, software_path, ransom_path): #특정 경로의 파일 내용을 읽어서 리스트에 저장
        with open(software_path, 'r', encoding='utf-8') as f:
            lines = csv.reader(f)
            software_sentences = [line for line in lines]
        with open(ransom_path, 'r', encoding='utf-8') as f:
            lines = csv.reader(f)
            ransomware_sentences = [line for line in lines]

        return software_sentences, ransomware_sentences

    #extract data함수로 가져온 데이터와 doc2vec 모델을 이용해서 데이터를 벡터로 변환
    def make_vectordata(self, model_name, software_sentences, ransomware_sentences):
        doc2vec_model_name = model_name
        model = Doc2Vec.load(doc2vec_model_name)

        software_vector = [model.infer_vector(sentence,alpha=0.025,min_alpha=0.025, epochs=30)
                         for sentence in software_sentences]
        ransomware_vector = [model.infer_vector(sentence,alpha=0.025,min_alpha=0.025, epochs=30)
                         for sentence in ransomware_sentences]

        software_arrays = np.array(software_vector)
        software_labels = np.zeros(len(software_vector)) # 어처피 소프트웨어니까 0으로 초기화 시킬꺼임

        ransomware_arrays = np.array(ransomware_vector)
        ransomware_labels = np.ones(len(ransomware_vector)) # 어처피 악성코드니까 1으로 초기화 시킬꺼임


        #데이터 셋 합치기.
        data = np.vstack((software_arrays,ransomware_arrays))
        labels = np.hstack((software_labels,ransomware_labels))

        return data, labels