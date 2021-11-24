# coding: utf-8
import csv

with open('./software_feature/behavior_api_order.csv', 'r', encoding='utf-8') as f:
    lines = csv.reader(f)
    software_sentences = [line for line in lines]

with open('./malware_feature/behavior_api_order.csv', 'r', encoding='utf-8') as f:
    lines = csv.reader(f)
    malware_sentences = [line for line in lines]

# # 2. TaggedDocument를 사용하여 Doc2Vec 모델을 생성할 수 있게 변환
# 모든 API 시퀀스 리스트를 일종의 문서(Document)로 생각하고 각 문서마다 태그를 입력한다.

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

tagged_data =[]
#word = ['word1', 'word2', 'word3', 'word4'....], tags = ['software'] 소프트웨어
for i,sentence in enumerate(software_sentences):
    tagged_data.append(TaggedDocument(words = sentence, tags = ['software_'+str(i)]))

#word = ['word1', 'word2', 'word3', 'word4'....], tags = ['malware'] 악성코드
for i,sentence in enumerate(malware_sentences):
    tagged_data.append(TaggedDocument(words =sentence, tags = ['malware_'+str(i)]))


#모델 파라미터 정하기.
model = Doc2Vec(vector_size=50,#300이엿음
                alpha=0.025,# 학습률. The initial learning rate.
                min_alpha=0.025,# 훈련이 진행될때마다 해당 값으로 떨어짐. Learning rate will linearly drop to min_alpha as training progresses.
                min_count=20,# 이 숫자보다 낮은 모든 단어를 무시함. Ignores all words with total frequency lower than this.
                window = 15,# 문장 내의 현재 위치와 예측 단어간의 최대 거리. The maximum distance between the current and predicted word within a sentence.
                dm =0, # distributed memory(dm) / distributed bag of words(dbow) / Defines the training algorithm. If dm=1, ‘distributed memory’ (PV-DM) is used. Otherwise, distributed bag of words (PV-DBOW) is employed.
                worker_count =6,# 사용할 스레드 개수. Use these many worker threads to train the model (=faster training with multicore machines).
                train_lbls=False#tags에 포함 학습 유무. 여기서 tags는 문장의 단어와 관련이 없으므로 False 설정
               )

model.build_vocab(tagged_data) #학습 전 빌드
model.train(tagged_data,total_examples=model.corpus_count, epochs=30)#epoch 반복 횟수 30회 많으면 좋지만, 많을수록 시간이 오래걸린다.
model_name = "Doc2vec_model"
model.save(model_name)#모델 저장하기