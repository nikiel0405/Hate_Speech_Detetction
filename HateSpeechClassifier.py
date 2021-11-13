import pandas as pd
from nltk.corpus import stopwords
import re
col_Names = ['label','tweet']

Training = pd.read_csv(r'C:\Users\nikie\PycharmProjects\Final_Project\Dataset\TRAIN.csv',usecols = col_Names)

Tweets_Train = Training.tweet.tolist()
Labels_Train = Training.label.tolist()

Test = pd.read_csv(r'C:\Users\nikie\PycharmProjects\Final_Project\Dataset\TEST.csv',usecols = col_Names)
Tweets_Test = Test.tweet.tolist()
Labels_Test = Test.label.tolist()


Tweets_TrainArray = []
Tweets_TestArray = []
print("Starting lemmas")
from nltk.stem import WordNetLemmatizer
stem = WordNetLemmatizer()
emoji_pattern = re.compile("["
         u"\U0001F600-\U0001F64F"  
         u"\U0001F300-\U0001F5FF"  
         u"\U0001F680-\U0001F6FF"  
         u"\U0001F1E0-\U0001F1FF"  
         u"\U00002702-\U000027B0"
         u"\U000024C2-\U0001F251"
         "]+", flags=re.UNICODE)

for i in range(0,len(Tweets_Train)):
    Tweet_Train = str(Tweets_Train[i])
    Tweet_Train = re.sub(r':', '', Tweet_Train)
    Tweet_Train = re.sub(r'‚Ä¶', '', Tweet_Train)
    #Tweet_Train - re.sub(r'/#\w+\s*/','', Tweet_Train)
    Tweet_Train = re.sub(r'[^\x00-\x7F]+', ' ', Tweet_Train)
    Tweet_Train = emoji_pattern.sub(r'', Tweet_Train)
    Tweet_Train = Tweet_Train.lower()
    Tweet_Train = Tweet_Train.split()
    Tweet_Train = [stem.lemmatize(t) for t in Tweet_Train]
    Tweet_Train = ' '.join(Tweet_Train)
    Tweets_TrainArray.append(Tweet_Train)

for i in range(0,len(Tweets_Test)):
    Tweet_Test = str(Tweets_Test[i])
    Tweet_Test = re.sub(r':', '', Tweet_Test)
    Tweet_Test = re.sub(r'‚Ä¶', '', Tweet_Test)
    #Tweet_Test = re.sub(r'/#\w+\s*/', '', Tweet_Test)
    Tweet_Test = re.sub(r'[^\x00-\x7F]+', ' ', Tweet_Test)
    Tweet_Test = emoji_pattern.sub(r'', Tweet_Test)
    Tweet_Test = Tweet_Test.lower()
    Tweet_Test = Tweet_Test.split()
    Tweet_Test = [stem.lemmatize(t) for t in Tweet_Test]
    Tweet_Test = ' '.join(Tweet_Test)
    Tweets_TestArray.append(Tweet_Test)

from sklearn.feature_extraction.text import CountVectorizer

print("Starting extraction")

vectorizer = CountVectorizer(max_features = 200, min_df = 50, max_df = 0.60, stop_words = stopwords.words('english'))
Tweets_Train = vectorizer.fit_transform(Tweets_TrainArray).toarray()
Tweets_Test = vectorizer.fit_transform(Tweets_TestArray).toarray()

print("Starting Transformation")

from sklearn.feature_extraction.text import TfidfTransformer
converter = TfidfTransformer()
Tweets_Train = converter.fit_transform(Tweets_Train).toarray()
Tweets_Test = converter.fit_transform(Tweets_Test).toarray()

from sklearn.ensemble import RandomForestClassifier
print("Starting training")
classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(Tweets_Train, Labels_Train)
import pickle

'''with open('classifier','wb') as picklefile:
    pickle.dump(classifier, picklefile)'''

'''with open('classifier', 'rb') as training_model:
    model = pickle.load(training_model)'''

print("Begining testing")
y_pred = classifier.predict(Tweets_Test)


from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
print("output from test subjects")
print(confusion_matrix(Labels_Test, y_pred))
print(classification_report(Labels_Test, y_pred))
print(accuracy_score(Labels_Test, y_pred))