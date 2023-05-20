import pickle
from pathlib import Path


####import####
import datetime
import pandas as pd
import numpy as np
import ast
import ast
from pythainlp import word_tokenize
from sklearn import preprocessing
__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent


with open(f"{BASE_DIR}/binary.pkl", "rb") as f:
    model = pickle.load(f)
with open(f"{BASE_DIR}/cku24g11.pkl", "rb") as f:
    ranking_model = pickle.load(f)
with open(f"{BASE_DIR}/cu29g16.pkl", "rb") as f:
    class_model = pickle.load(f)
with open(f"{BASE_DIR}/ru12.pkl", "rb") as f:
    regress_model = pickle.load(f)


def conToDate(date_str):
  # df_all_first_copy['created_time'] = df_all_first_copy['created_time'].map(lambda x: pd.to_datetime(x).tz_convert('Asia/Bangkok'))
  date_1 = datetime.strptime('2022-09-08', '%Y-%m-%d').date()
  date_2 = datetime.strptime(str(date_str).split()[0], '%Y-%m-%d').date()
  delta = date_1 - date_2
  return delta.days

def getDate(dateStr, option='normal'):
  date = dateStr.tz_convert('Asia/Bangkok')
  if option == "normal":
    return date.day
  # day of week
  if option == "dof":
    return date.isoweekday()
  # weekend
  if option == "weekend":
    return date.isoweekday()
  
def getMonth(dateStr, option='normal'):
  date = dateStr.tz_convert('Asia/Bangkok')
  if option == "normal":
    return date.month

def getHour(dateStr, option='normal'):
  date = dateStr.tz_convert('Asia/Bangkok')
  if option == "normal":
    return date.hour
  
def tokenText(text):
  return word_tokenize(text, engine="longest")

def getMessageTagLength(tagStr, option=""):
  tagList = ast.literal_eval(tagStr)
  tagList = [n.strip() for n in tagList]
  if option == "count":
    return len(tagList)
  else:
    tokenTag = [tokenText(item) for item in tagList]
    if len(tokenTag) > 0:
      rr = 0
      for j in range(len(tokenTag)):
        if "#" in tokenTag[j]:
          removeHash = tokenTag[j].remove("#")
        rr += (len(tokenTag[j]))
      tokenTag = rr
    else:
      rr = 0
    return rr
  
def getMessageLength(message):
  tokenMsg = tokenText(message)
  return len(tokenMsg)



def predict_pipeline(created_time, message_tags, msg, pl, pg):
    data = {
      "created_time": [created_time],
      "message_tags": [message_tags],
      "message": [msg],
      "page_like": [pl],
      "page_follow": [pg]

    }

    df = pd.DataFrame(data)

    # month
    month = np.load('/app/app/model/old-data/created_month_norm.npy')
    month = np.insert(month, 0, pd.to_datetime(df['created_time']).apply(getMonth)[0])
    df_nor = pd.DataFrame(month, columns = ['created_month'])
    monthNM = preprocessing.normalize([df_nor['created_month']])[0][0]


    # date
    day = np.load('/app/app/model/old-data/created_day_norm.npy')
    day = np.insert(day, 0, pd.to_datetime(df['created_time']).apply(getDate)[0])
    df_nor = pd.DataFrame(day, columns = ['created_date'])
    dayNM = preprocessing.normalize([df_nor['created_date']])[0][0]

    #hour
    hour = np.load('/app/app/model/old-data/created_hour_norm.npy')
    hour = np.insert(hour, 0, pd.to_datetime(df['created_time']).apply(getHour)[0])
    df_nor = pd.DataFrame(hour, columns = ['created_hour'])
    hourNM = (preprocessing.normalize([df_nor['created_hour']])[0][0])


    #tag
    tag = np.load('/app/app/model/old-data/post_messagetag_count_bin_norm.npy')
    tag = np.insert(tag, 0, df['message_tags'].apply(getMessageTagLength)[0])
    df_nor = pd.DataFrame(tag, columns = ['message_tags'])
    tagNM = (preprocessing.normalize([df_nor['message_tags']])[0][0])

    #message
    message = np.load('/app/app/model/old-data/post_message_count_bin_norm.npy')
    message = np.insert(message, 0, df['message'].apply(getMessageLength)[0])
    df_nor = pd.DataFrame(message, columns = ['message'])
    messageNM = (preprocessing.normalize([df_nor['message']])[0][0])

    #like_page
    pageLike = np.load('/app/app/model/old-data/page_likes_count_bin_norm.npy')
    pageLike = np.insert(pageLike, 0, df['page_like'][0])
    df_nor = pd.DataFrame(pageLike, columns = ['page_like'])
    pgNM = (preprocessing.normalize([df_nor['page_like']])[0][0])

    #follow_page
    pageFollow = np.load('/app/app/model/old-data/page_followers_count_bin_norm.npy')
    pageFollow = np.insert(pageFollow, 0, df['page_follow'][0])
    df_nor = pd.DataFrame(pageFollow, columns = ['page_follow'])
    pfNM = (preprocessing.normalize([df_nor['page_follow']])[0][0])
    my_array = np.array([[5.145066, 0.447165, messageNM,pfNM, pgNM, tagNM, dayNM, hourNM, monthNM]])

    df = pd.DataFrame(my_array, columns = ['nima','iipa','post_message_count_bin', 'page_followers_count_bin', 'page_likes_count_bin', 'post_messagetag_count_bin', 'created_day', 'created_hour', 'created_month'])
    pred = model.predict(df)
    if pred[0] == '0':
      res = {"message": "คอนเทนต์ของคุณอาจจะมีค่าความนิยมไม่เกิน 50"}
    else:
      if(type == 'classification'):
        with open(f"{BASE_DIR}/classification-model/cu1g11.pkl", "rb") as f:
            cu1g11 = pickle.load(f)
        with open(f"{BASE_DIR}/classification-model/cu2g11.pkl", "rb") as f:
            cu2g11 = pickle.load(f)
        with open(f"{BASE_DIR}/classification-model/cu3g11.pkl", "rb") as f:
            cu3g11 = pickle.load(f)
        with open(f"{BASE_DIR}/classification-model/cu4g11.pkl", "rb") as f:
            cu4g11 = pickle.load(f)
        with open(f"{BASE_DIR}/classification-model/cu5g11.pkl", "rb") as f:
            cu5g11 = pickle.load(f)
        with open(f"{BASE_DIR}/classification-model/cu6g11.pkl", "rb") as f:
            cu6g11 = pickle.load(f)
        with open(f"{BASE_DIR}/classification-model/cu7g11.pkl", "rb") as f:
            cu7g11 = pickle.load(f)
        with open(f"{BASE_DIR}/classification-model/cu8g11.pkl", "rb") as f:
            cu8g11 = pickle.load(f)
        with open(f"{BASE_DIR}/classification-model/cu9g11.pkl", "rb") as f:
            cu9g11 = pickle.load(f)
        with open(f"{BASE_DIR}/classification-model/cu10g11.pkl", "rb") as f:
            cu10g11 = pickle.load(f)
        with open(f"{BASE_DIR}/classification-model/cu11g11.pkl", "rb") as f:
            cu11g11 = pickle.load(f)
        with open(f"{BASE_DIR}/classification-model/cu12g11.pkl", "rb") as f:
            cu12g11 = pickle.load(f)
        with open(f"{BASE_DIR}/classification-model/cu13g11.pkl", "rb") as f:
            cu13g11 = pickle.load(f)
        with open(f"{BASE_DIR}/classification-model/cu14g11.pkl", "rb") as f:
            cu14g11 = pickle.load(f)
        pred = class_model.predict(df)

        pred1 = class_model.predict(cu1g11)
        pred2 = class_model.predict(cu2g11)
        pred3 = class_model.predict(cu3g11)
        pred4 = class_model.predict(cu4g11)
        pred5 = class_model.predict(cu5g11)
        pred6 = class_model.predict(cu6g11)
        pred7 = class_model.predict(cu7g11)
        pred8 = class_model.predict(cu8g11)
        pred9 = class_model.predict(cu9g11)
        pred10 = class_model.predict(cu10g11)
        pred11 = class_model.predict(cu11g11)
        pred12 = class_model.predict(cu12g11)
        pred13 = class_model.predict(cu13g11)
        pred14 = class_model.predict(cu14g11)

        res = {"score": pred[0]}
        res["dataList"] = [pred1[0], pred2[0], pred3[0], pred4[0], pred5[0], pred6[0], pred7[0], pred8[0], pred9[0], pred10[0], pred11[0], pred12[0], pred13[0], pred14[0]]
      if type == 'ranking':
        with open(f"{BASE_DIR}/ranking-model/cku1g11.pkl", "rb") as f:
           cku1g11 = pickle.load(f)
        with open(f"{BASE_DIR}/ranking-model/cku2g11.pkl", "rb") as f:
           cku2g11 = pickle.load(f)
        with open(f"{BASE_DIR}/ranking-model/cku3g11.pkl", "rb") as f:
           cku3g11 = pickle.load(f)
        with open(f"{BASE_DIR}/ranking-model/cku4g11.pkl", "rb") as f:
           cku4g11 = pickle.load(f)
        with open(f"{BASE_DIR}/ranking-model/cku5g11.pkl", "rb") as f:
           cku5g11 = pickle.load(f)
        with open(f"{BASE_DIR}/ranking-model/cku6g11.pkl", "rb") as f:
           cku6g11 = pickle.load(f)
        with open(f"{BASE_DIR}/ranking-model/cku7g11.pkl", "rb") as f:
           cku7g11 = pickle.load(f)
        with open(f"{BASE_DIR}/ranking-model/cku8g11.pkl", "rb") as f:
           cku8g11 = pickle.load(f)
        with open(f"{BASE_DIR}/ranking-model/cku9g11.pkl", "rb") as f:
           cku9g11 = pickle.load(f)
        with open(f"{BASE_DIR}/ranking-model/cku10g11.pkl", "rb") as f:
           cku10g11 = pickle.load(f)
        with open(f"{BASE_DIR}/ranking-model/cku11g11.pkl", "rb") as f:
           cku11g11 = pickle.load(f)
        with open(f"{BASE_DIR}/ranking-model/cku12g11.pkl", "rb") as f:
           cku12g11 = pickle.load(f)
        with open(f"{BASE_DIR}/ranking-model/cku13g11.pkl", "rb") as f:
           cku13g11 = pickle.load(f)
        with open(f"{BASE_DIR}/ranking-model/cku14g11.pkl", "rb") as f:
           cku14g11 = pickle.load(f)
        pred = ranking_model.predict(df)

        pred1 = class_model.predict(cku1g11)
        pred2 = class_model.predict(cku2g11)
        pred3 = class_model.predict(cku3g11)
        pred4 = class_model.predict(cku4g11)
        pred5 = class_model.predict(cku5g11)
        pred6 = class_model.predict(cku6g11)
        pred7 = class_model.predict(cku7g11)
        pred8 = class_model.predict(cku8g11)
        pred9 = class_model.predict(cku9g11)
        pred10 = class_model.predict(cku10g11)
        pred11 = class_model.predict(cku11g11)
        pred12 = class_model.predict(cku12g11)
        pred13 = class_model.predict(cku13g11)
        pred14 = class_model.predict(cku14g11)

        res = {"score": sum(pred[0])}
        res["dataList"] = [sum(pred1[0]), sum(pred2[0]), sum(pred3[0]), sum(pred4[0]), sum(pred5[0]), sum(pred6[0]), sum(pred7[0]), sum(pred8[0]), sum(pred9[0]), sum(pred10[0]), sum(pred11[0]), sum(pred12[0]), sum(pred13[0]), sum(pred14[0])]
      if type == 'regression':
        with open(f"{BASE_DIR}/regression-model/ru1.pkl", "rb") as f:
           ru1 = pickle.load(f)
        with open(f"{BASE_DIR}/regression-model/ru2.pkl", "rb") as f:
           ru2 = pickle.load(f)
        with open(f"{BASE_DIR}/regression-model/ru3.pkl", "rb") as f:
           ru3 = pickle.load(f)
        with open(f"{BASE_DIR}/regression-model/ru4.pkl", "rb") as f:
           ru4 = pickle.load(f)
        with open(f"{BASE_DIR}/regression-model/ru5.pkl", "rb") as f:
           ru5 = pickle.load(f)
        with open(f"{BASE_DIR}/regression-model/ru6.pkl", "rb") as f:
           ru6 = pickle.load(f)
        with open(f"{BASE_DIR}/regression-model/ru7.pkl", "rb") as f:
           ru7 = pickle.load(f)
        with open(f"{BASE_DIR}/regression-model/ru8.pkl", "rb") as f:
           ru8 = pickle.load(f)
        with open(f"{BASE_DIR}/regression-model/ru9.pkl", "rb") as f:
           ru9 = pickle.load(f)
        with open(f"{BASE_DIR}/regression-model/ru10.pkl", "rb") as f:
           ru10 = pickle.load(f)
        with open(f"{BASE_DIR}/regression-model/ru11.pkl", "rb") as f:
           ru11 = pickle.load(f)
        with open(f"{BASE_DIR}/regression-model/ru12.pkl", "rb") as f:
           ru12 = pickle.load(f)
        with open(f"{BASE_DIR}/regression-model/ru13.pkl", "rb") as f:
           ru13 = pickle.load(f)
        with open(f"{BASE_DIR}/regression-model/ru14.pkl", "rb") as f:
           ru14 = pickle.load(f)
        pred = regress_model.predict(df)
        res = {"score": (2**pred[0])-1 }
        pred1 = class_model.predict(ru1)
        pred2 = class_model.predict(ru2)
        pred3 = class_model.predict(ru3)
        pred4 = class_model.predict(ru4)
        pred5 = class_model.predict(ru5)
        pred6 = class_model.predict(ru6)
        pred7 = class_model.predict(ru7)
        pred8 = class_model.predict(ru8)
        pred9 = class_model.predict(ru9)
        pred10 = class_model.predict(ru10)
        pred11 = class_model.predict(ru11)
        pred12 = class_model.predict(ru12)
        pred13 = class_model.predict(ru13)
        pred14 = class_model.predict(ru14)
        
        res["dataList"] = [pred1[0], pred2[0], pred3[0], pred4[0], pred5[0], pred6[0], pred7[0], pred8[0], pred9[0], pred10[0], pred11[0], pred12[0], pred13[0], pred14[0]]
    return res

