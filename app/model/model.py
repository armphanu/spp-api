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
        pred = class_model.predict(df)
        res = {"score": pred[0]}
      if type == 'ranking':
        pred = ranking_model.predict(df)
        res = {"score": pred[0]}
      if type == 'regression':
        pred = regress_model.predict(df)
        res = {"score": (2**pred[0])-1 }
    return res

