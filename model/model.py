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
    month = np.load('./model/old-data/created_month_norm.npy')
    month = np.insert(month, 0, pd.to_datetime(df['created_time']).apply(getMonth)[0])
    df_nor = pd.DataFrame(month, columns = ['created_month'])
    monthNM = preprocessing.normalize([df_nor['created_month']])[0][0]


    # date
    day = np.load('./model/old-data/created_day_norm.npy')
    day = np.insert(day, 0, pd.to_datetime(df['created_time']).apply(getDate)[0])
    df_nor = pd.DataFrame(day, columns = ['created_date'])
    dayNM = preprocessing.normalize([df_nor['created_date']])[0][0]

    #hour
    hour = np.load('./model/old-data/created_hour_norm.npy')
    hour = np.insert(hour, 0, pd.to_datetime(df['created_time']).apply(getHour)[0])
    df_nor = pd.DataFrame(hour, columns = ['created_hour'])
    hourNM = (preprocessing.normalize([df_nor['created_hour']])[0][0])


    #tag
    tag = np.load('./model/old-data/post_messagetag_count_bin_norm.npy')
    tag = np.insert(tag, 0, df['message_tags'].apply(getMessageTagLength)[0])
    df_nor = pd.DataFrame(tag, columns = ['message_tags'])
    tagNM = (preprocessing.normalize([df_nor['message_tags']])[0][0])

    #message
    message = np.load('./model/old-data/post_message_count_bin_norm.npy')
    message = np.insert(message, 0, df['message'].apply(getMessageLength)[0])
    df_nor = pd.DataFrame(message, columns = ['message'])
    messageNM = (preprocessing.normalize([df_nor['message']])[0][0])

    #like_page
    pageLike = np.load('./model/old-data/page_likes_count_bin_norm.npy')
    pageLike = np.insert(pageLike, 0, df['page_like'][0])
    df_nor = pd.DataFrame(pageLike, columns = ['page_like'])
    pgNM = (preprocessing.normalize([df_nor['page_like']])[0][0])

    #follow_page
    pageFollow = np.load('./model/old-data/page_followers_count_bin_norm.npy')
    pageFollow = np.insert(pageFollow, 0, df['page_follow'][0])
    df_nor = pd.DataFrame(pageFollow, columns = ['page_follow'])
    pfNM = (preprocessing.normalize([df_nor['page_follow']])[0][0])
    my_array = np.array([[5.145066, 0.447165, messageNM,pfNM, pgNM, tagNM, dayNM, hourNM, monthNM]])

    df = pd.DataFrame(my_array, columns = ['nima','iipa','post_message_count_bin', 'page_followers_count_bin', 'page_likes_count_bin', 'post_messagetag_count_bin', 'created_day', 'created_hour', 'created_month'])
    pred = model.predict(df)
    return pred[0]

# print(predict_pipeline('2022-09-05T11:00:04+0000', "['#GlicoIceTH', '#GiantConeCrown', '#foodbymay', '#icecream']", 'test helloword t1112134 heeellloooo its me', 1000000, 222223121313133))


# data = {
#   "created_time": ['2022-09-05T11:00:04+0000'],
#   "message_tags": ["['#GlicoIceTH', '#GiantConeCrown', '#foodbymay', '#icecream']"],
#   "message": ['ครั้งแรกของมอคค่าในรูปแบบโคน!!! /n ใหม่ ไจแอนท์ โคน คราวน์ “มอคค่า อัลมอนด์” 🍦☕️🍫🥜🔥 เกินต้านให้ทิสุดในใจตอนนี้ ไอศกรีมเป็นรสมอคค่าแบบพรีเมียมแทรกช็อกโกแลตชิพกรุบ ท็อปด้วยดาร์กช็อกโกแลตดิสก์ และอัลมอนด์แท้จากอเมริกาใส่มาแบบแน่นๆบอกเลยว่าเคี้ยวกรุบกรอบเพลินม๊วกกก 🥜😋 และฟินให้สุดแบบฉุดไม่อยู่ด้วยดาร์กช็อกโกแลตแท้ที่ปลายโคนวานิลลากรอบ โคตรดีย์ สุดทุกคำสุดทุกโคน 🍫❤️ ดีต่อใจไม่ไหว!! ซื้อตุนเก็บไว้ในตู้เย็น เวลาไหนก็ฟินได้ 😍 อร่อย…ตื่นต๊าชชช ☕️🤩 บ่ายไหนก็ไม่มีน็อค กับความหอมหวาน ความเข้มข้นของไอศกรีมมอคค่า ลองเล๊ยยยย 💰ราคา 35 บาท/แท่ง📍พิกัด : หาซื้อมาได้แล้ววันนี้ที่  7-Eleven'],
#   "page_like": 1000000,
#   "page_follow": 2022222

# }

# df = pd.DataFrame(data)

# # month
# month = np.load('./old-data/created_month_norm.npy')
# month = np.insert(month, 0, pd.to_datetime(df['created_time']).apply(getMonth)[0])
# df_nor = pd.DataFrame(month, columns = ['created_month'])
# monthNM = preprocessing.normalize([df_nor['created_month']])[0][0]


# # date
# day = np.load('./old-data/created_day_norm.npy')
# day = np.insert(day, 0, pd.to_datetime(df['created_time']).apply(getDate)[0])
# df_nor = pd.DataFrame(day, columns = ['created_date'])
# dayNM = preprocessing.normalize([df_nor['created_date']])[0][0]

# #hour
# hour = np.load('./old-data/created_hour_norm.npy')
# hour = np.insert(hour, 0, pd.to_datetime(df['created_time']).apply(getHour)[0])
# df_nor = pd.DataFrame(hour, columns = ['created_hour'])
# hourNM = (preprocessing.normalize([df_nor['created_hour']])[0][0])


# #tag
# tag = np.load('./old-data/post_messagetag_count_bin_norm.npy')
# tag = np.insert(tag, 0, df['message_tags'].apply(getMessageTagLength)[0])
# df_nor = pd.DataFrame(tag, columns = ['message_tags'])
# tagNM = (preprocessing.normalize([df_nor['message_tags']])[0][0])

# #message
# message = np.load('./old-data/post_message_count_bin_norm.npy')
# message = np.insert(message, 0, df['message'].apply(getMessageLength)[0])
# df_nor = pd.DataFrame(message, columns = ['message'])
# messageNM = (preprocessing.normalize([df_nor['message']])[0][0])

# #like_page
# pageLike = np.load('./old-data/page_likes_count_bin_norm.npy')
# pageLike = np.insert(pageLike, 0, df['page_like'][0])
# df_nor = pd.DataFrame(pageLike, columns = ['page_like'])
# pgNM = (preprocessing.normalize([df_nor['page_like']])[0][0])

# #follow_page
# pageFollow = np.load('./old-data/page_followers_count_bin_norm.npy')
# pageFollow = np.insert(pageFollow, 0, df['page_follow'][0])
# df_nor = pd.DataFrame(pageFollow, columns = ['page_follow'])
# pfNM = (preprocessing.normalize([df_nor['page_follow']])[0][0])