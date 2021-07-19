'''
Data post processing
Prepare, Filter the data before the data is fed into the Dataloader

'''
import argparse
import pandas as pd
import os
from tqdm import tqdm
from utils import getWordEmbeddings
import numpy as np
from datetime import datetime

#========== ARGPARSE BLOCK ==========#
def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg

def is_free_file(parser, arg):
    savePath = os.path.join("datasets", arg + ".csv")
    if os.path.exists(savePath):
        print ("{} already exists!".format(arg))
        ui = input("Overwrite? (Y/N) > ")
        if ui.upper() == "Y":
            os.remove(savePath)
            return savePath
        else:
            exit()
    else:
        return savePath


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="input csv file for processing", type=lambda x: is_valid_file(parser, x), required=True)
parser.add_argument("-o", "--output", help="output csv file for dataloading", type=lambda x: is_free_file(parser, x), required=True)

def parserSummary(args):
    print ("Data Processor.")
    print ("Input File: {}".format(args.input))
    print ("Output File: {}".format(args.output))

#======= END OF ARGPARSE BLOCK =======#

def loadData(filePath):
    if "train" in filePath:
        totalRows = 13978697
    elif "val" in filePath:
        totalRows = 2995436
    elif "test" in filePath:
        totalRows = 2995435
    column_names = ["Tweet ID", "Username", "Timestamp", "No. of Followers", "No. of Friends", "No. of Retweets",
                "No. of Favourites", "Entities", "Sentiment", "Mentions", "Hashtags", "URLs"]
    with tqdm(total=totalRows, desc="Loading {} data".format(filePath)) as bar:
        df = pd.read_csv(filePath, header=0, names=column_names, skiprows=lambda x: bar.update(1) and False)

    return df

def counter(values, name, splitBy):
    '''Function returns the number of occurences'''
    data = []
    for value in tqdm(values, desc="Counting {}".format(name)):
        if value != "null;":
            data.append(sum(1 for x in [x for x in value.split(splitBy) if x]))
        else:
            data.append(0)
    return data

# def count_hashtags_mentions(values):
#     for value in tqdm(values, desc="Counting":
#         value = str(value)
#         if value != "null;":
#             HM_list = value.split(" ")
#             HM_list = [x for x in HM_list if x]
#             return len(HM_list)
#         else:
#             return 0

# def count_URLs(value):
#     value = str(value)
#     if value != "null;":
#         URL_list = value.split(":-:")
#         URL_list = [x for x in URL_list if x]
#         return len(URL_list)
#     else:
#         return 0

# def entityEmbed(embeddings, value):
#     if value != "null;":
#         entities_list = value.split(";")
#         entities_list = [x for x in entities_list if x]
#         numEntities = len(entities_list)
#         embed = np.zeros((1, 50)).astype(np.float32)
#         for e in entities_list:
#             root = e.split(":")[0]
#             if root in embeddings:
#                 embed += embeddings[root]
#             else:
#                 embed += np.random.rand(1, 50).astype(np.float32)
#         return embed.flatten()
#     else:
#         return np.random.rand(1,50).astype(np.float32).flatten()

def entityEmbed(value):
    embeddings = getWordEmbeddings()
    # loop through the values
    data = []
    for v in tqdm(value, desc="Assigning Word Embeddings"):
        entities = extractEntityWords(v)
        if entities != "null;":
            # no entity -> fill in zeros
            d = np.zeros(50).tolist()
        else:
            # get the word embeddings
            d = np.zeros(50)
            count = 0
            for entity in entities:
                for word in entity:
                    count += 1
                    if word in embeddings:
                        d += embeddings[word]
                    else:
                        d += np.random.rand(50)
            d /= count
            d = d.tolist()

        data.append(d)
    return data




def extractEntityWords(value):
    if value != "null;":
        entities_list = value.split(";")
        entities_list = [x for x in entities_list if x]
        numEntities = len(entities_list)
        entities = []
        for a in entities_list:
            entities.append(a.split(":")[0])
        return ":".join(entities)
    else:
        return "null;"

def getWeekday(values):
    '''Function converts given datetime to a datetimeobj and
    returns the corresponding weekday class
    '''
    data = []
    for value in tqdm(values, desc="Converting to weekday"):
        datetimeObj = datetime.strptime(value, '%a %b %d %X %z %Y')
        weekday = datetimeObj.strftime("%A")
        weekdayEnum = {"Monday":0,
                       "Tuesday":1,
                       "Wednesday":2,
                       "Thursday":3,
                       "Friday":4,
                       "Saturday":5,
                       "Sunday":6}
        data.append(weekdayEnum[weekday])
    return pd.Series(weekdayEnum[weekday])

def getFollowerFriendRatio(followers, friends):
    data = []
    for index, follower in tqdm(enumerate(followers), total=len(followers), desc="Calculate Follower Friend Ratio"):
        friend = friends[index]
        data.append(follower/(friend+1))
    return data

def processData(df):
    # fill in nan values
    df.fillna('null;', inplace=True)
    # remove the problem features
    df.drop(columns=["Tweet ID", "Username"], inplace=True)
    # Average the sentiment feature
    df["Sentiment"] = df["Sentiment"].apply(lambda x: sum([int(y) for y in x.split(" ")])/2)
    # count number hastags, mentions, Urls and entities
    df["Hashtags"] = counter(df["Hashtags"].values, "hashtags", " ")
    df["Mentions"] = counter(df["Mentions"].values, "mentions", " ")
    df["URLs"] = counter(df["URLs"].values, "urls", ":-:")
    df["Entities"] = counter(df["Entities"].values, "entities", ";")
    # get weekday timestamp
    df["Timestamp"] = getWeekday(df["Timestamp"].values)
    # get follower friend ratio
    df["FollowerFriendRatio"] = getFollowerFriendRatio(df["No. of Followers"].values, df["No. of Friends"])
    # normalize the features
    df["No. of Followers"] = (df["No. of Followers"]-df["No. of Followers"].min())/(df["No. of Followers"].max()-df["No. of Followers"].min())
    df["No. of Friends"] = (df["No. of Friends"]-df["No. of Friends"].min())/(df["No. of Friends"].max()-df["No. of Friends"].min())
    df["No. of Favourites"] = (df["No. of Favourites"]-df["No. of Favourites"].min())/(df["No. of Favourites"].max()-df["No. of Favourites"].min())

    return df

def save2CSV(df, savePath):
    chunks = np.array_split(df.index, 100)
    for index, data in tqdm(enumerate(chunks), total=100, desc="Saving to {}".format(savePath)):
        if index == 0:
            df.loc[data].to_csv(savePath, mode="w", index=False)
        else:
            df.loc[data].to_csv(savePath, mode="a", header=None, index=False)

if __name__ == "__main__":
    args = parser.parse_args()
    parserSummary(args)
    # load in df
    df = loadData(args.input)
    # process the df
    df = processData(df)
    # save the df
    print ("Complete! Shape: {}".format(df.shape))
    save2CSV(df, args.output)
