'''
Data post processing
Prepare, Filter the data before the data is fed into the Dataloader

'''
import argparse
import pandas as pd
import os
from tqdm import tqdm
from utils import getWordEmbeddings, encodeDateTime
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
    boolean = []
    for value in tqdm(values, desc="Counting {}".format(name)):
        if value != "null;":
            data.append(sum(1 for x in [x for x in value.split(splitBy) if x]))
            boolean.append(1)
        else:
            data.append(0)
            boolean.append(0)
    return data, boolean

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

def timeProcesser(values):
    '''Function converts given datetime to a datetimeobj and
    returns the corresponding hour and weekday
    '''
    weekdays = {
                "Monday": [],
                "Tuesday": [],
                "Wednesday": [],
                "Thursday": [],
                "Friday": [],
                "Saturday": [],
                "Sunday": []
                }
    hours = {"H_{}".format(x):[] for x in range(1, 25)}
    isWeekend = []
    day_sins = []
    day_coss = []
    month_sins = []
    month_coss = []
    time_sins = []
    time_coss = []
    for value in tqdm(values, desc="Processing Timestamp"):
        datetimeObj = datetime.strptime(value, '%a %b %d %X %z %Y')
        weekday = datetimeObj.strftime("%A")
        # hour = int(datetimeObj.strftime("%H")) + 1
        weekdayEnum = {"Monday":1,
                       "Tuesday":2,
                       "Wednesday":3,
                       "Thursday":4,
                       "Friday":5,
                       "Saturday":6,
                       "Sunday":7}

        day_sin, day_cos, month_sin, month_cos, time_sin, time_cos = encodeDateTime(value)
        day_sins.append(day_sin)
        day_coss.append(day_cos)
        month_sins.append(month_sin)
        month_coss.append(month_cos)
        time_sins.append(time_sin)
        time_coss.append(time_cos)
        # for day in weekdays:
        #     if day == weekday:
        #         weekdays[day].append(1)
        #     else:
        #         weekdays[day].append(0)
        # for h in hours:
        #     if str(h) == hour:
        #         hours[h].append(1)
        #     else:
        #         hours[h].append(0)
        if weekdayEnum[weekday] < 6:
            isWeekend.append(0)
        else:
            isWeekend.append(1)
    return day_sins, day_coss, month_sins, month_coss, time_sins, time_coss, isWeekend

def getFollowerFriendRatio(followers, friends):
    data = []
    for index, follower in tqdm(enumerate(followers), total=len(followers), desc="Calculate Follower Friend Ratio"):
        friend = friends[index]
        data.append(follower/(friend+1))
    return data

def getFavouriteFollowerRatio(favourites, followers):
    data = []
    for index, favourite in tqdm(enumerate(favourites), total=len(favourites), desc="Calculate Favourite Follower Ratio"):
        follower = followers[index]
        data.append(favourite/(follower+1))
    return data

def sentimentProcesser(values):
    overall = []
    positive = []
    negative = []
    for value in tqdm(values, desc="Processing Sentiments"):
        pos = int(value.split(" ")[0])
        neg = int(value.split(" ")[1])
        overall.append(pos + neg)
        positive.append(pos)
        negative.append(neg)
    return overall, positive, negative

def countTweetLength(numHashtags, numEntities, numMentions, numUrls):
    data = []
    for index, value in tqdm(enumerate(numHashtags), total=len(numHashtags), desc="Counting Tweet Length"):
        total = value + numEntities[index] + numMentions[index] + numUrls[index]
        data.append(total)
    return data

def processData(df):
    # fill in nan values
    df.fillna('null;', inplace=True)
    # remove the problem features
    df.drop(columns=["Tweet ID", "Username"], inplace=True)
    # Sentiment Processing
    overallSentiment, positiveSentiment, negativeSentiment = sentimentProcesser(df["Sentiment"].values)
    df["Sentiment"] = overallSentiment
    df["Positive"] = positiveSentiment
    df["Negative"] = negativeSentiment
    # count number hastags, mentions, Urls and entities
    numHashtags, hasHashtag = counter(df["Hashtags"].values, "hashtags", " ")
    df["Hashtags"] = numHashtags
    df["hasHashtag"] = hasHashtag
    numMentions, hasMentions = counter(df["Mentions"].values, "mentions", " ")
    df["Mentions"] = numMentions
    df["hasMentions"] = hasMentions
    numUrls, hasUrls = counter(df["URLs"].values, "urls", ":-:")
    df["URLs"] = numUrls
    df["hasUrls"] = hasUrls
    numEntities, hasEntities = counter(df["Entities"].values, "entities", ";")
    df["Entities"] = numEntities
    df["hasEntities"] = hasEntities
    # process timestamp
    day_sin, day_cos, month_sin, month_cos, time_sin, time_cos, isWeekend = timeProcesser(df["Timestamp"].values)
    df = pd.concat([df, pd.DataFrame(day_sin), pd.DataFrame(day_cos), pd.DataFrame(month_sin), pd.DataFrame(month_cos),
    pd.DataFrame(time_sin), pd.DataFrame(time_cos)], axis="columns")
    df["isWeekend"] = isWeekend
    df.drop(columns=["Timestamp"], inplace=True) # remove the timestamp column
    # get follower friend ratio
    df["FollowerFriendRatio"] = getFollowerFriendRatio(df["No. of Followers"].values, df["No. of Friends"].values)
    # get favourite follower ratio
    df["FavouriteFollowerRatio"] = getFavouriteFollowerRatio(df["No. of Favourites"].values, df["No. of Followers"].values)
    # count the tweet length
    df["TweetLength"] = countTweetLength(df["Hashtags"].values, df["Entities"].values, df["Mentions"].values, df["URLs"].values)
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
    # check the data
    print (df.columns)
    print (df.sample(5))
    # save the df
    print ("Complete! Shape: {}".format(df.shape))
    save2CSV(df, args.output)
