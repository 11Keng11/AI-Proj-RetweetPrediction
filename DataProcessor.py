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

def count_hashtags_mentions(value):
    value = str(value)
    if value != "null;":
        HM_list = value.split(" ")
        HM_list = [x for x in HM_list if x]
        return len(HM_list)
    else:
        return 0

def count_URLs(value):
    value = str(value)
    if value != "null;":
        URL_list = value.split(":-:")
        URL_list = [x for x in URL_list if x]
        return len(URL_list)
    else:
        return 0

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

def processData(df):
    # remove the problem features
    df.drop(columns=["Tweet ID", "Timestamp", "Username"], inplace=True)
    # normalize the features
    df["No. of Followers"] = (df["No. of Followers"]-df["No. of Followers"].min())/(df["No. of Followers"].max()-df["No. of Followers"].min())
    df["No. of Friends"] = (df["No. of Friends"]-df["No. of Friends"].min())/(df["No. of Friends"].max()-df["No. of Friends"].min())
    df["No. of Favourites"] = (df["No. of Favourites"]-df["No. of Favourites"].min())/(df["No. of Favourites"].max()-df["No. of Favourites"].min())
    # Average the sentiment feature
    df["Sentiment"] = df["Sentiment"].apply(lambda x: sum([int(y) for y in x.split(" ")])/2)
    # count number hastags and mentions
    df["Hashtags"] = df["Hashtags"].apply(count_hashtags_mentions, 1)
    df["Mentions"] = df["Mentions"].apply(count_hashtags_mentions, 1)
    # count URLs
    df["URLs"] = df["URLs"].apply(count_URLs, 1)
    # handle entities
    # df["Entities"] = df["Entities"].apply(lambda x: extractEntityWords(x), 1)
    data = entityEmbed(df["Entities"].values)
    embedColNames = ["feat" + str(x) for x in range(50)]
    tempDf = pd.DataFrame(data, columns=embedColNames)

    # merge the df and drop the entities
    df.drop(columns=["Entities"], inplace=True)
    df = pd.concat([df, tempDf], axis=1)
    del tempDf
    print (df.shape)

    return df

def save2CSV(df, savePath):
    df.to_csv(savePath)

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
