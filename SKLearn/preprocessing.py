import csv
import sys
import numpy as np
import calendar


maxInt = sys.maxsize

while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

# inFile: full_xxx.csv dataset, outFile: xxx_processed.csv, outFile1: xxx_processed_1.csv
def preprocess(inFile, outFile, outFile1):
    with open(inFile,"r") as data:
        rdr = csv.reader(data)
        ctr = 0
        modFile =  open(outFile,"w")
        modFile1 = open(outFile1,"w")
        wtr = csv.writer(modFile)
        wtr1 = csv.writer(modFile1)
        for row in rdr:
            if ctr == 0:
                newRow = ["No. of Followers","No. of Friends","No. of Favourites","Entities","Sentiment","Mentions","Hashtags","URLs","Positive","Negative","hasHashtag","hasMentions","hasUrls","hasEntities","dt1","dt2","dt3","dt4","dt5","dt6","isWeekend","FollowerFriendRatio","FavouriteFollowerRatio","TweetLength","No. of Retweets", "hasRetweet"]
                wtr.writerow(newRow)
                ctr +=1
                continue
            rt = [int(row[2])]
            hasRT = [1 if rt != [0] else 0]
            del row[2]
            newRow = row + rt + hasRT
            wtr.writerow(newRow)
            if hasRT == [1]:
                wtr1.writerow(newRow)
            ctr += 1

        modFile.close()
        modFile1.close()

# input String object: dateString. Example "Tue Jun 30 02:12:41 +0000 2020"
# output List object: encDateTime. Format: [day_sin, day_cos, month_sin, month_cos, time_sin, time_cos]
def encodeDateTime(dateString):
    dayStr = dateString[:3]
    monStr = dateString[4:7]
    hourStr = dateString[11:13]
    minStr = dateString[14:16]
    secStr = dateString[17:19]

    dayEnc = encodeDay(dayStr)
    monEnc = encodeMon(monStr)
    timeEnc = encodeTime(hourStr, minStr, secStr)

    encDateTime = dayEnc + monEnc + timeEnc
    return encDateTime

# input Sting obj: monStr. Example: "Jan"
# output List obj: monEnc. Format: [mon_sin, mon_cos]
def encodeMon(monStr):
    monDict = {month: index for index, month in enumerate(calendar.month_abbr) if month}
    monInt = monDict[monStr]
    monEnc = cyclicEncode(monInt,12)
    return monEnc

# input Sting obj: dayStr. Example: "Tue"
# output List obj: monEnc. Format: [day_sin, day_cos]
def encodeDay(dayStr):
    dayDict = {day: index for index, day in enumerate(calendar.day_abbr) if day}
    dayInt = dayDict[dayStr]
    dayEnc = cyclicEncode(dayInt,7)
    return dayEnc

# input Sting obj: hourStr,minStr,secStr. Example: "02", "12", "41"
# output List obj: timeEnc. Format: [time_sin, time_cos]
def encodeTime(hourStr,minStr,secStr):
    hourInt = int(hourStr)
    secInt = int(minStr)*60 + int(secStr)
    secFlt = secInt/3600
    timeFlt = hourInt + secFlt
    timeEnc = cyclicEncode(timeFlt,24)
    return timeEnc

# input Int obj: num, size. Example: 2, 12. **2 to represent febuary. 12 to represent total number of months**
# output List obj: [x,y]. Format: [XX_sin, XX_cos]
def cyclicEncode(num, size):
    x = np.sin(2 * np.pi * num/size)
    y = np.cos(2 * np.pi * num/size)
    return [x,y]


if __name__ == '__main__':

    fin = "datasets/final_test.csv"
    fout = "datasets/final_test_mod.csv"
    fout2 = "datasets/final_test_mod_1.csv"
    preprocess(fin,fout,fout2)







