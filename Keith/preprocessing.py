import csv
import sys

maxInt = sys.maxsize

while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

# Add isRetweeted? column. 1 for retweeted , 0 for not retweeted
def mod_data(inFile, outFile):
    with open(inFile,"r") as source:
        rdr = csv.reader(source)
        with open(outFile,"w") as result:
            wtr = csv.writer( result )
            rowNum = 0
            for row in rdr:
                if rowNum == 0:
                    row.append('Has Rewteet')
                else:
                    hasRetweet = 1 if (int(row[-1])> 0) else 0
                    row.append(hasRetweet)
                rowNum += 1
                wtr.writerow(row)

# Filter out only the tweets with retweets
def mod_1_data(inFile, outFile):
    with open(inFile,"r") as source:
        rdr = csv.reader(source)
        with open(outFile,"w") as result:
            wtr = csv.writer( result )
            rowNum = 0
            for row in rdr:
                if (rowNum == 0) or (int(row[-1]) > 0):
                    wtr.writerow(row)
                rowNum += 1

if __name__ == '__main__':

    # data_Fin = "datasets/filtered_test.csv"
    # data_Fout = "datasets/mod_test.csv"
    # mod_data(data_Fin, data_Fout)

    data_Fin = "datasets/filtered_test.csv"
    data_Fout = "datasets/mod_test_1.csv"
    mod_1_data(data_Fin, data_Fout)




