import datetime
import os

def writeToLogs(message):
    date = datetime.datetime.now()
    f = open("logs.txt", "a")
    f.write(date + ": " + message + os.linesep)
    f.close()