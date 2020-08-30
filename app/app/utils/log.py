import datetime
import os

def writeToLogs(message):
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f = open("logs.txt", "a")
    f.write(date + ": " + message + os.linesep)
    f.close()
