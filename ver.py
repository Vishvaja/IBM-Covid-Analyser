import twint
import csv
import nest_asyncio
nest_asyncio.apply()
csvfile = pd.read_csv('tweets.csv' ,engine='python')
X=csvfile.iloc[:,7].values


#print([x[0] for x in csv.reader(csvfile)])

#userx = [x[0] for x in csv.reader(csvfile)]

c = twint.Config()
for x in X:
    c.Username = x
    c.Verified=True
    c.Store_object = True
    c.User_full = True
    c.Search
    #userlist = twint.output.users_list
    


c.User_full = True
c.Output = "verified.csv"
c.Store_csv = True