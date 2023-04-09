# import sys  
# sys.path.insert(1, '../config/constants.py')

import tweepy
import pandas as pd
import numpy as np
from constants import BEARER_TOKEN

client = tweepy.Client(wait_on_rate_limit=True, bearer_token=BEARER_TOKEN)

# Replace with time period of your choice
start_time = '2014-10-28T00:00:00Z'

# Replace with time period of your choice
end_time = '2023-02-20T00:00:00Z'

queries = ["from:KingJames -is:retweet", "from:Dame_Lillard -is:retweet",
           "from:KDTrey5 -is:retweet", "from:StephenCurry30 -is:retweet",
           "from:luka7doncic -is:retweet", "from:jaytatum0 -is:retweet",
           "from:Giannis_An34 -is:retweet", "from:kawhileonard -is:retweet", 
           "from:JHarden13 -is:retweet", "from:JoelEmbiid -is:retweet",
           "from:Money23Green -is:retweet", "from:SDinwiddie_25 -is:retweet", 
           "from:gb3elite -is:retweet","from:AndreDrummond -is:retweet", "from:BucketsONeale00 -is:retweet"]

tweets = []
for query in queries:
  try:
    result = client.search_all_tweets(query=query, tweet_fields=['author_id', 'created_at'],
                                    start_time=start_time, end_time=end_time, max_results=500)
    
    tweets.extend(result.data)
  except Exception as e:
    print(str(e))
    continue

data = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])
data['AuthorID'] = np.array([tweet.author_id for tweet in tweets])

data.to_csv('data/raw/raw_tweets.csv')

