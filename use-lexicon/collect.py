import sys
import tweepy1 as t 
from tweepy1.parsers import ModelParser
from tweepy1 import StreamListener
from tweepy1 import Stream
import json

class PrintListener(StreamListener):
    output = None
    def on_data(self, data):
        if self.output is None:
            print data
            return True
        else:
            print>>self.output, data.strip()
            return True

    def on_error(self, status):
        print status
        
    def set_output(self, output_json):
        self.output = output_json

if __name__ == "__main__": 
    
    #set api access
    CONSUMER_KEY = 'qmE2eb259auAlikxzBbNMg'
    CONSUMER_SECRET = 'hmyCKbHjoS3Q2ec45g69SrRWwiRsHNYzYbezWp8joQ'
    ACCESS_KEY = '87219036-7C03QWtDhqyC9sWBJ6io4ySvE6HCJlHTf0EermQi0'
    ACCESS_SECRET= 'NJ8q43X6nWeZj1Sfk0vakNdckF1fnRHQ61Oekbpt7PY'
    
    auth = t.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_KEY, ACCESS_SECRET)
    api = t.API(auth_handler=auth, parser = ModelParser())
    
    to_track = ['flood crisis', 'victims', 'flood victims', 'flood powerful', 'powerful storms', 'hoisted flood', 'storms amazing', 'explosion', 'amazing rescue', 'rescue women', 'flood cost', 'counts flood', 'toll rises', 'braces river', 'river peaks', 'crisis deepens', 'prayers', 'thoughts prayers', 'affected tornado', 'affected', 'death toll', 'tornado relief', 'photos flood', 'water rises', 'toll', 'flood waters', 'flood appeal', 'victims explosion', 'bombing suspect', 'massive explosion', 'affected areas', 'praying victims', 'injured', 'please join', 'join praying', 'prayers people', 'redcross', 'text redcross', 'visiting flood', 'lurches fire', 'video explosion', 'deepens death', 'opposed flood', 'help flood', 'worsens', 'died explosions', 'marathon explosions', 'flood relief', 'donate', 'first responders', 'flood affected', 'donate cross', 'braces', 'tornado victims', 'deadly', 'prayers affected', 'explosions running', 'evacuated', 'relief', 'flood death', 'deaths confirmed', 'affected flooding', 'people killed', 'dozens', 'footage', 'survivor finds', 'worsens eastern', 'flood worsens', 'flood damage', 'people dead', 'girl died', 'flood', 'donation help', 'major flood', 'rubble', 'another explosion', 'confirmed dead', 'rescue', 'latest', 'send prayers', 'flood warnings', 'tornado survivor', 'damage', 'devastating', 'flood toll', 'affected hurricane', 'prayers families', 'releases photos', 'hundreds injured', 'inundated', 'crisis', 'text donation', 'redcross give', 'recede', 'bombing', 'massive', 'bombing victims', 'explosion ripped', 'gets donated', 'donated victims', 'relief efforts', 'news flood', 'flood emergency', 'update', 'give online', 'fire flood', 'huge explosion', 'bushfire', 'torrential rains', 'residents', 'breaking news', 'redcross donate', 'affected explosion', 'disaster', 'someone captured', 'tragedy', 'enforcement', 'people injured', 'twister', 'blast', 'crisis deepensthe', 'injuries reported', 'fatalities', 'donated million', 'donations assist', 'dead explosion', 'survivor', 'death', 'suspect dead', 'peaks deaths', 'love prayers', 'explosion fertiliser', 'explosion reported', 'return home', 'evacuees', 'large explosion', 'firefighters', 'morning flood', 'praying', 'public safety', 'txting redcross', 'destroyed', 'displaced', 'durant donated', 'fertilizer explosion', 'unknown number', 'unknown', 'donate tornado', 'retweet donate', 'flood tornado', 'casualties', 'recovery', 'climate change', 'financial donations', 'stay strong', 'dead hundreds', 'major explosion', 'bodies recovered', 'waters recede', 'medical', 'response disasters', 'victims donate', 'unaccounted', 'fire fighters', 'explosion victims', 'prayers city', 'proceeds', 'accepting financial', 'torrential', 'bomber', 'captured', 'disasters txting', 'explosion registered', 'missing flood', 'volunteers', 'brought hurricane', 'relief fund', 'help tornado', 'explosion fire', 'ravaged', 'prayers tonight', 'tragic', 'enforcement official', 'saddened', 'dealing hurricane', 'impacted', 'flood recovery', 'stream', 'dead torrential', 'flood years', 'nursing', 'recover', 'responders', 'massive tornado', 'buried alive', 'alive rubble', 'crisis rises', 'flood peak', 'homes inundated', 'flood ravaged', 'explosion video', 'killed injured', 'killed people', 'people died', 'missing explosion', 'make donation', 'floods kill', 'tornado damage', 'entire crowd', 'cross tornado', 'terrifying', 'need terrifying', 'even scary', 'cost deaths', 'facing flood', 'efforts', 'deadly explosion', 'dead missing', 'floods force', 'flood disaster', 'tornado disaster', 'medical examiner', 'help victims', 'hundreds homes', 'severe flooding', 'shocking video', 'bombing witnesses', 'magnitude', 'firefighters police', 'fire explosion', 'storm', 'flood hits', 'floodwaters', 'emergency', 'flash flood', 'flood alerts', 'crisis unfolds', 'daring rescue', 'tragic events', 'medical office', 'deadly tornado', 'people trapped', 'police officer', 'explosion voted', 'lives hurricane', 'bombings reports', 'breaking suspect', 'bombing investigation', 'praying affected', 'reels surging', 'surging floods', 'teenager floods', 'rescue teenager', 'appeal launched', 'explosion injured', 'injured explosion', 'responders killed', 'explosion caught', 'government', 'city tornado', 'help text', 'name hurricane', 'damaged hurricane', 'breaking arrest', 'suspect bombing', 'massive manhunt', 'releases images', 'shot killed', 'multiple', 'rains severely', 'house flood', 'live coverage', 'fund', 'devastating tornado', 'lost lives', 'reportedly dead', 'following explosion', 'remember lives', 'tornado flood', 'want help', 'feared', 'seconds bombing', 'reported dead', 'imminent', 'rebuild', 'safe hurricane', 'surviving', 'injuries', 'prayers victims', 'police suspect', 'warning', 'help affected', 'kills forces', 'dead floods', 'flood threat', 'military', 'flood situation', 'thousands homes', 'risk running', 'dead injured', 'dying hurricane', 'loss life', 'thoughts victims', 'bombing shot', 'breaking enforcement', 'police people', 'video capturing', 'feared dead', 'terrible explosion', 'prayers involved', 'reported injured', 'seismic', 'victims waters', 'flood homeowners', 'flood claims', 'homeowners reconnect', 'reconnect power', 'power supplies', 'rescuers help', 'free hotline', 'hotline help', 'please stay', 'investigation', 'saddened loss', 'identified suspect', 'bombings saddened', 'killed police', 'dead', 'praying thecommunity', 'registered magnitude', 'leave town', 'reported explosion', 'heart praying', 'life heart', 'prepare hurricane', 'landfall', 'crisis worsens', 'arrest', 'bombing case', 'suspect run', 'communities damaged', 'destruction', 'levy', 'tornado', 'hurricane coming', 'toxins flood', 'release toxins', 'toxins', 'supplies waters', 'crisis found', 'braces major', 'government negligent', 'attack', 'hurricane', 'climate', 'rebuilt communities', 'help rebuilt', 'rebuilt', 'rescuers', 'buried', 'heart prayers', 'flood levy', 'watch hurricane', 'victims lost', 'soldier', 'waiting hurricane', 'run massive', 'high river', 'assist', 'terror', 'memorial service', 'terror attack', 'coast hurricane', 'terrified hurricane', 'aftermath', 'suspect killed', 'suspect pinned', 'lost legs', 'prepare', 'path', 'hurricane category', 'names terrified', 'authorities', 'shocking', 'assist people', 'hurricane black', 'unknown soldier', 'events', 'safety', 'troops', 'disaster relief', 'cleanup', 'cost', 'troops lend', 'effected hurricane', 'time hurricane', 'saying hurricane', 'praying families', 'dramatic', 'path hurricane']
    print len(to_track)
    pl = PrintListener()
    pl.set_output(open("your_json_file.json","w"))
    stream = Stream(auth, pl)
    stream.filter(track=to_track)