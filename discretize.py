from datetime import datetime
#date to gap
# Input:
#    date: array of date-string
# Transform date string into gap days to datetime.now()
def GapDays(date, date_format) :
   # assert type(date) == pd.Timestamp
   delta = (pd.to_datetime(str(datetime.now()),format='%Y-%m-%d') 
            - pd.to_datetime(date, format=date_format))
   delta = delta.apply(lambda x : pd.to_timedelta(x).days)
   return delta

# discretize
# Input: 
#   V: the vector to be discretized
#   n: number of bins
#   way: currently support 'equal_width' only
# Rules: 
#   ways are stored in dict and call by way string
# Return:
#   tagv: use the cut-off as bin tags, not simply transform the digits into 
#     integers. It can be used into 
def discretize(V, n, way) :
    def equal_width(V, n) :
        tags = np.array(range(1, n+2)) * np.max(V) / n
        tagv = V.apply(lambda x : tags[x * n / np.max(V)])
        #tags = range(1,n+1) * max(V) / n
        #offset = V * n / max(V)
        #tagv = V.apply(lambda x : tags[np.floor(x * n / max(V))])
        return tagv
    def equal_freq(V, n) :
        return V        
    dic = {'equal_width': equal_width, 'equal_freq': equal_freq}
    return dic[way](V, n)
