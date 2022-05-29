class _DataStore(object):

    def __init__(self):
        self.name = None
        self.data = {}
    
    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name
	
    def check_cache(self, id):
        if id in self.data:
            print('found event in cache: ' + id)
            return True
        else:
            print('did not find event in cache: ' + id)
            return False

    def get_data(self, id):
        """Return a dataframe from the data dictionary based on id"""
        if self.check_cache(id):
            print('getting event from cache: ' + id)
            return self.data[id]
        else:
            return None

    def set_data(self, id, df_event):
        """Add dataframe to cache with id as identifier"""
        print('adding event to cache: ' + id)
        self.data[id] = df_event
