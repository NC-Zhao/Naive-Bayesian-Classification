import numpy as np
# this class will classify the 10 velocity data
class CLF:
    # read two files
    def __init__(self):
        self.pdf = np.loadtxt('pdf.txt', delimiter = ',') # the pdf
        # row 0 of pdf is bird
        # row 1 of pdf is aircraft
        self.x = np.loadtxt('data.txt', delimiter = ',') # 10 velocities
        
    # normalize the pdf
    def normalize(self):
        self.pdf[1] = self.pdf[1]/sum(self.pdf[1])
        self.pdf[0] = self.pdf[0]/sum(self.pdf[0])
        return self
        
    # smooth out nan to previous number
    def cleaner(self):
        nan_mat = np.isnan(self.x)
        for i in range(self.x.shape[0]): # for each velocity data
            for j in range(self.x.shape[1]): # for each entry
                if nan_mat[i,j]: # is nan
                    if j == 0: # the first entry in the row
                        self.x[i,j] = 0
                    else:# not the first entry of the row
                        self.x[i,j] = self.x[i, j-1]
        # the velocity is doubled because the probability is given for every 0.5 velocity
        self.x = np.rint(self.x * 2).astype(int) 
        return self
    
    
    # classify the data to bird or aircraft
    # parameter: data: a row of integers representing the velocity 
    def predict(self, data):
        p_a = 0.5 # probability of being an aircraft
        p_b = 0.5 # probability of being a bird
        transition = 0.1
        for t in range(data.shape[0]):
            
            # update the probability
            new_p_b = self.pdf[0][data[t]] * (p_a * transition + p_b * (1 - transition))
            new_p_a = self.pdf[1][data[t]] * (p_b * transition + p_a * (1 - transition))
            
            if (new_p_a + new_p_b) != 0: # if both are 0, skip
                p_a = new_p_a/(new_p_a + new_p_b)
                p_b = new_p_b/(new_p_a + new_p_b)
            
        print('bird: ', p_b)
        print('aircraft: ', p_a)
        if p_b > p_a:
            return 'bird'
        else:
            return 'aircraft'
    
    def run(self):
        self.normalize()
        self.cleaner()
        for t in range(self.x.shape[0]):
            print('Object_{}'.format(t+1))
            print('Predict Object_{} to be :  {}'.format(t+1, self.predict(self.x[t])))
            print()
        return

if __name__ == '__main__':
    print('Radar Trace Classifier\n')
    clf = CLF()
    clf.run()

        