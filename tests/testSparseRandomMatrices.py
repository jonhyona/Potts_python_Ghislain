from scipy.sparse import random
import scipy.sparse as spsp
from scipy import stats

class CustomRandomState(np.random.RandomState):
    def randint(self, k):
        i = np.random.randint(k)
        return i - i % 2
    
np.random.seed(12345)
rs = CustomRandomState()
rvs = stats.bernoulli(1).rvs
S = random(3, 4, density=0.25, random_state=rs, data_rvs=rvs)
print(S.A)