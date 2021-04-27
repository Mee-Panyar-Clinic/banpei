import numpy as np
from scipy import stats
from banpei.base.model import BaseModel


class Hotelling(BaseModel):
    def __init__(self):
        pass

    def detect(self, data, threshold, trained_mean=None,trained_var=None):
        """
        Parameters
        ----------
        data : array_like
               Input array or object that can be converted to an array.
        threshold : float

        trained_mean (optional): float
                Input an expected mean, otherwise mean is derived from data
        trained_var (optional): float
                Input an expected std, otherwise variance is derived from data
        Returns
        -------
        List of tuples where each tuple contains index number and anomalous value.
        """
        data = self.convert_to_nparray(data)

        # Set the threshold of abnormality
        abn_th = stats.chi2.interval(1-threshold, 1)[1]

        # Covert raw data into the degree of abnormality
        if trained_mean is None:
            avg = np.average(data)
        else:
            avg = trained_mean
        
        if trained_var is None:
            var = np.var(data)
        else:
            var = trained_var
        data_abn = [(x - avg)**2 / var for x in data]

        # Return abnormality score and threshold detection
        result = []
        for i in range(len(data_abn)):
            x = data_abn[i]
            flag = 0
            if x > abn_th:
                flag = 1
            entry = [x, flag]
            result.append(entry)
        
        return np.array(result)
