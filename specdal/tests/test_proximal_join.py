import os
import sys
import numpy as np
import pandas as pd
import pandas.util.testing as pdt
import unittest
from collections import OrderedDict
sys.path.insert(0, os.path.abspath("../../"))
from specdal.operator import proximal_join
from specdal.spectrum import Spectrum

class proximalTests(unittest.TestCase):
    def setUp(self):
        pass
    def test_proximal_join(self):

        rover_df = pd.DataFrame({'gps_time_tgt':[1, 4, 6, 8],
                                 1:[100, 100, 100, 100],
                                 2:[100, 100, 100, 100],
                                 3:[100, 100, 100, 100]},
                                index=pd.Index(['s1', 's2', 's3', 's4']))
        base_df = pd.DataFrame({'gps_time_tgt':[1, 3, 7, 10],
                                1:[2, 2, 2, 2],
                                2:[50, 50, 50, 10],
                                3:[1, 2, 3, 4]},
                            index=pd.Index(['s1', 's2', 's3', 's4']))
        proximal_join(base_df, rover_df)
        # proximal_df_correct = pd.DataFrame({'s4':[1.0, 1.0, 1.0, 1.0],
        #                                     's5':[1.0, 1.0, 1.0, 1.0],
        #                                     's6':[1.0, 1.0, 1.0, 1.0]},
        #                                    index=pd.Index([1, 2, 3, 4],
        #                                                   name='wavelength'))
        # pdt.assert_frame_equal(proximal.data, proximal_df_correct)
        


def main():
    unittest.main()
            
if __name__ == "__main__":
    main()
