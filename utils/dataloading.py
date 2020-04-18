from pathlib import Path
import numpy as np
import scipy.io
import pandas as pd

class dataset:
    """Dataset class.
    """

    def __init__(self, basedir, subject_list='', group=''):
        """Loads complete dataset
        """
        if not all(type(i) == str for i in [basedir, subject_list, group]):
            raise Exception(f"Input error. {basedir} is no string.")

        assert Path(basedir).exists(), f"Basedirectory {basedir} not found."
        self.dir = basedir
        self.subject_list = subject_list.split()

        if len(group) != 0:
            print('Loading all subjects of group', group)
            self.subject_list = []
            for g in group.split():
                group_dir=Path(self.dir, g)
                self.subject_list.extend([str(s)[str(s).find('sub-'):] for s in group_dir.glob('sub-*')])
        elif len(self.subject_list) == 0:  #if no subject id is specified
            print('Loading all subjects')
            self.subject_list.extend([str(s)[str(s).find('sub-'):] for s in Path(basedir).rglob('*/sub-*')]) #list of all subjects found in subdirectories

class subject:
    """Subject class.
    """
    def __init__(self, subjectID, basedir, file_dict={'Corr_Matrix':'corr_mat', 'Time_Course_Matrix':'tc'}):
        """
        Load the correct pathways to the
        """
        if type(subjectID) != str or type(basedir) != str or type(file_dict) != dict:
            raise Exception(f"Input error of {subjectID}.")

        assert Path(basedir).exists(), f"Base Directory of {subjectID} not found."
        self.basedir = basedir
        self.id = subjectID
        self.file_dict=file_dict

        assert len(list(Path(basedir).rglob(f'*/{subjectID}'))) == 1, f"Directory of {subjectID} not found."
        self.dir = list(Path(basedir).rglob(f'*/{subjectID}'))[0]

        self.group=""
        for group_name in ['SCZ', 'HC', 'SCZaff']:
            if str(self.dir).find(group_name) != -1:
                self.group=group_name
                break
        assert len(self.group) != 0, f"Group of subject {self.id} not specified."

        for f in file_dict:
            assert list(self.dir.glob(f'**/{f}*')), f"{f} of {self.id} not found."
            setattr(self, f'{file_dict[f]}_file', list(self.dir.glob(f'**/{f}*'))[0])

    def loadfiles(self):
        """
        :param file: path to .mat file that should be loaded
        :return: np.array containing data from .mat file
        """
        load_dict = {}
        for f in self.file_dict.values():
            matlab_dict = scipy.io.loadmat(getattr(self, f'{f}_file'))
            assert type(matlab_dict[f]) != np.array, "Matrix in .mat file not found. Consider key error"
            load_dict[f] = matlab_dict[f]
        return load_dict

    def conv2pd(self):
        load_dict=self.loadfiles()
        pd_dict={f:pd.DataFrame(load_dict[f]) for f in load_dict}
        return pd_dict