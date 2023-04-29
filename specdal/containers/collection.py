# collection.py provides class for representing multiple
# spectra. Collection class is essentially a wrapper around
# pandas.DataFrame.
import pandas as pd
import numpy as np
from numbers import Number
from collections import OrderedDict, defaultdict
from .spectrum import Spectrum
import specdal.operators as op
from itertools import groupby
from specdal.readers import read
import copy
import logging
from os.path import abspath, expanduser, splitext
import os
import sys

logging.basicConfig(level=logging.WARNING,
        format="%(levelname)s:%(name)s:%(message)s\n")
################################################################################
# key functions for forming groups
def separator_keyfun(spectrum, separator, indices):
    elements = spectrum.name.split(separator)
    return separator.join([elements[i] for i in indices if i<len(elements)])

def separator_with_filler_keyfun(spectrum, separator, indices, filler='.'):
    elements = spectrum.name.split(separator)
    return separator.join([elements[i] if i in indices else
                           fill for i in range(len(elements))])

def df_to_collection(df, name, measure_type='pct_reflect'):
    '''
    Create a collection from a pandas.DataFrame
    
    Parameters
    ----------

    df: pandas.DataFrame
        Must have spectrum.name as index and metadata or wavelengths as columns
    
    name: string
        Name to assign to collection
    
    Returns
    -------
    c: specdal.Collection object
    
    '''
    # Sanitize as numeric:
    df = df.astype("float")
    df.columns = df.columns.astype("float")
    
    c = Collection(name=name, measure_type=measure_type)
    wave_cols, meta_cols = op.get_column_types(df)
    metadata_dict = defaultdict(lambda: None)
    if len(meta_cols) > 0:
        metadata_dict = df[meta_cols].transpose().to_dict()
    measurement_dict = df[wave_cols].transpose().to_dict('series')
    for spectrum_name in df.index:
        c.append(Spectrum(name=spectrum_name,
                          measurement=measurement_dict[spectrum_name],
                          measure_type=measure_type,
                          metadata=metadata_dict[spectrum_name]))
    return c

def proximal_join(base, rover, on='gps_time_tgt', direction='nearest'):
    '''
    Perform proximal join and return a new collection.

    Parameters
    ----------
    
    base: DataFrame or specdal.Collection object
    
    rover: DataFrame or specdal.Collection object

    Returns
    -------
    result: proximally joined dataset
        default: specdal.Collection object
        if output_df is True: pandas.DataFrame object
    '''
    result = None
    return_collection = False
    name = 'proximally_joined'
    # ensure that wavelength indices are monotonically increasing
    if (pd.Series(rover.data.index).diff()[1:] <= 0).any() or \
            (pd.Series(base.data.index).diff()[1:] <= 0).any():
        logging.error("Cannot proximally join dataset with non-increasing"
        " wavelengths. Try stitching.")
        sys.exit(1)

    if not (all([b.interpolated for b in base.spectra]) and all(
            [r.interpolated for r in rover.spectra])):
        logging.warning("Proximal join should be done on datasets interpolated "
                "to the same wavelengths.")
    if isinstance(base, Collection):
        return_collection = True
        base = base.data_with_meta(fields=[on])
    if isinstance(rover, Collection):
        return_collection = True
        name = rover.name
        rover = rover.data_with_meta(fields=[on])
    result = op.proximal_join(base, rover, on=on, direction=direction)
    if return_collection:
        result = df_to_collection(result, name=name)
    return result

################################################################################
# main Collection class
class Collection(object):
    """
    Represents a dataset consisting of a collection of spectra
    """
    def __init__(self, name, directory=None, spectra=None, ext=[".asd", ".sed", ".sig",".pico",".light"],
                 measure_type='pct_reflect', metadata=None, flags=None,
                 reader=None):
        self.name = name
        self.spectra = spectra
        self.measure_type = measure_type
        self.metadata = metadata
        self.flags = flags
        if directory:
            self.read(directory, measure_type, ext=ext, reader=reader)
    @property
    def spectra(self):
        """
        A list of Spectrum objects in the collection
        """
        return list(self._spectra.values())


    @property
    def spectra_dict(self):
        return self._spectra

    @spectra.setter
    def spectra(self, value):
        self._spectra = OrderedDict()
        if value is not None:
            # assume value is an iterable such as list
            for spectrum in value:
                assert spectrum.name not in self._spectra
                self._spectra[spectrum.name] = spectrum
    @property
    def flags(self):
        """
        A dict of flags for each spectrum in the collection
        """
        return self._flags
    @flags.setter
    def flags(self, value):
        '''
        TODO: test this
        '''
        self._flags = defaultdict(lambda: False)
        if value is not None:
            for v in value:
                if v in self._spectra:
                    self._flags[v] = True
    def flag(self, spectrum_name):
        self.flags[spectrum_name] = True

    def unflag(self, spectrum_name):
        del self.flags[spectrum_name]

    def as_flagged(self):
        """ Return a collection with just the flagged spectra """
        flags = set(self.flags)
        spectra = [s for s in self.spectra if s.name in flags]
        return Collection(self.name+'_flagged', None,
                spectra=spectra, metadata=self.metadata, flags=self.flags)
    def as_unflagged(self):
        """ Return a collection with just the flagged spectra """
        flags = set(self.flags)
        spectra = [s for s in self.spectra if not s.name in flags]
        return Collection(self.name+'_unflagged', None,
                spectra=spectra, metadata=self.metadata, flags=None)
        
    def _check_uniform_wavelengths(self):
        warning =\
"""Multiple wavelength spacings found in dataset. This may indicate input files 
from multiple datasets are being processed simultaneously, and can cause 
unpredictable behavior."""
        wavelengths0 = self.spectra[0].measurement.index
        for s in self.spectra[1:]:
            if len(s.measurement.index) != len(wavelengths0):
                logging.warning(warning)
                break
            if not (s.measurement.index == wavelengths0).all():
                logging.warning(warning)
                break

    @property
    def data(self):
        '''
        Get measurements as a Pandas.DataFrame
        '''
        try:
            self._check_uniform_wavelengths()
            objs = [s.measurement for s in self.spectra]
            keys = [s.name for s in self.spectra]
            return pd.concat(objs=objs, keys=keys, axis=1)
        except pd.core.indexes.base.InvalidIndexError as err:
            # typically from duplicate index due to overlapping wavelengths
            if not all([s.stitched for s in self.spectra]):
                logging.warning('{}: Try after stitching the overlaps'.format(err))
            raise err
        except Exception as e:
            print("Unexpected exception occurred")
            raise e

    def _unflagged_data(self):
        try:
            spectra = [s for s in self.spectra if not s.name in self.flags]
            return pd.concat(objs=[s.measurement for s in spectra],
                             axis=1, keys=[s.name for s in spectra])
        except (ValueError, pd.core.indexes.base.InvalidIndexError) as err:
            # typically from duplicate index due to overlapping wavelengths
            if not all([s.stitched for s in self.spectra]):
                logging.warning('{}: Try after stitching the overlaps'.format(err))
            return None
        except Exception as e:
            print("Unexpected exception occurred")
            raise e


    def append(self, spectrum):
        """
        insert spectrum to the collection
        """
        assert spectrum.name not in self._spectra
        assert isinstance(spectrum, Spectrum)
        self._spectra[spectrum.name] = spectrum
        
    def data_with_meta(self, data=True, fields=None):
        """
        Get dataframe with additional columns for metadata fields
        
        Parameters
        ----------
        
        data: boolean
            whether to return the measurement data or not
        
        fields: list
            names of metadata fields to include as columns.
            If None, all the metadata will be included.
        
        Returns
        -------
        pd.DataFrame: self.data with additional columns
        
        """
        if fields is None:
            fields = ['file', 'instrument_type', 'integration_time',
                      'measurement_type', 'gps_time_tgt', 'gps_time_ref',
                      'wavelength_range']
        meta_dict = {}
        for field in fields:
            meta_dict[field] = [s.metadata[field] if field in s.metadata
                                else None for s in self.spectra]
        meta_df = pd.DataFrame(meta_dict, index=[s.name for s in self.spectra])
        if data:
            result = pd.merge(meta_df, self.data.transpose(),
                              left_index=True, right_index=True)
        else:
            result = meta_df
        return result

    ##################################################
    # object methods
    def __getitem__(self, key):
        return self._spectra[key]
    def __delitem__(self, key):
        self._spectra.__delitem__(key)
        self._flags.__delitem__(key)
    def __missing__(self, key):
        pass
    def __len__(self):
        return len(self._spectra)
    def __contains__(self, item):
        self._spectra.__contains__(item)
    ##################################################
    # reader
    def read(self, directory, measure_type='pct_reflect',
             ext=[".asd", ".sed", ".sig",".pico",".light"], recursive=False,
             verbose=False, reader=None):
        """
        read all files in a path matching extension
        """
        directory = abspath(expanduser(directory))
        for dirpath, dirnames, filenames in os.walk(directory):
            if not recursive:
                # only read given path
                if dirpath != directory:
                    continue
            for f in sorted(filenames):
                f_name, f_ext = splitext(f)
                if f_ext not in list(ext):
                    # skip to next file
                    continue
                filepath = os.path.join(dirpath, f)
                try:
                    spectrum = Spectrum(name=f_name, filepath=filepath,
                                        measure_type=measure_type,
                                        verbose=verbose, reader=reader)
                    self.append(spectrum)
                except UnicodeDecodeError:
                    logging.warning("Input file {} contains non-unicode "
                                    "character. Please inspect input file.".format(
                                    f_name))
                except KeyError:
                    logging.warning("Input file {} missing metadata key. "
                                    "Please inspect input file.".format(f_name))

    ##################################################
    # Subsetter class to subset a collection
    class Subsetter:
        def __init__(self, collection, locator):
            self.collection = collection
            self.locator = locator

        def __getitem__(self, *vargs, **kwargs):
            for spectra in self.collection.spectra:
                # We get the name of the spectra
                name = spectra.name
                # We get the subset
                tmp = pd.Series(self.locator.__getitem__(*vargs, **kwargs)[name])
                # We save it as spectra
                spectra.measurement = tmp
                if isinstance(tmp, Number): 
                    spectra.metadata["wavelength_range"] = None
                else:
                    spectra.metadata["wavelength_range"] = (np.min(tmp.index),
                                        np.max(tmp.index))

            return self.collection

    @property
    def loc(self):
        return self.Subsetter(copy.deepcopy(self), self.data.loc)

    ##################################################
    # wrapper around spectral operations
    def interpolate(self, spacing=1, method='slinear'):
        '''
	'''
        for spectrum in self.spectra:
            spectrum.interpolate(spacing, method)
    def stitch(self, method='max'):
        '''
	'''
        for spectrum in self.spectra:
            try:
                spectrum.stitch(method)
            except Exception as e:
                logging.error("Error occurred while stitching {}".format(spectrum.name))
                raise e
    def jump_correct(self, splices, reference, method='additive'):
        '''
	'''
        for spectrum in self.spectra:
            spectrum.jump_correct(splices, reference, method)

    def savgol_filter(self, window_length, polyorder, deriv=0,
                    delta=1.0, axis=-1, mode='interp', cval=0.0):
        self.metadata["savgol_window_length"] = window_length
        self.metadata["savgol_polyorder"] = polyorder

        # We iterate over all spectra 
        for spectra_tmp in self.spectra:
            spectra_tmp.savgol_filter(window_length, 
                            polyorder, deriv, delta, axis, mode, cval)
            
    def normalize(self, wave="max", interpolate="False", maximum=1.0, value_norm=None):
        '''
        This methods normalizes an spectra an returns a new spectra
        '''
        c_tmp = Collection(name=self.name, metadata=self.metadata)

        if c_tmp.metadata is None:
            c_tmp.metadata = {"normalized":True}

        # We iterate over all spectra 
        for spectra_tmp in self.spectra:
            norm_tmp = spectra_tmp.normalize(wave, interpolate, 
                                             maximum, value_norm)
            c_tmp.append(norm_tmp)

        return c_tmp

    def derivative(self):
        '''
        '''
        for spectrum in self.spectra:
            spectrum.derivative()
            

    ##################################################
    # group operations
    def groupby(self, separator, indices, filler=None):
        """
        Group the spectra using a separator pattern
        
        Returns
        -------
        OrderedDict consisting of specdal.Collection objects for each group
            key: group name
            value: collection object
        
        """
        args = [separator, indices]
        key_fun = separator_keyfun
        if filler is not None:
            args.append(filler)
            key_fun = separator_with_filler_keyfun
        spectra_sorted = sorted(self.spectra,
                                  key=lambda x: key_fun(x, *args))
        groups = groupby(spectra_sorted,
                         lambda x: key_fun(x, *args))
        result = OrderedDict()
        for g_name, g_spectra in groups:
            coll = Collection(name=g_name,
                              spectra=[copy.deepcopy(s) for s in g_spectra])
            result[coll.name] = coll
        return result

    def plot(self, *args, **kwargs):
        '''
        '''
        self.data.plot(*args, **kwargs)
        pass
    def to_csv(self, *args, **kwargs):
        '''
        '''
        self.data.transpose().to_csv(*args, **kwargs)
    ##################################################
    # aggregate
    def mean(self, append=False, ignore_flagged=True):
        '''
        '''
        data =  self._unflagged_data() if ignore_flagged else data
        spectrum = Spectrum(name=self.name + '_mean',
                            measurement=data.mean(axis=1),
                            measure_type=self.measure_type)
        if append:
            self.append(spectrum)
        return spectrum
    def median(self, append=False, ignore_flagged=True):
        '''
	'''
        data =  self._unflagged_data() if ignore_flagged else data
        spectrum = Spectrum(name=self.name + '_median',
                            measurement=data.median(axis=1),
                            measure_type=self.measure_type)
        if append:
            self.append(spectrum)
        return spectrum
    def min(self, append=False, ignore_flagged=True):
        '''
	'''
        data =  self._unflagged_data() if ignore_flagged else data
        spectrum = Spectrum(name=self.name + '_min',
                            measurement=data.min(axis=1),
                            measure_type=self.measure_type)
        if append:
            self.append(spectrum)
        return spectrum
    def max(self, append=False, ignore_flagged=True):
        '''
	'''
        data =  self._unflagged_data() if ignore_flagged else data
        spectrum = Spectrum(name=self.name + '_max',
                            measurement=data.max(axis=1),
                            measure_type=self.measure_type)
        if append:
            self.append(spectrum)
        return spectrum
    def std(self, append=False, ignore_flagged=True):
        '''
	'''
        data =  self._unflagged_data() if ignore_flagged else data
        spectrum = Spectrum(name=self.name + '_std',
                            measurement=data.std(axis=1),
                            measure_type=self.measure_type)
        if append:
            self.append(spectrum)
        return spectrum

    ##################################################
    # duplicate collection
    def copy(self):
        return copy.deepcopy(self)

    ##################################################
    # method for computing the values for a specific satellite

    def getSatellite(self, satellite="Aqua", sensor="MODIS", rsr_path = __file__.replace("/containers/collection.py","/rsr/"),rsr=None):
        c_tmp = Collection(name=self.name, metadata={})
        c_tmp.metadata["satellite"] = satellite
        c_tmp.metadata["sensor"] = sensor
        # compute reflectance by bande
        size_compute = len(self.spectra)
        i = 1
        # We iterate over all spectra to compute the reflectance per band
        for spectra_tmp in self.spectra:
            # we print current spectra
            print(f"Spectra {i}/{size_compute}.", end="\r")
            c_tmp.append(spectra_tmp.getSatellite(satellite, sensor, rsr_path,rsr))
            i+=1

        return c_tmp