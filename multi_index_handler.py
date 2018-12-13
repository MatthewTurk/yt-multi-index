"""
Handler for multiple types of chunks and indices put together




"""

import numpy as np
import os
import weakref
import uuid

from yt.data_objects.static_output import Dataset
from yt.utilities.logger import ytLogger as mylog
from yt.geometry.geometry_handler import Index, YTDataChunk
from yt.utilities.lib.mesh_utilities import smallest_fwidth
from yt.funcs import setdefaultattr
from yt.utilities.io_handler import BaseIOHandler
from yt.fields.field_info_container import FieldInfoContainer

class MultiIndexIOHandler(BaseIOHandler):
    _dataset_type = "multi_index"

class MultiIndex(Index):

    def __init__(self, ds, dataset_type):
        self.dataset_type = dataset_type
        self.dataset = weakref.proxy(ds)
        self.index_filename = self.dataset.parameter_filename
        self.directory = os.path.dirname(self.index_filename)
        self.float_type = np.float64
        super(MultiIndex, self).__init__(ds, dataset_type)

    def _setup_geometry(self):
        mylog.debug("Initializing Unstructured Mesh Geometry Handler.")
        self._initialize_indices()

    def get_smallest_dx(self):
        """
        Returns (in code units) the smallest cell size in the simulation.
        """
        dx = min(_.get_smallest_dx for _ in self.indices)
        return dx

    def convert(self, unit):
        return self.dataset.conversion_factors[unit]

    def _initialize_indices(self):
        self.indices = [ds.index for ftype, ds in
                sorted(self.dataset.base_datasets.items())]

    def _detect_output_fields(self):
        fields = []
        for ds_name, ds in self.dataset.base_datasets.items():
            fields.extend( [ (ds_name, _) for _ in ds.field_list ] )
        self.field_list = fields

    def _identify_base_chunk(self, dobj):
        if getattr(dobj, "_chunk_info", None) is None:
            dobj._chunk_info = self.indices
        dobj._current_chunk = list(self._chunk_all(dobj))[0]

    def _count_selection(self, dobj, indices = None):
        if indices is None: indices = dobj._chunk_info
        count = 0
        for ind in indices:
            with dobj._ds_hold(ind.ds):
                count += ind._count_selection(dobj)
        return count

    def _chunk_all(self, dobj, cache = True):
        indices = getattr(dobj._current_chunk, "objs", self.indices)
        yield YTDataChunk(dobj, "all", indices, None, cache)

    def _chunk_spatial(self, dobj, ngz, sort = None, preload_fields = None):
        indices = getattr(dobj._current_chunk, "objs", dobj._chunk_info)
        # We actually do not really use the data files except as input to the
        # ParticleOctreeSubset.
        # This is where we will perform cutting of the Octree and
        # load-balancing.  That may require a specialized selector object to
        # cut based on some space-filling curve index.
        for ind in indices:
            with dobj._ds_hold(ind.ds):
                for o in ind._chunk_spatial(dobj, ngz, sort, preload_fields):
                    yield o

    def _chunk_io(self, dobj, cache = True, local_only = False):
        indices = getattr(dobj._current_chunk, "objs", dobj._chunk_info)
        print(indices)
        for ind in indices:
            with dobj._ds_hold(ind.ds):
                for c in ind._chunk_io(dobj, cache, local_only):
                    yield c

class MultiIndexFieldInfo(FieldInfoContainer):
    known_other_fields = (
        ("density", ("code_mass/code_length**3", ["density"], None)),
        ("dark_matter_density", ("code_mass/code_length**3", ["dark_matter_density"], None)),
        ("number_density", ("1/code_length**3", ["number_density"], None)),
        ("pressure", ("dyne/code_length**2", ["pressure"], None)),
        ("thermal_energy", ("erg / g", ["thermal_energy"], None)),
        ("temperature", ("K", ["temperature"], None)),
        ("velocity_x", ("code_length/code_time", ["velocity_x"], None)),
        ("velocity_y", ("code_length/code_time", ["velocity_y"], None)),
        ("velocity_z", ("code_length/code_time", ["velocity_z"], None)),
        ("magnetic_field_x", ("gauss", [], None)),
        ("magnetic_field_y", ("gauss", [], None)),
        ("magnetic_field_z", ("gauss", [], None)),
        ("radiation_acceleration_x", ("code_length/code_time**2", ["radiation_acceleration_x"], None)),
        ("radiation_acceleration_y", ("code_length/code_time**2", ["radiation_acceleration_y"], None)),
        ("radiation_acceleration_z", ("code_length/code_time**2", ["radiation_acceleration_z"], None)),
        ("metallicity", ("Zsun", ["metallicity"], None)),

        # We need to have a bunch of species fields here, too
        ("metal_density",   ("code_mass/code_length**3", ["metal_density"], None)),
        ("hi_density",      ("code_mass/code_length**3", ["hi_density"], None)),
        ("hii_density",     ("code_mass/code_length**3", ["hii_density"], None)),
        ("h2i_density",     ("code_mass/code_length**3", ["h2i_density"], None)),
        ("h2ii_density",    ("code_mass/code_length**3", ["h2ii_density"], None)),
        ("h2m_density",     ("code_mass/code_length**3", ["h2m_density"], None)),
        ("hei_density",     ("code_mass/code_length**3", ["hei_density"], None)),
        ("heii_density",    ("code_mass/code_length**3", ["heii_density"], None)),
        ("heiii_density",   ("code_mass/code_length**3", ["heiii_density"], None)),
        ("hdi_density",     ("code_mass/code_length**3", ["hdi_density"], None)),
        ("di_density",      ("code_mass/code_length**3", ["di_density"], None)),
        ("dii_density",     ("code_mass/code_length**3", ["dii_density"], None)),
    )

    known_particle_fields = (
        ("particle_position", ("code_length", ["particle_position"], None)),
        ("particle_position_x", ("code_length", ["particle_position_x"], None)),
        ("particle_position_y", ("code_length", ["particle_position_y"], None)),
        ("particle_position_z", ("code_length", ["particle_position_z"], None)),
        ("particle_velocity", ("code_length/code_time", ["particle_velocity"], None)),
        ("particle_velocity_x", ("code_length/code_time", ["particle_velocity_x"], None)),
        ("particle_velocity_y", ("code_length/code_time", ["particle_velocity_y"], None)),
        ("particle_velocity_z", ("code_length/code_time", ["particle_velocity_z"], None)),
        ("particle_index", ("", ["particle_index"], None)),
        ("particle_gas_density", ("code_mass/code_length**3", ["particle_gas_density"], None)),
        ("particle_gas_temperature", ("K", ["particle_gas_temperature"], None)),
        ("particle_mass", ("code_mass", ["particle_mass"], None)),
        ("smoothing_length", ("code_length", ["smoothing_length"], None)),
        ("density", ("code_mass/code_length**3", ["density"], None)),
        ("temperature", ("code_temperature", ["temperature"], None)),
        ("creation_time", ("code_time", ["creation_time"], None)),
        ("age", ("code_time", [], None))
    )

class MultiDataset(Dataset):
    _index_class = MultiIndex
    _field_info_class = MultiIndexFieldInfo

    def __init__(self, base_datasets, dataset_type = "multi_index"):
        # This takes a mapping of field_types to datasets
        self.base_datasets = base_datasets
        super(MultiDataset, self).__init__("", dataset_type)

    def _parse_parameter_file(self):
        self.parameters = {}
        self.dimensionality = 3
        self.unique_identifier = uuid.uuid4().hex
        self.current_time = 0.0
        DLEs = np.array([ds.domain_left_edge.in_units("cm").d
            for _, ds in sorted(self.base_datasets.items())])
        DREs = np.array([ds.domain_right_edge.in_units("cm").d
            for _, ds in sorted(self.base_datasets.items())])
        self.domain_left_edge, self.domain_right_edge = \
                DLEs.min(axis=0), DREs.max(axis=0)
        self.periodicity = (False, False, False)

        # These attributes don't really make sense for unstructured
        # mesh data, but yt warns if they are not present, so we set
        # them to dummy values here.
        self.domain_dimensions = np.ones(3, "int32")
        self.cosmological_simulation = 0
        self.current_redshift = 0
        self.omega_lambda = 0
        self.omega_matter = 0
        self.hubble_constant = 0
        self.refine_by = 0

    def _set_code_unit_attributes(self):
        setdefaultattr(self, "length_unit", self.quan(1.0, "cm"))
        setdefaultattr(self, "mass_unit", self.quan(1.0, "g"))
        setdefaultattr(self, "time_unit", self.quan(1.0, "s"))
