import numpy as np
import yt
import multi_index_handler

px1 = np.random.normal( size = 128**3 )
py1 = np.random.normal( size = 128**3 )
pz1 = np.random.normal( size = 128**3 )

bbox1 = np.array([[px1.min(), px1.max()],
                  [py1.min(), py1.max()],
                  [pz1.min(), pz1.max()]])

px2 = np.random.normal( size = 128**3, loc = 10.0 )
py2 = np.random.normal( size = 128**3, loc = 10.0 )
pz2 = np.random.normal( size = 128**3, loc = 10.0 )

bbox2 = np.array([[px2.min(), px2.max()],
                  [py2.min(), py2.max()],
                  [pz2.min(), pz2.max()]])

print(bbox1, bbox2)

ds1 = yt.load_particles(
        {'particle_position_x' : px1,
         'particle_position_y' : py1,
         'particle_position_z' : pz1,
         'particle_mass' : np.ones(128**3)}, 1.0, 
        bbox = bbox1)

ds2 = yt.load_particles(
        {'particle_position_x' : px2,
         'particle_position_y' : py2,
         'particle_position_z' : pz2,
         'particle_mass' : np.ones(128**3)}, 1.0, 
        bbox = bbox2)

ds = multi_index_handler.MultiDataset( {'ds1': ds1, 'ds2': ds2} )

ds.index
dd = ds.all_data()

for c in dd.chunks([], "io"):
    print(c["particle_ones"].size)

print(c["particle_ones"].size)
