# meta-tagger

The meta-tagger developed for improved tagging of the GigaFida corpus by
combining the obeliks and reldi-tagger tagger outputs.

The meta-tagger was developed inside the project ARRS J6-8256: Nova slovnica 
sodobne standardne slovenščine: viri in metode.

An example input file can be found in the ```in/``` folder. The
```apply_meta.py``` script reads all files from the ```in/``` folder and
writes down disambiguated tokens in the ```out/``` folder. The order of
tagger outputs in the input files are (1) Obeliks and (2) ReLDI (i.e., the
second and third column come from the Obeliks tagger, the fourth and fifth
from the ReLDI tagger).
