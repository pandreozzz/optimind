# OptimiNd
OptimiNd: An Open-Source Atmospheric Optimization Framework
===============================================================================
Setup Instructions
======================
Create and activate a virtual environment.

Tested with Python3.11  
```
% python -m venv optimind_venv && source optimind_venv/bin/activate
% pip install --upgrade pip
% pip install requirements.txt
```


Compile the shared Fortran libraries:
```
% cd optimind/libs
% for f in get_lutval vertinterp; do \
gfortran -fPIC -fopenmp -shared -o shared/f$f.so src/$f.f90 \
done
% cd ../c_libs
```

Fields are located under `data/fields`. These include cams aerosols (either prognostic output in `cams/` or climatology in `cams_clim/`), era5 data on model levels, modis Nd monthly data.
