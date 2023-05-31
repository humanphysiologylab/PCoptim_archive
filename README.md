# PCoptim

This distribution contains code for processing patch-clamp data via experimetnal setup mnodel optimization. Details are in the article 'Human sodium current voltage-dependence at physiological temperature measured by coupling patch-clamp experiment to a mathematical model.'

## Installation and usage

1. This code uses Sundials CVODE solver v. 6.5.1. While necessary shared and headers libraries are included in the distribution, you might want to modify path in /src/model_ctypes/Makefile according to you distribution. OpenMPI is not neccessary,but highly recommended.


2. Compile the patch-clamp model C code. 
```
cd ./src/model_ctypes
make
cd ../..
````


3. Install necessary python libraries
```
pip install numpy pandas matplotlib mpi4py tqdm
```

Conda users might want to create virtual environment from the ina_env.txt instead.
```
conda create --prefix ./ina_env --file spec-file.txt
conda activate ina_env/
```


4. Install pypoptim library from https://github.com/humanphysiologylab/pypoptim


5. Test the model
```
cd examples
python test_model.py
```


6. Test model oprimization with a single thread
```
python3 ./ga/mpi_scripts/mpi_script.py ./ga/configs/test.json 
```

Test model oprimization with 2 threads using  Open MPI.

```
mpirun -n 2 python3 ./ga/mpi_scripts/mpi_script.py ./ga/configs/test.json 
```


## Authors
Veronica Abrasheva and Roman Syunyaev
