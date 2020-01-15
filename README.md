# IonExplorer (beta)
# Overview
IonExplorer is a program for searching for ion migration pathways and evaluation of the ion diffusion barriers in fast ion conductors. The program is based on the topological analysis of procrystal electron density distribution. Within the approach, the lowest energy migration pathways are approximated by the gradient paths with the lowest electron density values between initial and final positions of migrating ion, while the migration barrier is attributed to the maximum value of electron density along the path. The program can be used as a tool for analysis of ion diffusion processes independently as well as a fine first approximation for the subsequent modeling approaches (e.g. DFT-based NEB calculations). Due to the low requirements on the computational resources, the algorithm allows to carry out a comprehensive analysis of ion migration pathways as well as a high-throughput searching for new crystalline ion conductors.
# Using IonExplorer
The input file for the program requires information about crystal structure written in the Crystallographic Information File format (CIF) https://www.iucr.org/resources/cif. The command to run the program is:
```
python IonExplorer.py –i input.cif
```
To see a full list of available options:
```
python IonExplorer.py –h
```

To run the program one needs to have installed:
-	Python3 https://www.python.org/downloads/release/python-373/
-	Scipy https://docs.scipy.org/doc/scipy-1.2.1/reference/
-	PyCifRW  https://pypi.org/project/PyCifRW/4.3/
-	Critic2 https://github.com/aoterodelaroza/critic2
# References and citation
P. N. Zolotarev, A. A. Golov, N. A. Nekrasova and R. A. Eremin. Topological analysis of procrystal electron densities as a tool for computational modeling of solid electrolytes: A case study of known and promising potassium conductors. AIP Conf. Proc. – 2019 – Vol. 2163, –№ 020007
# Copyright notice
Copyright (c) 2019-2020 Andrey A. Golov, Roman A. Eremin, Pavel N. Zolotarev, Nadezhda A. Nekrasova.

