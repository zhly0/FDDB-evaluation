# FDDB
* The example is output from centerface_small. steps:
  *1 python test_fddb.py to get fddb detection result txt file
  *2 python FDDB.py to get two ROC txt
  *3 python runFDDB_example.py to get final ROC image
  * ![](./img/PNET_result.png)
___
### Information
* The `FDDB.exe` is build by [FDDB: Face Detection Data Set and Benchmark](http://vis-www.cs.umass.edu/fddb/) open source.
* Extra function are added to include:
  * 1. Simplify input parameters (result.txt, savepath, and index).
  * 2. Include index parameters :
    * If index is given value `-1`, then FDDB run as `original mode`
    * If index is given value with positive integer, then FDDB run as `debug mode`, that will show the output of `index` image. 
    * ![](./img/temp.png)
___
### Notice
* Before you run `FDDB.exe`, make sure to download FDDB dataset.
* Other parameter be changed in `FDDB.py`, include where annotations, original picture, and FDDB folder are located.
___
### Example
* Take a look in `runFDDB_example.py`
