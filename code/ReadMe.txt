=======================================================================
POINCLOUD ANIMATION CODE

The library numpy and matplotlib are needed.

The code creates the single frames of every object in the folder "pointclouds", and then joins them in the final frames of the gridded pointclouds in a folder inside it called "sets".

The number of sets can be changed inside the code, they are now set to low numbers so that the output does not occupy too much space. 

To replicate the animation shown in the report "number_of_sets" needs to be set to 20 and "number_of_frames" to 30

=======================================================================
RF CODE

Open the folder named "RF code" in an IDE to ensure proper working code

all empty folders are available now

the RF part of the code relies on multiple python libraries that are not part of the standard libraries:

scipy
sklearn
matplotlib
numpy
seaborn
statsmodels
mpl_toolkits

these will have to be installed either through pip or anaconda

The libraries that are part of the standard libraries:

glob
random

for these no actions have to be taken

As requested none of the pointcloud data is included in the folder as such these have to be manually added to the folder "data\pointclouds\"

The file is a jupyter notebook which is capable of running the entire code in about 25 minutes. (enjoy a cup of coffee :) )

by the end all images used in the report are recreated and stored in the folders

=======================================================================
SVM CODE

Open the code file named "SVM_code" in an IDE to ensure proper working code

the RF part of the code relies on multiple python libraries that are not part of the standard libraries:

scipy
sklearn
matplotlib
numpy
pandas
seaborn
statsmodels
mpl_toolkits
itertools

these will have to be installed either through pip or anaconda

The libraries that are part of the standard libraries:

glob
random

for these no actions have to be taken

As requested none of the pointcloud data is included in the folder as such these have to be manually added to the folder "data\pointclouds\"

The file is a jupyter notebook which is capable of running the entire code in about 25 minutes. (enjoy a cup of coffee :) )

by the end all images used in the report are recreated and can be view in the IDE file itself..