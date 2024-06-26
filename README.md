These are the codes accompanying paper "Fluorescence Diffuse Optical Monitoring of Bioreactors: A Hybrid Deep Learning and Model-Based Approach for Tomography"
You will need NITFASTer toolbox and PyTorch to run the codes

In folder "NIRFASTer_FLpatch":
A few extra functions needed in simulation. Please be noted that we *may* include them in a new release of NIRFASTer, but it is not guaranteed.

In folder "SphericalInclusion":
Model shown in Fig. 3(c) left, and code for generating the forward data used in Sec. 3.1, using parameters in Table 1

In folder "CylindricalInclusion":
Model shown in Fig. 3(c) right, and code for generating the training data used in Sec. 3.2, using parameters in Table 2

In folder "NN":
The code for defining and training the network. Please change the appropriate lines to choose between to two training sets.
Also included are,
- bnd.mat: boundary of the cylindrical model, used only for visualization
- plotresults: code for visualization. Will generate figures similar to those shown in Fig. 5-7
- metrics: calculating of intersect-over-union metric on the testing set. Was used to calculate the results reported in Sec 3.1

For further questions and enquiries please contact:
- Dr. Jiaming Cao: j.cao.2@bham.ac.uk
- Prof. Hamid Dehghani: h.dehghani@bham.ac.uk
