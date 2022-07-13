# PBTS-2D: Pose-Based Tactile Servo Control (2D)

A Python library for tactile servo control using pose models built from convolutional neural networks for optical tactile sensors (TacTip, DigiTac, DIGIT) 

Methods described in  
DigiTac: A DIGIT-TacTip Hybrid Tactile Sensor for Comparing Low-Cost High-Resolution Robot Touch  
N Lepora, Y Lin, B Money-Coomes, J Lloyd (2022) IEEE Robotics & Automation Letters    
https://arxiv.org/abs/2206.13657.pdf  
https://lepora.com/digitac

Data and models use the (x, y, z, rz) components of pose, suitable for 4DoF robot arms. Here a Dobot MG400 is used.

## Installation

To install the package on Windows or Linux, clone the repository and run the setup script from the repository root directory:

```sh
pip install -e .
```

Code was developed and run from Visual Studio Code in Windows but has Unix compatibility.

## Examples

Some examples that demonstrate how to use the library are included in the `\examples` directory.  

E.g. to test the Dobot MG400 robot
```sh
python mg400_robot_test.py
```

## Requirements

Needs installation of these packages

Common Robot Interface (CRI) fork for use with Dobot Desktop Robots
https://github.com/nlepora/cri

Video Stream Processor (VSP) fork at
https://github.com/nlepora/vsp 

Also needs Tensorflow 2; tested with  
tensorflow==2.30  
tensorflow-gpu==2.30

Some functionality also needs hyperopt; tested with
hyperopt==0.2.5

## Workflow 

# Pose Models 2D

0. Check the installation with \examples
1. Collect the 2d edge or 2d surface tactile data with scripts in \collect
2. Process the data into training and test sets 
3. Train the pose models with scripts in \train: (a) single model; (b) optimize model hyperparameters
4. Test the pose models with scripts in \test: (a) single model; (b) batch of models from optimizer

Note: all code requires an environment variable DATAPATH to the data directory.

# Tactile Servoing 2D

Assumes you have already used Pose Models 2D to construct the models or downloaded a repository of data and models.  
0. Optional: check the installation with \examples

New experiments:  
1. Run servo_edge2d or servo_surface2d. The poses will be saved in predictions.csv. The data will be saved in frames_bw  

Existing data:  
2. You can replay previously collected trajectories either on the physical robot (MG400 controller) or virtually (dummy controller)  
3. Optionally you can visualize the contour and tactile images during replay (saved as contour.mp4 or frames.mp4)  
4. Optionally you can replay to use a camera to video the experiment. 

Note: all code requires an environment variable DATAPATH to the data directory.

## Repository structure 

\pose_models_2d  
&ensp;\collect  
&ensp;&ensp; collect_edge2d.ipynb - Notebook to collect data on 2d edge stimulus (Workflow step 1)  
&ensp;&ensp; collect_surface2d.ipynb - Notebook to collect data on 2d surface stimulus (alternative Workflow step 1)  
&ensp;&ensp; process.ipynb - Notebook to partition into training and test data sets (Workflow step 2)  
&ensp;\examples    
&ensp;&ensp; frames_robot_test.py - Script to check sensor is collecting frames of data  
&ensp;&ensp; mg400_robot_test.py - Script to check MG400 movement commands are working  
&ensp;\lib  
&ensp;&ensp; \models  
&ensp;&ensp; &ensp; cnn_model.py - Class for training/testing/predicting using a ConvNet on tactile images  
&ensp;&ensp; &ensp; dummy_model.py - Class for test purposes  
&ensp;&ensp; common.py - common functions used in repository   
&ensp;\test  
&ensp;&ensp; test2d_cnn.ipynb - Notebook to test a trained model (Workflow step 4)  
&ensp;&ensp; test2d_cnn_batch.py - Script to test a batch of trained models (alternative Workflow step 4)  
&ensp;\train  
&ensp;&ensp; train2d_cnn.ipynb - Notebook to train a model (Workflow step 3)  
&ensp;&ensp; train2d_cnn_opt.py - Script to optimize hyperparameters over trained models (alternative Workflow step 3)

\tactile_servoing_2D  
&ensp;\examples    
&ensp;&ensp; cnn_robot_test - Script to check data-to-model throughflow is working  
&ensp;\lib  
&ensp;&ensp; \visualise  
&ensp;&ensp; &ensp; plot_contour2d - Class to plot in real-time the servo control trajectory  
&ensp;&ensp; &ensp; plot_frames2d - Class to plot in real-time the captured tactile images   
&ensp;&ensp; common_camera.py - Library of functions (home, control) customized to video the robot   
&ensp;&ensp; common.py - Library of functions (home, control) for the robot  
\servo  
&ensp;&ensp; replay_camera.py - Replay a saved servo control run while videoing the robot (Workflow step 4)   
&ensp;&ensp; replay.py - Replay a saved servo control run (Workflow steps 2 & 3)  
&ensp;&ensp; servo_edge2d.ipynb - Notebook to collect data on 2d edge stimulus (Workflow step 1)  
&ensp;&ensp; servo_surface2d.ipynb - Notebook to collect data on 2d surface stimulus (alternative Workflow step 1)    
\videos  
&ensp;&ensp; batch_run.py - Script to run multiple offline result scripts  
&ensp;&ensp; make_video.py - Script to prepare a results video for a servo control run

## Precollected data, models and servoing data

Data, models and servo control runs for use with this code are available in the data repository
tactile-servoing-2d-dobot
https://doi.org/10.5523/bris.110f0tkyy28pa2joru2pxxbrxd

There are 2 types of tactile servo control:  
- servo_edge2d: around a horizontal edge of a planar object   
- servo_surface2d: around a vertical wall of a planar object

Available for 4 sensors:  
- digit: DIGIT sensor with GelSight elastomeric skin  
- digitac: TacTip version of the DIGIT sensor  
- tactip-127: Hemispherical TacTip (127 pin version; 40mm dia.)  
- tactip-331: Hemispherical TacTip (331 pin version; 40mm dia.)  

Note: servo_surface2d only available for TacTip-type sensors.

## Papers

The methods and data are described and used in this paper

DigiTac: A DIGIT-TacTip Hybrid Tactile Sensor for Comparing Low-Cost High-Resolution Robot Touch  
N Lepora, Y Lin, B Money-Coomes, J Lloyd (2022) IEEE Robotics & Automation Letters  
https://arxiv.org/abs/2206.13657.pdf  
https://lepora.com/digitac

Related papers 

Optimal Deep Learning for Robot Touch: Training Accurate Pose Models of 3d Surfaces and Edges  
N Lepora, J Lloyd (2020) IEEE Robotics & Automation Magazine  
https://arxiv.org/pdf/2003.01916.pdf  
https://lepora.com/optimal_touch

Pose-Based Tactile Servoing: Controlled Soft Touch with Deep Learning  
N Lepora, J Lloyd (2021) IEEE Robotics & Automation Magazine  
https://arxiv.org/pdf/2012.02504.pdf  
https://lepora.com/pose-based_tactile_servoing

From Pixels to Percepts: Highly Robust Edge Perception and Contour Following using Deep Learning and an Optical Biomimetic Tactile Sensor  
N Lepora et al (2019) IEEE Robotics & Automation Letters  
https://arxiv.org/pdf/1812.02941.pdf

## Meta

pbts-2d:

Nathan Lepora – n.lepora@bristol.ac.uk

[https://github.com/nlepora/pbts-2d](https://github.com/nlepora/pbts-2d)

Distributed under the GPL v3 license. See ``LICENSE`` for more information.
