# CGP_Neuron_Masters

Code for masters thesis - Sara Rambjør

Defines an abstract model of a neuron, and conducts evolutionary search using Carthesian Genetic Programming to learn functions, in the vein of Developmental Neural Networks and Artificial Life, and work by Julian F. Miller. 

## CODEBASE OVERVIEW
/cgp_visualization/: Folder for code for visualizing CGP functions. Does not support modular CGP functions, does support hex variants
- nodevis.html: HTML file
- nodevis.js: Javascript script for controlling visualization 
- Other files are supporting files. 
/neuron_3d_graph_visualization/: Visualizes phenotypes
- all_direct_connections.py: Script that checks if for every phenotype in a logfile every neuron that is connected to an input is also connected to an output
- otherwise same as above
cgp_program_executor.py: Script which implements a VM which runs CGP programs. A CGP program graph is compiled into an instruction set, and then for each call it is exectured by this script. This script might be a good target for lower level optimization. 
CGPEngine.py: Implements CGP, mutation and execution and compilation into cgp_program_executor scripts. 
*.slurm: Files used for executing code on NTNU's IDUN computational cluster
*.ini: Various config files for NMS-LOC
config_file_calculator.py: Script used in masters to compute the search space size of config files
runfile.py: Convenience entry point for calling rest of library 
sanity_check_one_pole.py: Script for analyzing random policy performance on one pole balancing problem. 
stupid_problem_test.py: Test problem, originally used in the preliminary thesis. Simple classification problem. Outdated
yaml_file_fixer.py: Script for fixing some errors in yaml statistics files
engine.py: Implements the Neuron Engine, used to evaluate genotypes by producing phenotypes. Interacts with input problem and input genotype to return a fitness score. 
genotype.py: Implements the genotypes, i.e. hex variants and chromosomal mutation and crossover and control code
HelperClasses.py: Various helper functions used in other places in the code. 
one_pole_problem.py: Implementation of one-pole balancing by providing an interface to OpenAI Gym implementation. Is now slightly ouitdated as code is rewritten in some places to also produce a fitness on the validation data for supervised learning problems, but should be easy to revert/fix. 
iris_problem.py: The IRIS flower classification problem. Must be supplied with loaded iris dataset. 
Logger.py: Contains code for logging. 
MainController.py: Contains code for evolution, and control logic for other parts of code (i.e. main logic loop)

