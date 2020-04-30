# Utility-Tree-Based Analysis

## 1. Directory Structure

I only list the main files for the simulated data generation.

**test.py**: Test for the multi-agent interactor.

**MultiAgentInteractor.py:** The class for the multi-agent interactor.

**extracted_data/adjacent_map.csv:** A graph encoding adjacency in the game map. It should be re-computed for the new map.

**extracted_data/test_data.pkl:** A small proportion of data for tests.

## 2. Basic Idea
1. Construct an instance of class ``MultiAgentInteractor`` in the very beginning.

2. For every time step, pass the game status to agents with ``resetStatus()`` and call the function ``estimateDir()`` to estimate the moving direction.  

3. Save all the required game status at every time steps. The status should be stored may refer to the previous data file. 

## 3. Usage

The usage may refer to the file ``test.py``.


## 4. Dependencies

### Required (must be installed for running):

Interpretor: Python 3.x

Packages: pandas, numpy, anytree

### Optional (required for the test code but not mandatory for running the code)

Packages: pickle


## 5. TODO:

1. The design of agents should be re-considered. Especially for the global agent and local agent.

2. How to choose the path for global/local agents if there are multiple best paths.

3. A probabilistic agent.