# machine-learning-project-itu-fall-2021

Welcome to the Fall 2021 Machine Learning Project ([Github](https://github.com/jonas-mika/ml-project)). 
The goal of this project was to build a classifier to predict glass fragments for forensic science purposes in criminal investigations. 

## Problem Description

Glass is a material that is prominently part of criminal investigation processes.   When a suspect is apprehended for a crime involving shattered glass, it is a standard 
procedure to submit particles ofhis clothing to a forensic science laboratory, in order to determine 
whether or not evidentiary materialis present. However, even in the case where glass particles were 
detected, it often remains unclearwhether those particles are connected to the crime. On this basis, 
the goal of this project was to build a machine learning model, that is able to classify different  
types of glass fragments based on their elemental composition and refractive index (RI).

## Data
The data used within this project was obtained from a study carried out as part of a research program 
for the UK  Forensic  Science  Service.  The data set contained a total of 214 glass fragments that were 
obtained in a pre-split of 149 training and 65 test samples.  For each glass fragment, a total of nine features 
were recorded, including a measure of the refractive index (RI), which describes how fast light travels through 
a material.  It is a standard measure in glass analysis as refit varies significantly for different types of glass.  
The remaining eight measured features described the chemical composition of the glass fragment in percent. Each glass fragment in the data set belonged to one of six classes.

## Run this Project 

Reproduce the results in the few following steps: 

1. **Clone repository**
   Navigate to the desired location of the project on your local machine and run the following command
   to clone this GitHub repository:

   ```
   https://github.com/jonas-mika/ml-project.git
   ```

2. **Create `VENV`** *(optional, but recommended to get all dependencies)*
    First, navigate to the folder where you are storing your venvs and then use venv to create virtual env 
    for this project as follows: 
    
    ```
    python3 -m venv [name of venv]
    ```

    Now, you can activate the venv through the command: 

    ```
    source [name of env]/bin/activate
    ```
    
    Deactivate through:

    ```
    deactive
    ```


    Lastly, make sure your pip is updated: `pip install --upgrade pip`

3. **Install requirements**
   Run the following command from the root of the cloned directory to install all dependencies into the virtual venv.
    
   ```
   pip install -r requirements.txt
   ```

4. You are all set. Run main.py in the `src/` folder and run the entire project.

## Documentation 

The project is extensively documented. All classes and helper-function used in this project in the folder `scripts/` come with a docstring. The entire documentation is hosted online through [*sphinx*](https://www.sphinx-doc.org/en/master/) and is visitable [here](https://ml-project-itu.readthedocs.io/en/latest/index.html)

## Contributers

<table>
  <tr>
    <td align="center"><a href="https://github.com/LudekCizinsky"><img src="https://github.com/LudekCizinsky.png?size=100" width="100px;" alt=""/><br /><sub><b>Ludek Cizinsky</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/jonas-mika"><img src="https://github.com/jonas-mika.png?size=100" width="100px;" alt=""/><br /><sub><b>Jonas-Mika Senghaas</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/lukyrasocha"><img src="https://github.com/lukyrasocha.png?size=100" width="100px;" alt=""/><br /><sub><b>Lukas Rasocha</b></sub></a><br /></td>
  </tr>
</table>
