"""
The setup.py file is an essential part of packaging and 
sistributing Python projects. It is use dby setuptools
(or distutils in plder Python Versions) to define the configuration
of your projects, such as its metadata, dependencies, and more.
"""

from setuptools import find_packages, setup
from typing import List

def get_requirements() -> List[str]:
    '''
    This function will return list of requirements
    '''
    requirement_list:List[str] = []
    try:
        with open('requirements.txt', 'r') as file:
            ## read lines from the file
            lines = file.readlines()

            ##Process each line
            for line in lines:
                requirement = line.strip()

                ## Ignore empty line and -e.
                if requirement and requirement != '-e .':
                    requirement_list.append(requirement)
    
    except FileNotFoundError:
        print("Requirements.txt file not found.")

    return requirement_list


setup(
    name = "Network Security",
    version= '0.0.1',
    author="Om Rajput",
    author_email="forcoding247@gmail.com",
    packages= find_packages(),
    install_requires = get_requirements()
)