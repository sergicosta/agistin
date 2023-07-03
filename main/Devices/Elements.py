# -*- coding: utf-8 -*-
"""
AGISTIN project 

...\Devices\Elements.py

Class Elements contains general characteristics of all devices.
"""

class Elements():
    
    def __init__(self):
        self.id = []
        pass
    
    def add(self, id_elem):
        if id_elem in self.id:
            print("*** ERROR: ID "+ str(id_elem) + " already exists in " + str(self.id) + " ***")
            return -1
        self.id.append(id_elem)