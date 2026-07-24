#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


# In[1]:


def rotation_matrix_2d(theta_deg):
    theta_rad = np.radians(theta_deg)
    return np.array([
        [np.cos(theta_rad), - np.sin(theta_rad)],
        [np.sin(theta_rad), np.cos(theta_rad)]
    ])


# In[3]:


def scale_matrix_2d(sx, sy):
    return np.array([
        [sx, 0],
        [0, sy]
    ])


# In[ ]:


def translate_2d(point, tx, ty):
    p = np.asarray(point, dtype=float)
    t = np.array([tx, ty], dtype=float)
    return p + t


# In[ ]:




