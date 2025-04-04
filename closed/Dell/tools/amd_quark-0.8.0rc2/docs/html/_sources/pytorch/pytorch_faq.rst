Frequently Asked Questions (FAQ)
================================

.. note::  
  
    In this documentation, **AMD Quark** is sometimes referred to simply as **"Quark"** for ease of reference. When you  encounter the term "Quark" without the "AMD" prefix, it specifically refers to the AMD Quark quantizer unless otherwise stated. Please do not confuse it with other products or technologies that share the name "Quark."

AMD Quark for Pytorch
-----------------

Environment Issues
~~~~~~~~~~~~~~~~~~

**Issue 1**:

Windows CPU mode does not support fp16.

**Solution**:

Because of torch `issue <https://github.com/pytorch/pytorch/issues/52291>`__\ , Windows CPU mode cannot perfectly support fp16.

C++ Compilation Issues
~~~~~~~~~~~~~~~~~~~~~~

**Issue 1**:

Stuck in the compilation phase for a long time (over ten minutes), the terminal shows like:

.. code:: shell

   [QUARK-INFO]: Configuration checking start. 

   [QUARK-INFO]: C++ kernel build directory [cache folder path]/torch_extensions/py39...

**Solution**:

delete the cache folder ``[cache folder path]/torch_extensions`` and run AMD Quark again.

.. raw:: html

   <!-- 
   ## License
   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
