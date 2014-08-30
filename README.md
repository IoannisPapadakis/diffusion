##Knowledge diffusion effects of patents

*Nathan Goldschlag*

*August 26, 2014*

*Version 1.0*


The python files in this repository are used to gather and analyze textual data on both academic abstracts and patents. 

The order of execution is as follows:

1. diffusion_gen_data.py 
  * Scrapes abstract data from Nature Biotechnology and arXiv, finds top patents by citations and extracts patent abstracts
2. diffuion_run_sim.py
  * Calculates longest common subsequence similarity of each patent to each abstract for biotech, ai, and the test case ai paper
3. diffusion_analysis.py
  * Runs regression analysis on similarity measures and preps csv files for plotting
4. diffuion_plots.ipynnb
  * Generates descriptive statistics and plots


License: This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version. Included libraries are subject to their own licenses.
