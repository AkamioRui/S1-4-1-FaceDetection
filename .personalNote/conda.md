# conda
- a package & environment manager. it download a package, and edit the environment variable.
- *activating an env* sets env var & run activation script

#
- conda init bash
- conda init --reverse --all

- conda activate
- conda deactivate

- conda create (remove using conda env remove [-n]/-p)

# interaction
- consider an env as a folder. it has
    - *python.exe*
    - */Lib/site-package* where python will search for dependency, Even if conda is not on 