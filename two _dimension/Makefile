VAR='execute vtk.py and clean the .npy'
VAR1='python execution'
VAR2='after vtk execution clean .npy'

execute:
	$(info Caches have been deleted)

middle:execute
	@rm -r __pycache__
cleaning : middle
	@rm -r __epyccel__

