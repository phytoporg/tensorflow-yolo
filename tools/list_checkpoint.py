import sys
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

checkpoint_path = sys.argv[1]

# List ALL tensors example output: v0/Adam (DT_FLOAT) [3,3,1,80]
print_tensors_in_checkpoint_file(file_name=checkpoint_path, tensor_name='', all_tensors=True)
