import cairosvg
import numpy as np
import arckit
import arckit.vis as vis
import os
import csv

def string_to_numpy_matrix(matrix_str):
    matrix_str = matrix_str[1:-1]
    matrix_arr = []
    for char in matrix_str:
      if char == '[':
        row = []
      elif char == ']':
        matrix_arr.append(row)
      else:
        row.append((int)(char))
    try:
        return np.array(matrix_arr)
    except ValueError:
        return -1



def read_from_csv_skip_header(filename):
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        
        # Skip the header
        next(reader)
        
        # Read the first data row
        row = next(reader)
        
    return row

# filename = 'samples_csv/icl_overfit_source_0_thinking/ckpt_35000.csv'
filename = 'samples_csv/icl_overfit_source_0_long_thinking/ckpt_15000.csv'
row = read_from_csv_skip_header(filename)
prompt = row[0]
test_input = prompt[prompt.rfind('i') + 1:]
test_input = string_to_numpy_matrix(test_input)
test_output = row[1]
test_output = string_to_numpy_matrix(test_output)
samples = row[2:]

output_dir = 'images/' + filename[12:-4]
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
grid = vis.draw_grid(test_input, xmax=30, ymax=30, padding=.5, label='Test input')
vis.output_drawing(grid, output_dir+"/test_input.png") # svg/pdf/png

grid = vis.draw_grid(test_output, xmax=30, ymax=30, padding=.5, label='Test output')
vis.output_drawing(grid, output_dir+"/test_output.png") # svg/pdf/png

for (i, s) in enumerate(samples):
    sample = string_to_numpy_matrix(s)
    if np.any(sample == -1): 
       continue
    grid = vis.draw_grid(sample, xmax=30, ymax=30, padding=.5, label='Sample_' + str(i))
    vis.output_drawing(grid, output_dir+ '/sample_' + str(i) + '.png') # svg/pdf/png
   