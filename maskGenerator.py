import numpy as np
import time
import matplotlib.pyplot as plt
import math
import pdb
import random

def get_mask_pseudo_diagonal_numpy(mask_shape, sparsity,file_name=None,experimentType="randomWithZero", 
                                   layerNum = None, numDiag = None, diag_pos = None, currLayer=1, debug=0):
  """Creates a psuedo-diagonal mask with the specified sparsity.
  Args:
    mask_shape: list, used to obtain shape of the random mask.
    sparsity: float, between 0 and 1.
  Returns:
    numpy.ndarray
  """
  # Create an array of zeros with the specified shape
  mask = np.zeros(mask_shape)
  print("DONT LOOK AT THIS FOR varyDIAG EXPERIMENT")
  print("Sparsity is ", sparsity)
  if(sparsity != float(0)):
    elemBudget = (1 - sparsity)*mask_shape[0]*mask_shape[1]
  else:
    elemBudget = float(0)

  # Calculate the length of the diagonals
  #diag_length = min(mask_shape[0], mask_shape[1])
  diag_length = max(mask_shape[0], mask_shape[1])
  totalDiag = math.floor(float(elemBudget)/float(diag_length))

  print("Element budget is ", elemBudget)
  print("Total Diag count is ", totalDiag)
  print("Shape is ",mask_shape)

  #TODO: Change it to depend on the sparsity
  #r = 6
  r = []

  np.random.seed(int(time.time()))

  # Determine custom sequence of starting positions
  """
  Types of diagonals to generate:
  1) Random position with a necessary diagonal at zeroth row (randomWithZero).
  2) Random position without a necessary diagonal at zeroth row (randomWithoutZero).
  3) Diagonal variation in one layer: Pick a layer, have one main diagonal. Then add one diagonal
     and vary position. Add another, sample some positions, add another and so on (oneDiagVar).
  """
  start_positions = []
  used_rows = set()
  start_row = 0
  start_col = 0

  # Set the parameters
  input_size = 28*28  # images are 28x28 pixels
  hidden_size = 128  # number of neurons in each hidden layer
  output_size = 10   # 10 classes


  if(experimentType == "randomWithZero"):
    print("Choosing diagonal position randomly with zeroth row.")
    used_rows.add(0)
    start_positions.append((0,0))  
    for i in range(totalDiag-1):
      start_row = np.random.choice([row for row in range(mask_shape[0]) if row not in used_rows])
      used_rows.add(start_row)
      start_positions.append((start_row,start_col))
  elif(experimentType == "randomWithoutZero"):
    print("Choosing diagonal position randomly without zeroth row.")
    for i in range(totalDiag):
      start_row = np.random.choice([row for row in range(mask_shape[0]) if row not in used_rows])
      used_rows.add(start_row)
      start_positions.append((start_row,start_col))
  elif(experimentType == "oneDiagVarWithDense"):
    print("Varying the number of diagonals and their positions.")
    totalDiag = int(numDiag)+1
    if int(layerNum) == 1:
      mask_shape = [hidden_size,input_size]
    elif int(layerNum) == 2:
      mask_shape = [hidden_size,hidden_size]
    else:
      mask_shape = [output_size,hidden_size]
    
    used_rows.add(0)
    start_positions.append((0,0))      
    #pdb.set_trace()
    if int(numDiag) == 1:
      start_row = int(diag_pos[0])
      start_positions.append((start_row,start_col))
    '''for i in range(totalDiag-1):
      start_row = np.random.choice([row for row in range(mask_shape[0]) if row not in used_rows])
      used_rows.add(start_row)'''
  else:
    print("Varying diagonal position but with sparse remaining layers")
    #pdb.set_trace()
    if currLayer == int(layerNum):
      used_rows.add(0)
      #start_positions.append((0,0)) 
      np.random.seed(int(time.time()))
      for i in range(len(diag_pos)):
        start_row = int(diag_pos[i])
        start_positions.append((start_row,start_col))
    elif currLayer == 1:
      #Generate starting diagonal positions including (0,0)
      start_positions.append((0,0))
      random.seed(9104)
      for i in range(totalDiag-1):
        start_row = random.randint(1,99)
        start_positions.append((start_row,0))
    else:
      #Generate 9 starting diagonal positions including (0,0)
      start_positions.append((0,0))

  #Check for the dimension of elements in start_positions
  r = [start_positions[i][0] for i in range(len(start_positions))]
  print(r)
  #pdb.set_trace()

  for start_row in r:
    current_row, current_col = (start_row)% mask_shape[0], 0
    for _ in range(diag_length):
      mask[current_row % mask_shape[0], current_col % mask_shape[1]] = 1
      current_row += 1
      current_col += 1

  if(debug == 0):
    print("Saving to file")
    with open('/p/dataset/abhishek/diag_pos_'+str(experimentType)+"_"+str(layerNum)+"_"+str(numDiag)+'.txt', 'a') as f:
      f.write(str(r))
      f.write("\n")

  return mask

