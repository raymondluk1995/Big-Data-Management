# 20T2 COMP9313 Project 1
# Student Name: Raymond Lu
# Student Number: z5277884

########## Question 1 ##########
# do not change the heading of the function

# Collision counting LSH
def c2lsh(data_hashes, query_hashes, alpha_m, beta_n):

  #------------- Inner Function Definition Begins------------------# 
  # Get the absolute difference between the ith element between data hash and query_hashes
  def get_diff(data_hash,query_hashes):
    return ([abs(data_hash[i]-query_hashes[i]) for i in range(len(query_hashes))])

  # Check whether there are sufficient collisions for the given offset
  def check_col(diff_list,alpha_m,offset):
    count = 0 
    for i in range(len(diff_list)):
      if (diff_list[i]<= offset):
        count +=1 
    return (alpha_m<=count)

  # For a new offset, update the data hash in case of duplicate matching
  def update(x):
    if(x[2]==True):
      return (x) 
    else:
      return((x[0],x[1],check_col(x[1],alpha_m,offset))) 
  #------------- Inner Function Definition Ends------------------# 


  if (beta_n>data_hashes.count()):
    raise ValueError("Beta_n cannot be greater than the length of data_hahses!")

  if (alpha_m > len(query_hashes)):
    raise ValueError("Alpha_m cannot be greater than the length of query_hashes!")

  offset = 0
  can_count = 0 # candidate count
  # Add a flag showing whether a data hash collides with the query hash 
  data_hashes = data_hashes.map(lambda x:(x[0],get_diff(x[1],query_hashes),False)) 
  while True:
    data_hashes = data_hashes.map(update) 
    candidate = data_hashes.flatMap(lambda x: [x[0]] if x[2] else [])    
    can_count = candidate.count() 
    print("Now offset is ",offset,"And candidates are: ",candidate.collect())
    # offset += 1
    if (can_count>=beta_n):
     break 
    offset +=1
  return (candidate)
