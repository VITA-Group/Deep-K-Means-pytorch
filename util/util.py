import numpy as np
import os

def lindexsplit(some_list, *args):
	# Checks to see if any extra arguments were passed. If so,
	# prepend the 0th index and append the final index of the
	# passed list. This saves from having to check for the beginning
	# and end of args in the for-loop. Also, increment each value in
	# args to get the desired behavior.
	if args:
		args = np.cumsum(args)
		args = (0,) + tuple(data for data in args)

	# For a little more brevity, here is the list comprehension of the following
	# statements:
	#    return [some_list[start:end] for start, end` in zip(args, args[1:])]
	my_list = []
	for start, end in zip(args, args[1:]):
		my_list.append(some_list[start:end])
	return my_list

def check_dir(dir):
	if not os.path.exists(dir):
		os.makedirs(dir)

