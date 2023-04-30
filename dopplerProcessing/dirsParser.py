

import argparse
import os

def parse_arguments_directories(json=False):

	parser = argparse.ArgumentParser(description='Input and output directories')
	parser.add_argument('-i', '--input', dest = "input_dir", 
						required = True,
	                    help='an input directory')
	parser.add_argument('-o','--output', dest = "output_dir", 
						required = True,
	                    help='an output directory')
	parser.add_argument('-j','--json', dest = "json", 
					required = False,
                    help='ajson file with segmentation information')

	try:
		args = parser.parse_args()
	except:
		print("Arguments are not valid")
		parser.print_help()

	if not os.path.exists(args.input_dir):
		print("\n\nThe input path does NOT exist. Please enter a valid input directory.\n")

	if json and  args.json is None:
		print("A segmentation file is required. Please input it with the parameter -j.\n")
	
	return args.input_dir, args.output_dir, args.json

if __name__ == "__main__": 
	input_dir, output_dir, json_file = parse_arguments_directories()
	print("Input directory:", input_dir)
	print("Output directory:", output_dir)
	print("Json File:", json_file)

	print("\n")

