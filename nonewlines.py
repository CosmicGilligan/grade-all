# Specify the path to your input file
input_file_path = './text/Transcripts/rvwquery.prmt'
# Specify the path to your output file
output_file_path = './text/Transcripts/rvwnnl.prmt'

# Read the contents of the input file
with open(input_file_path, 'r') as file:
    content = file.read()

# Replace all newline characters with spaces
modified_content = content.replace('\n', ' ')

# Write the modified content to the output file
with open(output_file_path, 'w') as file:
    file.write(modified_content)

print("Newline characters have been replaced with spaces.")