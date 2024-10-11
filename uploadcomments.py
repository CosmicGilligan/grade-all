import requests

# Your Canvas API URL and token
API_URL = 'https://sdccd.instructure.com/api/v1'
API_TOKEN = '1069~F4enUCTFHyakX62ytfNP3wfRvNRXvB23hax44xr7y7CCN89hrGzwEUa44c2t4czc'
course_id = '51228'  # Replace with the actual course ID

# Headers for the request
headers = {
    'Authorization': f'Bearer {API_TOKEN}'
}

# URL for listing all courses with additional parameters
#url = f'{API_URL}/courses/2465721/users'

# Function to post a comment on a submission
def post_comment(course_id, assignment_id, student_id, comment):
    url = f'{API_URL}/courses/{course_id}/assignments/{assignment_id}/submissions/{student_id}'
    payload = {
        'comment': {
            'text_comment': comment
        }
    }
    response = requests.put(url, headers=headers, json=payload)
    if response.status_code == 200:
        print(f'Successfully added comment for student ID {student_id}')
    else:
        print(f'/nFailed to add comment for student ID {student_id}: {response.content}')

# Example usage
course_id = '2465721'
assignment_id = '20732229'
student_id = '8688734'
comment = 'Great job on the assignment!'

post_comment(course_id, assignment_id, student_id, comment)