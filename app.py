import re

import gradio as gr
from transformers import pipeline, set_seed

generator = pipeline('text-generation', model='gpt2')
# generator = pipeline('text-generation', model='EleutherAI/gpt-j-2.7B')

# ideally we would use a larger model but it's either not free or too big for my macbook air lol
# generator = pipeline('text-generation', model='EleutherAI/gpt-j-6B')

# set_seed(42)


def produce_text(company, job_title, job_description)->str:

    prompt = """The following is a list of predictions about how AI will automate people out of jobs.

==========

Job title and description: Customer Service Representative, answering customer questions and providing support via phone, email, and chat.

How this job will be automated by AI: AI will be able to answer customer questions and provide support via phone, email, and chat. This will reduce if not eliminate the need for a human to perform this task.

==========

Job title and description: Data Scientist, analyzing data and building machine learning models.

How this job will be automated by AI: AI will be able to analyze data and build machine learning models. This will increase the productivity of data scientists, allowing them to perform more tasks in the same amount of time. This will result in a net reduction in demand for data scientists.

=========="""

    model_input = f"""
{prompt}

Job title and description: {job_title}{' at ' + company if company else ''}, 
{job_description}

How this job will be automated by AI:"""

    output = generator(model_input, max_length=512, num_return_sequences=1)



    result = output[0]['generated_text']
    print(result)
    final_result = result[len(model_input):]

    # split the string if either an equals sign or a dash is encountered using regex
    final_result = re.split('-|=', final_result)
    final_result = final_result[0]

    return final_result


iface = gr.Interface(
    fn=produce_text,
    inputs=[
        gr.inputs.Textbox(label="Company (Optional)"),
        gr.inputs.Textbox(label="Job Title"),
        gr.inputs.Textbox(label="Job description"),
    ],
    outputs=[
        gr.outputs.Textbox(label="How this job may be automated by AI"),
    ],
    title="Job Automation by AI", 
    description="A simple app to predict how certain jobs may be automated by AI, as predicted by an AI. (Note: this is using GPT-2, and would be substantially better with GPT-3 or GPT-J-6B, but those models are not free and are too big for my macbook air to run.)",
    examples=[
        ['Google', 'Backend Software Engineer', 'Programming and maintaining backend services for Google products. Writing code in Python, Go, and Java.'],
        ['Accenture', 'Senior Consultant', 'Working with clients to solve business problems.'],
        ['', 'Social Media Coordinator', 'Creating and posting content on social media. Managing social media accounts. Monitoring social media for mentions of the company.'],
        ['', 'Content Creator', 'Creating short and long form video content for social media'],
        ['Goldman Sachs', 'Junior Analyst', 'Analysing financial data, creating financial models to predict future market movements and trends. Writing reports for clients and senior management.'],
        ['Santa Clara University', 'Associate Professor', 'Teaching undergraduate and graduate level courses.']
    ]
)
iface.launch()
