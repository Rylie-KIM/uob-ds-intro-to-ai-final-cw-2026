# uob-ds-intro-to-ai-final-cw-2026

# EMATM0067 Introduction to AI and Text Analytics Coursework
- Spring 2026. Taught by: Edwin Simpson, Conor Houghton, Karina Nurlybayeva
- Deadline: 13.00 on Tuesday 5th May
- Estimated effort: approximately up to 70 hours per student.
- Group size: five to six students.
- Weighting: group report = 80%, individual reflection = 20%.

# Overview
For this coursework, you will each be assigned to a team of five to six students. 
Task5 is designed to:
- Develop your ability to design and implement real-world pipelines for analysing text data
using AI methods
- Encourage hypothesis-driven exploration and careful evaluation over model complexity
- Provide an opportunity to work in a team to produce more powerful results.
Working as a team, you will implement an analysis pipeline, experiment with key design choices and
analyse different aspects of the data, to produce a group report describing your method, results and
insights. 
- You will also write a short individual reflection, documenting your personal contributions to
the project.
The tasks have been carefully selected to represent similar levels of difficulty. 


# Supervision
Your group will be assigned a supervisor (Professor. Cononr) who will meet with you at three different points. Their role
is not to provide technical advice on solving your task, but to advise on group working and help to
ensure all members of the team play a fair part.

### Each meeting will be 20 minutes and your supervisor will ask:
1. How has each person contributed since the last meeting?
2. What challenges are you facing and could you overcome them?
3. What objectives and responsibilities does each team member have for the next stage of
work?

### Supervision points: supervision will take place in the following weeks during the labs.
- Week 21 (before Easter vacation)
- Week 22 (after Easter vacation)
- Week 24 (one week before submission).


# Task 5: Build a Neural Network to Describe Simple Pictures.
Relating visual inputs to natural language descriptions is a core challenge in AI. This task involves
working with multimodal data to explore how simple neural networks can be adapted to generate image descriptions.

## Objective: The goal of this task is to see if a simple neural network can describe simple pictures; 
it is intended that the task use generated images so that there is an abundance of labelled data.
You will find sample code to help you get started on this task in the Text Analytics Github repository.

## Student group work:
1. Dataset generation – write a script to generate sentences based on some simple
relationships. 
Here are some examples:
- a. Arrangements of simple shapes, such as “a big blue sphere is above a small red
cube”; by reducing each part of the sentence to a simple symbol, it should be easy
to generate a large number of sentences. For example, define a list of objects {o_1,
o_2, . . ., o_n} and colours {c_1, c_2, . . ., c_m}, then use these to complete nm twoword colour-object phrases by selecting objects o_i and colours c_j to complete a
sentence template. Ideally, the sentences won’t all have the same structure.
- b. Make images of numbers of varying length and sizes with different colours – “large
blue 67”, “small red 1337” and so on.
- c. Make tic-tac-toe boards in different game states corresponding to different
sentences describing the game, e.g., “X has gone in the center, 0 in the top left and
bottom right, X in the middle top”.

2. Write a script to generate images from your symbol sequences. For 1a and 1c, you could use a Python graphics library to draw shapes or draw Xs and 0s on a grid. For 1b use, for
example, MNIST to draw different versions of each number.

3. Design and implement a neural network for mapping images to sentences:
- a. Text representation: convert the sentences to a suitable representation, e.g., use a
language model to obtain sentence embeddings.
- b. Map images to text: build a CNN (or similar) to predict the sentences from the
pictures.
- c. Compare your approach to a commercial LLM
- d. Choose a further analytic axis to explore, such as: what text representation or CNN
architecture variant is most effective?
- e. For your chosen analytic axis, compare 2-3 alternative approaches, motivated by a
clear hypothesis about how each approach will perform.


4. Evaluate each method in terms of:
- a. Sentence prediction accuracy – does your method work?
- b. Does a commercial LLM do better than your neural network?
- c. Error analysis, giving an understanding of performance variation with different
sentence or image structures


5. Discuss how each design choice affects performance and why.

# Group Report (80%)
Your team’s report should be structured into the headings described below. The descriptions below
are guidelines, rather than strict rules, and each section should include all the information that you
think is relevant.

1) Abstract (0.25 pages): a brief summary of your project, stating the main challenge, the ideas you
tested, and key findings from the experiments.
2) Introduction (1.5 pages): outline the task you have been assigned, its motivations, the available
data and requirements for a good solution. You may wish to include examples from the dataset
and/or some data exploration plots.
3) Methods (3.5 pages):
- a) Explain the baseline system, including important preprocessing steps and chosen text
representations or features. Include figures and/or equations to make your design clearer.
- b) Discuss the axes of experimentation, how they could affect performance.
- c) Describe your proposed improvements along each chosen axis. Include hypotheses about
why you think these improvements will help.

4) Experimental evaluation (3.5 pages):
- a) Explain your choice of performance metrics and their limitations.
- b) Describe your experimental setup: how each dataset split was used, how hyperparameters
were set, and any other important information about how your methods were run. It is not
necessary to list all the software packages you used, but reference any libraries used for the
core parts of your system (e.g., links to HuggingFace models).
- c) Present your results in suitable plots and/or tables, providing interpretation of the results to
highlight important observations about the comparative performance of your approaches.
- d) Analyse the errors that your methods make to understand if there are particular text
patterns that cause more frequent errors.
- e) Interpret and discuss your results, explaining the strengths and limitations of each method,
and insights or recommendations based on your analysis.

5) Conclusions (1 page): describe the key findings from your experiments, explaining what worked
well, what did not work, and the open challenges for this task.

- What we are looking for
Motivated design choices, supported by evidence. We are not looking for novel hypotheses, stateof-the-art leaderboard scores, or ground-breaking new research. A good report will justify design
decisions, with data exploration and experimental analysis to provide support.
A few insightful claims about the nature of the task, and what works well, presented as a coherent
story.
Depth, rather than breadth – a report that provides a few interesting insights, with sound
experimental results, rather than testing a large number of different models and configurations in a
superficial way.
Evidence of understanding the application itself – the requirements for a good system and how your
system compares.
Understanding of this particular task or dataset – how its peculiarities may affect the design of your
pipeline and the performance of AI and NLP methods.
10% of marks will be given for the overall clarity of the report and how well it communicates your
ideas, analysis and results.


# Group Report Formatting
• Maximum 10 pages of A4 for the group task
- Font must be at least 11pt, margins at least 2.5cm/1 inch, with single line spacing.
- Do not try to squeeze extra content in – you will lose marks. Conversely, there will be
higher marks for more concise writing.
- References do not count toward the page limit.
• Use the Overleaf template from either NeurIPS or ACL to format your report (or the
MSWord equivalent). This will automatically set margins and fonts.
• The text in your figures must be big enough to read without zooming in.
• Use citations where appropriate, following a typical citation and reference style, as in the NeurIPS or ACL guidelines above.

# Individual Reflection (20%)
The individual reflection is intended to help us understand your contributions to the group work, as
well as your own understanding of the assigned tasks and your proposed solution.
Your individual reflection should use the same formatting as the group report, and include the
following sections (available marks in brackets):
1. A brief summary of your contributions, a visualisation of your individual activity on Github
and your project management tool (max 1 page A4, max 150 words; 8 marks)
2. Technical understanding (max 300 words; 7 marks):
a. Explain one technical challenge you worked on personally.
b. How did your contribution affect the results of the group project?
3. Teamwork (max 200 words; 5 marks): How the team coordinated their work and ensured
that every member was included. Explain the roles of each team member who contributed
and mention any tools that helped you coordinate your teamwork.

Submission
- Each group must submit a report to the submission point named 
“Group Report – Turnitin Submission Point”. Make sure your report includes a link to a repository of your team’s code
(e.g., a public Github link or OneDrive link that is readable by members of the university).
- Each person must submit their own individual reflection to the submission point “Individual
Reflection – Turnitin Submission Point”
- Avoiding Academic Offences
Please re-read the university’s plagiarism rules to make sure you do not break any rules. Academic
offences include submission of work that is not your own, falsification of data/evidence or the use of
materials without appropriate referencing. Note that sharing your report with others is also not
allowed. These offences are all taken very seriously by the University.
Do not copy text directly from your sources – always rewrite in your own words and provide a
citation.

- Work independently – do not share your code or reports with others.
Do not use AI to generate parts of your report – this includes automatically translating passages of
text from another language.
Suspected offences will be dealt with in accordance with the University’s policies and procedures. If
an academic offence is suspected in your work, you will be asked to attend an interview, where you
will be given the opportunity to defend your work. The plagiarism panel can apply a range of
penalties, depending on the severity of the offence. 

These include a requirement to resubmit work,
capping of grades and the award of no mark for an element of assessment.
Extensions and Exceptional Circumstances
If the completion of your assignment has been significantly disrupted by serious health conditions,
personal problems, or other serious issues, you can apply for consideration in accordance with
university policies. Refer to the guidance and complete the application forms as soon as possible.


### Please see the guidance below and discuss with your personal tutor for more advice:
https://www.bristol.ac.uk/students/support/academic-advice/assessment-support/request-acoursework-extension/
https://www.bristol.ac.uk/students/support/academic-advice/assessment-support/exceptionalcircumstances/


# Tasks before lab  
1. Week21 (this week): 
- Do you understand the main goals of the task?
- How will you distribute tasks so that everyone in the team can contribute?
- What tools do you need to co-ordinate the groupwork (Github; task/project management tools such as Trello).


