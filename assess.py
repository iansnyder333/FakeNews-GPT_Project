import os
import streamlit as st
import google.generativeai as palm

st.markdown(
    "<style>body { color: green; background-color: #f0f0f0; }</style>",
    unsafe_allow_html=True,
)

api_key = "Redacted-o"
palm.configure(api_key=api_key)

defaults = {
    "model": "models/chat-bison-001",
    "temperature": 0.25,
    "candidate_count": 1,
    "top_k": 40,
    "top_p": 0.95,
}


def get_recommendation(
    recommendation_type,
    formality_level,
    first_name,
    last_name,
    company,
    title,
    years_worked,
    skills_emphasis,
    improvement_emphasis,
    relationship,
    user_first_name,
    user_last_name,
    user_company,
    user_title,
    manual_notes,
    pronoun=None,
):
    skills = {
        "Communication Skills": 0,
        "Leadership": 0,
        "Problem-solving": 0,
        "Adaptability/Flexibility": 0,
        "Teamwork/Collaboration": 0,
        "Time Management": 0,
        "Creativity/Innovation": 0,
        "Critical Thinking": 0,
        "Emotional Intelligence": 0,
        "Resilience": 0,
        "Self-Motivation": 0,
        "Attention to Detail": 0,
        "Analytical Thinking": 0,
        "Integrity": 0,
        "Strong work ethic": 0,
        "Dependability": 0,
        "Initiative": 0,
        "Team player": 0,
        "Excellent communication skills": 0,
        "Adaptability": 0,
        "Problem-solving abilities": 0,
        "Critical thinking": 0,
        "Creative thinking": 0,
        "Analytical skills": 0,
        "Attention to detail": 0,
        "Time management": 0,
        "Organizational skills": 0,
        "Self-motivation": 0,
        "Resilience": 0,
        "Flexibility": 0,
        "Integrity": 0,
        "Professionalism": 0,
        "Positive attitude": 0,
        "Enthusiasm": 0,
        "Quick learner": 0,
        "Self-confidence": 0,
        "Empathy": 0,
        "Emotional intelligence": 0,
        "Open-mindedness": 0,
        "Cultural sensitivity": 0,
        "Interpersonal skills": 0,
        "Trustworthiness": 0,
        "Honesty": 0,
        "Reliability": 0,
        "Goal-oriented": 0,
        "Results-driven": 0,
        "Resourcefulness": 0,
        "Willingness to take on challenges": 0,
        "Ability to handle pressure": 0,
        "Conflict resolution skills": 0,
        "Networking abilities": 0,
        "Decision-making skills": 0,
        "Ability to give and receive constructive feedback": 0,
        "Self-awareness": 0,
        "Initiative": 0,
        "Entrepreneurial mindset": 0,
        "Public speaking skills": 0,
        "Problem-solving skills": 0,
        "Professional development": 0,
        "Strong academic background": 0,
        "Research skills": 0,
        "Attention to ethical considerations": 0,
        "Commitment to continuous learning and growth": 0,
    }

    for skill in skills_emphasis:
        skills[skill] = 1
    for skill in improvement_emphasis:
        skills[skill] = -1
    skills_text = "\n\n".join(
        [
            f"- {skill}: {('positive emphasis' if emphasis == 1 else 'potential area for improvement')}"
            for skill, emphasis in skills.items()
            if emphasis != 0
        ]
    )

    user_information = (
        f"Your information: {user_first_name} {user_last_name}, {user_title} at {user_company}\n"
        if user_first_name and user_last_name and user_company and user_title
        else ""
    )
    context = user_information
    tone = {
        "Informal": "casual and friendly ",
        "Neutral": "balanced and professional",
        "Formal": "formal and highly professional",
    }
    if recommendation_type == "Letter of Recommendation":
        context += f"Write a {tone[formality_level]} Letter of Recommendation without a header and signature for a job addressed to the hiring manager. The recommendation is for a candidate named {first_name} {last_name}, who is a {title} at {company} and has worked together for {years_worked} years in the capacity of {relationship}. Emphasize the following skills positivley, or as a potential area of improvement according to their label:\n\n{skills_text}"
    else:
        context += f"Write a {tone[formality_level]} testimonial for LinkedIn that is general for any roles in this persons field, for a candidate named {first_name} {last_name}, who is a {title} at {company} and has worked together for {years_worked} years in the capacity of {relationship}.  Emphasize the following skills positivley, or as a potential area of improvement according to their label:\n\n{skills_text} and make recommendation person specific not for a specific company"

    context += f"\n\n{manual_notes}" if manual_notes else ""

    if pronoun:
        context = context.replace("candidate", f"candidate ({pronoun})")

    messages = [context]
    response = palm.chat(**defaults, context=context, examples=[], messages=messages)

    return response.last


st.title("Recommendation Generator")
st.markdown(
    "Choose the qualities you want to emphasize and the type of recommendation you want. For example, if you want to emphasize Communication Skills and Leadership, make sure to check those boxes."
)

first_name = st.text_input("First Name")
last_name = st.text_input("Last Name")
company = st.text_input("Company")
title = st.text_input("Title")

relationship = st.selectbox(
    "Your Relationship with the Person",
    [
        "You managed directly",
        "You reported directly",
        "You were senior",
        "They were senior",
        "You worked with in the same group",
        "You worked in a different group",
        "You worked at a different company",
        "They were a client of yours",
        "You were a client",
        "You taught this person",
        "You studied together",
    ],
)

with st.expander("Advanced Optional Settings"):
    if st.checkbox("Add Pronoun"):
        pronoun = st.text_input("Pronoun")
    else:
        pronoun = None

    if st.checkbox("Add Your Own Information"):
        user_first_name = st.text_input("Your First Name")
        user_last_name = st.text_input("Your Last Name")
        user_company = st.text_input("Your Company")
        user_title = st.text_input("Your Title")
    else:
        user_first_name = user_last_name = user_company = user_title = None

    if st.checkbox("Add Manual Notes"):
        manual_notes = st.text_input("Manual Notes")
    else:
        manual_notes = None

years_worked = st.slider("Years Worked Together", min_value=0, max_value=20, value=0)

skills_emphasis = st.multiselect(
    "Select Skills to Emphasize",
    options=[
        "Communication Skills",
        "Leadership",
        "Problem-solving",
        "Adaptability/Flexibility",
        "Teamwork/Collaboration",
        "Time Management",
        "Creativity/Innovation",
        "Critical Thinking",
        "Emotional Intelligence",
        "Resilience",
        "Self-Motivation",
        "Attention to Detail",
        "Analytical Thinking",
        "Integrity",
        "Strong work ethic",
        "Dependability",
        "Initiative",
        "Team player",
        "Excellent communication skills",
        "Adaptability",
        "Problem-solving abilities",
        "Critical thinking",
        "Creative thinking",
        "Analytical skills",
        "Attention to detail",
        "Time management",
        "Organizational skills",
        "Self-motivation",
        "Resilience",
        "Flexibility",
        "Integrity",
        "Professionalism",
        "Positive attitude",
        "Enthusiasm",
        "Quick learner",
        "Self-confidence",
        "Empathy",
        "Emotional intelligence",
        "Open-mindedness",
        "Cultural sensitivity",
        "Interpersonal skills",
        "Trustworthiness",
        "Honesty",
        "Reliability",
        "Goal-oriented",
        "Results-driven",
        "Resourcefulness",
        "Willingness to take on challenges",
        "Ability to handle pressure",
        "Conflict resolution skills",
        "Networking abilities",
        "Decision-making skills",
        "Ability to give and receive constructive feedback",
        "Self-awareness",
        "Initiative",
        "Entrepreneurial mindset",
        "Public speaking skills",
        "Problem-solving skills",
        "Professional development",
        "Strong academic background",
        "Research skills",
        "Attention to ethical considerations",
        "Commitment to continuous learning and growth",
    ],
)

improvement_emphasis = st.multiselect(
    "Select Skills to Improve",
    options=[
        "Communication Skills",
        "Leadership",
        "Problem-solving",
        "Adaptability/Flexibility",
        "Teamwork/Collaboration",
        "Time Management",
        "Creativity/Innovation",
        "Critical Thinking",
        "Emotional Intelligence",
        "Resilience",
        "Self-Motivation",
        "Attention to Detail",
        "Analytical Thinking",
        "Integrity",
        "Strong work ethic",
        "Dependability",
        "Initiative",
        "Team player",
        "Excellent communication skills",
        "Adaptability",
        "Problem-solving abilities",
        "Critical thinking",
        "Creative thinking",
        "Analytical skills",
        "Attention to detail",
        "Time management",
        "Organizational skills",
        "Self-motivation",
        "Resilience",
        "Flexibility",
        "Integrity",
        "Professionalism",
        "Positive attitude",
        "Enthusiasm",
        "Quick learner",
        "Self-confidence",
        "Empathy",
        "Emotional intelligence",
        "Open-mindedness",
        "Cultural sensitivity",
        "Interpersonal skills",
        "Trustworthiness",
        "Honesty",
        "Reliability",
        "Goal-oriented",
        "Results-driven",
        "Resourcefulness",
        "Willingness to take on challenges",
        "Ability to handle pressure",
        "Conflict resolution skills",
        "Networking abilities",
        "Decision-making skills",
        "Ability to give and receive constructive feedback",
        "Self-awareness",
        "Initiative",
        "Entrepreneurial mindset",
        "Public speaking skills",
        "Problem-solving skills",
        "Professional development",
        "Strong academic background",
        "Research skills",
        "Attention to ethical considerations",
        "Commitment to continuous learning and growth",
    ],
)
recommendation_type = st.selectbox(
    "Recommendation Type", ["Letter of Recommendation", "LinkedIn Recommendation"]
)
formality_level = st.selectbox("Formality Level", ["Informal", "Neutral", "Formal"])
recommendation = ""

if st.button("Generate Recommendation"):
    with st.spinner("Generating recommendation..."):
        recommendation = get_recommendation(
            recommendation_type,
            formality_level,
            first_name,
            last_name,
            company,
            title,
            years_worked,
            skills_emphasis,
            improvement_emphasis,
            relationship,
            user_first_name,
            user_last_name,
            user_company,
            user_title,
            manual_notes,
            pronoun,
        )

if recommendation:
    st.text_area("Recommendation", recommendation, height=850, key="recommendation")
"""
 Facebook Plans to Crack Down on Vaccine MisinformationFacebook CEO Mark Zuckerberg is reportedly mulling an initiative that could make it easier for companies and health insurers not only by offering vaccines but also providing them more targeted information. The company’s proposed changes, which include an incentive program with funding from the Food Safety Modernization Act of 2010 (FSA) would have a major impact in terms how health information on parents should be protected, according reports. “The goal will likely change over time as research into how the system works has been completed so far,” Zuckerberg wrote in a blog post, citing current studies and new research. He said his"""
