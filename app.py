import streamlit as st

# --------------------- Section -0 : Heading & Title ------------------------------------- #

# App Title
st.set_page_config(page_title="LinkedIn Post Generator")
st.title("📢 LinkedIn Post Generator")
st.caption("Generate engaging LinkedIn posts using Gemini + LangChain + Streamlit")



# --------------------- Section -1 : API Provider Logic ----------------------------------- #


# 🎯 Sidebar Title
st.sidebar.title("🔐 LLM Provider Setup")

provider = st.sidebar.selectbox(
    "Choose LLM Provider",
    ["Default", "Gemini", "OpenAI", "Mistral"]
)

# 🧪 Test: Show which model being used
st.write(f"🧠 Selected LLM Provider: **{provider}**")



from myapikey import shiv_gemini_key

# ✅ Show proper label depending on provider
if provider == "Default":
    st.sidebar.markdown("✅ Using Default key built into app")
    api_key = shiv_gemini_key
elif provider == "Gemini":
    api_key = st.sidebar.text_input("🔑 Enter your Gemini API Key", type="password")
elif provider == "OpenAI":
    api_key = st.sidebar.text_input("🔑 Enter your OpenAI API Key", type="password")
elif provider == "Mistral":
    api_key = st.sidebar.text_input("🔑 Enter your Mistral API Key", type="password")
else:
    st.sidebar.error("❌ Invalid provider selected")
    st.stop()  # exit the app if something went wrong

if not api_key:
    st.sidebar.warning("⚠️ Please enter a valid API key to proceed.")
    st.stop()  # exit the app if no key is provided



# --------------------- Section -2 : LangChain LLM Setup ----------------------------------- #

from langchain_core.language_models import BaseLanguageModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI

import os
import traceback


# ✅ Function: Dynamically loads the correct LLM (with fallback to default Gemini Flash)
def load_llm(provider: str, api_key: str) -> BaseLanguageModel:
    """
    Returns an LLM instance based on selected provider and key.
    If key or model fails, falls back to default Gemini Flash.
    """

    if provider in ["Default", "Gemini"]:
        # Set Gemini key for LangChain to use
        os.environ["GOOGLE_API_KEY"] = api_key
        # Load Gemini Flash (faster + free in most cases)
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=1.0)
    elif provider == "OpenAI":
        os.environ["OPENAI_API_KEY"] = api_key
        # Load GPT-3.5-turbo (cheap + usually has free tier)
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=1.0)
    elif provider == "Mistral":
        os.environ["MISTRAL_API_KEY"] = api_key
        # Load Open Mistral 7B (free via their hosted API or HuggingFace hub)
        return ChatMistralAI(model="open-mistral-7b", temperature=1.0)

# 👇 Load the model 
llm = load_llm(provider, api_key)

def test_llm_connection():
    global llm  # 👈 So we can update it if needed

    try:
        test_prompt = "Just answer in 20-25 words about Langchain, without any extra text." 
        response = llm.invoke(test_prompt)
        st.write(f"{response.content}")
        st.success("✅ Successfully connected to LLM!")

    except Exception as e:
        st.warning("⚠️ Error loading selected LLM. Please check your API key or provider settings.")
        print(traceback.format_exc())
        st.stop()  # Stop execution if connection fails

# Run the test connection
if st.button("🔁 Test LLM Connection"):
    test_llm_connection()

st.markdown("---")



# --------------------- Section -3 : Post Generation fields ----------------------------------- #

st.subheader("📝 LinkedIn Post Generator")
st.markdown("Fill in the details below to generate your LinkedIn post:")
st.markdown("---")

# 1. 🔘 Post Type
post_type = st.selectbox("🔘 Post Type", ["Project", "Internship", "Achievement", "Workshop", "Event", "Collab"])

# 2. 🧠 Topic / Title
topic = st.text_input("🧠 Topic / Title", placeholder="e.g. Customer Churn Prediction with ANN")

# 3. 🛠 Tools / Tech Used
tool_options = list([
    "Python", "Flask", "Django", "Streamlit", "Render", "Netlify", "Vercel",
    "FastAPI", "Docker", "AWS", "GCP", "Azure", "Heroku", "Firebase",
    "TensorFlow", "Keras", "PyTorch", "Scikit-learn", "OpenCV",
    "HuggingFace", "LangChain", "Transformers", "spaCy",
    "Pandas", "NumPy", "Matplotlib", "Plotly", "PowerBI",
    "YOLO", "Git", "GitHub", "SQL", "MongoDB", "PostgreSQL",
    "ANN", "CNN", "RNN", "LSTM", "GRU", "Logistic Regression", "Decision Trees", "Random Forests",
    "SVM", "XGBoost", "LightGBM", "CatBoost", "Reinforcement Learning", "Deep Learning",
    "NLP", "Computer Vision", "Time Series Analysis", "Data Visualization",
    "RLHF", "Prompt Engineering", "Fine-tuning", "HTML", "CSS", "JavaScript",
    "React", "Vue.js", "Angular", "Bootstrap", "Tailwind CSS",
    "Others (Type Below)"
]) 

tools = st.multiselect("🛠 Tools / Tech Used", sorted(tool_options))

# Add "Others" if selected
other_tools = ""
if "Others (Type Below)" in tools:
    other_tools = st.text_input("✍️ Enter additional tools (comma separated)", placeholder="e.g., Gradio, ONNX")

# Final tools list
final_tools_list = [tool for tool in tools if tool != "Others (Type Below)"]
if other_tools.strip():
    final_tools_list.extend([tool.strip() for tool in other_tools.split(",") if tool.strip()])

# 4. 📍 Your Role
role = st.text_input("📍 Your Role", placeholder="e.g. Developed the full ML pipeline and deployed using Flask")

# 5. 🎯 Objective / Goal
objective = st.text_input("🎯 Objective / Goal", placeholder="e.g. To reduce churn rate using deep learning")

# 6. 💡 Key Features / Highlights
features = st.text_area("💡 Key Features / Highlights", placeholder="Use bullet points like:\n> 90% accuracy\n> Clean UI\n> Real-time prediction")

# 7. 📎 Links (optional)
links = st.text_input("📎 Links (optional)", placeholder="e.g. GitHub: github.com/xyz | Live: xyz.netlify.app")

# 8. 🗣️ Tone
tone = st.selectbox("🗣️ Tone", ["Formal", "Enthusiastic", "Humble", "Storytelling"])

# 9. 👥 Audience
audience = st.selectbox("👥 Audience", ["General public", "Recruiters", "Peers"])

# 10. 📸 Want hashtags?
want_hashtags = st.checkbox("📸 Want hashtags?", value=True)

# 11. 📝 Custom note at the end (optional)
include_custom_note = st.checkbox("📝 Include a custom note at the end?", value=True)

# Divider before preview
st.markdown("---")
st.subheader("🔍 Live Preview")

if topic:
    st.markdown(f"### 📌 {post_type} — {topic}")

if final_tools_list:
    st.markdown(f"**🛠 Tools:** {', '.join(final_tools_list)}")

if role:
    st.markdown(f"**📍 Role:** {role}")

if objective:
    st.markdown(f"**🎯 Objective:** {objective}")

if features:
    st.markdown("**💡 Key Highlights:**")
    for line in features.split("\n"):
        if line.strip():
            st.markdown(f"{line.strip()}")

if links:
    st.markdown(f"**📎 Links:** {links}")

st.markdown(f"**🗣️ Tone:** {tone} | **👥 Audience:** {audience}")

if want_hashtags:
    st.markdown("*✅ Hashtags will be included automatically.*")

if include_custom_note:
    st.markdown("*📝 A custom note will be added at the end.*")
st.markdown("---")
# ---------------- Disable button if essentials are missing ---------------- #

required_fields_filled = all([
    post_type, topic.strip(), final_tools_list, role.strip(), objective.strip(), features.strip()
])

if required_fields_filled:
    generate = st.button("🚀 Generate LinkedIn Post")
else:
    # create disabled button
    st.button("🚀 Generate LinkedIn Post", disabled=True)
    generate = False



# --------------------- Section -4 : Post Generation Logic ----------------------------------- #

from langchain.prompts import PromptTemplate

# Prompt template
prompt_template = PromptTemplate(
    input_variables=[
        "post_type", "topic", "tools", "role", "objective", "features",
        "tone", "audience", "links", "custom_note", "hashtag_instructions"
    ],
    template="""
You are an expert LinkedIn content creator with a deep understanding of tone, storytelling, and professional appeal.

Write a compelling, high-quality LinkedIn post in a {tone} tone. The post should feel authentic, engaging, and tailored for {audience}. Use natural language and real-world phrasing — avoid robotic or templated expressions.

Important Note: 
1. Do not include any introductory phrases like "Here's my post" or "I want to share". Start directly with the content.
2. Also, generate 2 versions of the post so that the user can choose the one they like best.
3. Each version should be started by a heading like "Version 1" or "Version 2" to differentiate them with line breaks in between.
4. Make sure to include a catchy main heading of the post at the start, which is based on the topic/title provided. Just For example, if the topic is "Customer Churn Prediction with ANN", the heading could be "🔍 Customer Churn Prediction: A Deep Dive into ANN Techniques".
5. Emojis can be used to enhance readability and engagement, but should not be overused. Use them where they naturally fit the content.

Here are the details:  

🔖 **Post Type**: {post_type}  
📌 **Topic/Title**: {topic}  
🧰 **Tools/Technologies Used**: {tools}  
👤 **My Role**: {role}  
🎯 **Goal / Objective**: {objective}  
✨ **Key Highlights / Features**:
{features}

🔗 **Additional Links (if any)**: {links}  
🗒️ **Custom Note / Personal Touch (if any)**: {custom_note}

End the post with a closing that resonates with the chosen tone ({tone}) and aligns with the purpose of the post. Make sure to inspire engagement from {audience} — whether it's likes, comments, or shares.

{hashtag_instructions}

Make the structure clean and visually digestible. Use line breaks, emojis (where helpful), and keep the post with a good length of around 300-400 words.

Your goal is to make it feel like a real, thoughtful post by a human, not an AI.

"""
)

# Prepare hashtag instructions based on checkbox
hashtag_instructions = (
    "Add relevant, popular LinkedIn hashtags (e.g. #MachineLearning #AI #ProjectShowcase) at the end."
    if want_hashtags
    else "Do not include any hashtags."
)

if generate:
    try:
        # Prepare the input data for the prompt
        input_data = {
            "post_type": post_type,
            "topic": topic,
            "tools": ", ".join(final_tools_list),
            "role": role,
            "objective": objective,
            "features": "\n".join([f"> {line.strip()}" for line in features.split("\n") if line.strip()]),
            "tone": tone,
            "audience": audience,
            "links": links if links.strip() else "Do not include any links.",
            "custom_note": "Include a custom ending note" if include_custom_note else "Do not include any custom note.",
            "hashtag_instructions": hashtag_instructions
        }

        from langchain.chains import LLMChain

        # Step 1: Create the chain with LLM and prompt
        post_chain = LLMChain(
            llm=llm,
            prompt=prompt_template
        )

        # Step 2: Generate the post using the chain
        response = post_chain.invoke(input_data)

        # Step 3: Display the result
        st.subheader("📝 Generated LinkedIn Post")
        st.markdown(response['text'])  # or response.get('text') safely
        st.markdown("---")
    
    except Exception as e:
        st.error(f"❌ Error generating post: {str(e)}")
        st.warning(f"Please check your inputs and try again. You can select default model to avoid issues.")
        

    # --------------------- End of app.py ----------------------------------- #