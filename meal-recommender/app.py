import gradio as gr
from agent_menu_matcher_old import MenuMatcherAgent
from agent_menu_matcher import MetaLlamaLLM, MealPlanStrategistAgent
import os

# Load API key from Hugging Face secret
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def recommend_meals(query):
    # Example fixed lat/lng for now
    lat, lng = 17.4110382, 78.3725869
    matcher = MenuMatcherAgent(lat=lat, lng=lng, extracted_items=[query])
    matches = matcher.run()
    llm = MetaLlamaLLM()
    strategist = MealPlanStrategistAgent(llm=llm)
    top_meals = strategist.run(matches)
    return top_meals

iface = gr.Interface(
    fn=recommend_meals,
    inputs=gr.Textbox(label="Enter food name"),
    outputs="json",
    title="Meal Recommender",
    description="Enter a food item and get top healthy meal suggestions."
)

if __name__ == "__main__":
    iface.launch()
