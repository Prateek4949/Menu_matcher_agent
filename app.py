import gradio as gr
from agent_menu_matcher_old import MenuMatcherAgent
from agent_menu_matcher import MetaLlamaLLM, MealPlanStrategistAgent

def recommend_meals(extracted_items, lat, lng):
    try:
        matcher = MenuMatcherAgent(lat=float(lat), lng=float(lng), extracted_items=[extracted_items])
        matches = matcher.run()
        strategist = MealPlanStrategistAgent(llm=MetaLlamaLLM())
        top_meals = strategist.run(matches)

        if not top_meals:
            return "❌ No meal plans found."

        result_text = ""
        for idx, meal in enumerate(top_meals, 1):
            result_text += f"{idx}. {meal['matched_item']} @ {meal['restaurant']} ({meal['location']}) — ₹{meal['price']}\n"
            result_text += f"   Ingredients: {meal['final_ingredients']}\n\n"
        return result_text.strip()

    except Exception as e:
        return f"⚠️ Error: {e}"

iface = gr.Interface(
    fn=recommend_meals,
    inputs=[
        gr.Textbox(label="Dish keyword (e.g. wrap)"),
        gr.Textbox(label="Latitude"),
        gr.Textbox(label="Longitude")
    ],
    outputs="text",
    title="Meal Recommendation Agent"
)

if __name__ == "__main__":
    iface.launch()
