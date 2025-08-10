import requests
import json, os
from typing import List, Dict
from agent_menu_matcher_old import MenuMatcherAgent

class MetaLlamaLLM:
    def __init__(self):
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        # self.headers = {
        #     "Authorization": "Bearer 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",
        #     "Content-Type": "application/json"
        # }
        self.headers = {
            "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
            "Content-Type": "application/json"
        }
        self.model = "meta-llama/llama-4-scout-17b-16e-instruct"

    def _call(self, prompt: str) -> str:
        """Call the Meta-Llama model via API and return the response."""
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}]
        }
        try:
            response = requests.post(self.url, headers=self.headers, json=data)
            response.raise_for_status()  # Raise an error for HTTP issues
            response_json = response.json()
            return response_json["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error calling Meta-Llama API: {e}")
            return '{"error": "LLM call failed"}'

    @property
    def _llm_type(self) -> str:
        return "meta-llama"

# Replace BedrockClaudeLLM with MetaLlamaLLM
custom_llm_wrapper = MetaLlamaLLM()

# --- Update MealPlanStrategistAgent to use MetaLlamaLLM ---
class MealPlanStrategistAgent:
    def __init__(self, llm: MetaLlamaLLM):
        self.llm = llm

    def run(self, matched_items: List[Dict]) -> List[Dict]:
        # Only send compact fields
        compact_items = [
            {
                "restaurant": item["restaurant"],
                "matched_item": item["matched_item"],
                "location": item["location"],
                "price": item["price"],
                "rating": item.get("rating", "N/A"),
                "description": item["ingredients"] or "N/A"
            }
            for item in matched_items
        ]

        prompt = f"""
You are a meal recommendation expert.

You are given a list of menu items, each with:
- restaurant
- matched_item
- location
- price
- rating
- description (or missing ingredients)

Your job:
- From the given dishes, select Top 10 best meal options.
- Prefer items from restaurants with HIGHER RATINGS.
- Prefer HEALTHIER meals (less fried, less sugary, more veggies/protein).
- For missing or vague descriptions, assume realistic ingredients intelligently.
- Keep ingredient description short (under 20 words).
- If two meals are similar, prefer the CHEAPER one.

⚡ Output Format:
***ONLY output a VALID JSON array list like***:
[
  {{
    "restaurant": "Restaurant Name",
    "matched_item": "Dish Name",
    "location": "Area Name",
    "price": 199.0,
    "final_ingredients": "brief healthy ingredients"
  }},
  ...
]

Do NOT write explanations. Do NOT write anything else. Only output clean JSON.

Here is the menu data:
{compact_items}
"""

        result = self.llm._call(prompt)
        try:
            return json.loads(result)
        except Exception as e:
            print(f"Failed to parse response: {e}\nRaw Output: {result}")
            return []

# --- Run the full flow ---
if __name__ == "__main__":
    # Sample input
    extracted_items = ["wrap"]
    lat = 17.4110382   # Manikonda
    lng = 78.3725869   # Manikonda

    # Step 1: Menu Matching
    matcher = MenuMatcherAgent(lat=lat, lng=lng, extracted_items=extracted_items)
    matches = matcher.run()

    # Step 2: Meal Planning
    custom_llm = MetaLlamaLLM()
    strategist = MealPlanStrategistAgent(llm=custom_llm)
    top_meals = strategist.run(matches)
    print("top_meals:", top_meals)
    
    if not top_meals:
        print("❌ No meal plans found.")
        
    # Step 3: Output
    print("\n🍽️ Top 10 Meal Plans:\n")
    for idx, meal in enumerate(top_meals, 1):
        print(f"{idx}. {meal['matched_item']} @ {meal['restaurant']} ({meal['location']}) — ₹{meal['price']}")
        print(f"   Ingredients: {meal['final_ingredients']}\n")