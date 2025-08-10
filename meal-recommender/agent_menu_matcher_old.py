from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from typing import List, Dict
import json
import os, boto3
import requests
import time
import pandas as pd
from langchain_core.language_models.llms import LLM

# --- Setup: AWS + Bedrock Claude 3.5 ---
region_name = 'us-west-2'
script_dir = os.path.dirname(os.path.abspath(__file__))


SYNONYM_MAP = {
    "curry": ["masala", "gravy"],
    "biryani": ["pulao", "rice"],
    "chicken": ["murgh", "tandoori"],
    "mutton": ["lamb", "gosht"],
    "paneer": ["cottage cheese"],
    "veg": ["vegetarian", "veggie"],
    "non-veg": ["non-vegetarian", "nonveg"],
    "samosa": ["snack"],
    
}

def expand_query(query):
    """Expand query with synonyms."""
    query = query.lower().strip()
    synonyms = [query]
    for key, values in SYNONYM_MAP.items():
        if key in query:
            synonyms.extend(values)
        for val in values:
            if val in query:
                synonyms.append(key)
    return list(set(synonyms))  # Unique terms

# class BedrockClaudeLLM(LLM):
#     def __init__(self):
#         super().__init__(model=model_id) 

#     def _call(self, prompt: str, callbacks=None, **kwargs) -> str:
#         print("prompt:", prompt)
#         """Call Claude-3 via Bedrock and return the response."""
#         try:
#             response = client.invoke_model(
#             modelId = "anthropic.claude-3-5-sonnet-20240620-v1:0",
#             contentType="application/json",
#             accept="application/json",
#             body=json.dumps({
#                 "messages": [
#                     {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": str(prompt).strip()  # ✅ Ensure it's always a string
#                     }
#                 ]
#             }
#                 ],
#                 "temperature": 0.5,
#                 "max_tokens": 1000,
#                 "anthropic_version": "bedrock-2023-05-31"
#             })
#         )
#             response_body = json.loads(response["body"].read().decode("utf-8"))
#             return response_body["content"][0]["text"].strip()
#         except Exception as e:
#             print(f"Error calling Bedrock: {e}")
#             return '{"error": "LLM call failed"}'

#     @property
#     def _llm_type(self) -> str:
#         return "bedrock-claude-3"

# custom_llm_wrapper = BedrockClaudeLLM()

# --- Your Existing MenuMatcherAgent ---
class MenuMatcherAgent:
    def __init__(self, lat, lng, extracted_items):
        self.lat = lat
        self.lng = lng
        self.extracted_items = extracted_items

    def run(self):
        headers = {"User-Agent": "Mozilla/5.0"}

        print("📍 Getting restaurants in area...")
        restaurant_url = f"https://www.swiggy.com/dapi/restaurants/list/v5?lat={self.lat}&lng={self.lng}&is-seo-homepage-enabled=true&page_type=DESKTOP_WEB_LISTING"
        resp = requests.get(restaurant_url, headers=headers)
        data = resp.json()

        restaurants = []
        cards = data.get("data", {}).get("cards", [])
        for card in cards:
            nested = card.get("card", {}).get("card", {}).get("gridElements", {}).get("infoWithStyle", {}).get("restaurants", [])
            if nested:
                restaurants.extend(nested)

        print(f"🔎 Found {len(restaurants)} restaurants. Scanning menus...")
        found_matches = []

        for restaurant in restaurants:
            info = restaurant.get("info", {})
            rest_name = info.get("name")
            rest_id = info.get("id")
            locality = info.get("locality")
            rating = info.get("avgRating")

            menu_url = f"https://www.swiggy.com/dapi/menu/pl?page-type=REGULAR_MENU&complete-menu=true&lat={self.lat}&lng={self.lng}&restaurantId={rest_id}"
            try:
                menu_resp = requests.get(menu_url, headers=headers)
                menu_data = menu_resp.json()
                items = menu_data["data"].get("cards", [])

                item_names = []
                item_map = []

                for card in items:
                    group = card.get("groupedCard", {}).get("cardGroupMap", {}).get("REGULAR", {}).get("cards", [])
                    for g in group:
                        item_card = g.get("card", {}).get("card", {})
                        if item_card.get("@type", "").endswith("ItemCategory"):
                            for item in item_card.get("itemCards", []):
                                item_info = item.get("card", {}).get("info", {})
                                name = item_info.get("name", "")
                                price_paise = item_info.get("price") or item_info.get("defaultPrice") or 0
                                price_rupees = price_paise / 100 if price_paise else "N/A"
                                description = item_info.get("description", "")
                                item_names.append(name.lower())
                                item_map.append({
                                    "name": name.lower(),
                                    "display_name": name,
                                    "price": price_rupees,
                                    "description": description,
                                })
                matched_any = False
                for q in self.extracted_items:
                    q_clean = q.lower().strip()
                    expanded_queries = expand_query(q_clean)

                    for item in item_map:
                        item_name_clean = item["name"].strip()

                        if any(eq in item_name_clean or item_name_clean in eq for eq in expanded_queries):
                                found_matches.append({
                                "restaurant": rest_name,
                                "matched_item": item["display_name"],
                                "query": q,
                                "location": locality,
                                "price": item["price"],
                                "ingredients": item["description"],
                                "rating": rating      # <-- Include rating in match
                            })

                if matched_any:
                    print(f"✅ Match found in {rest_name} ")
                else:
                    print(f"ℹ️ No match in {rest_name} — Sample items: {item_names[:5]}")

                time.sleep(0.7)

            except Exception as e:
                print(f"❌ Error fetching menu for {rest_name}: {e}")
                continue

        return found_matches

# --- 🆕 Meal Plan Strategist Agent ---
# class MealPlanStrategistAgent:
#     def __init__(self, llm: BedrockClaudeLLM):
#         self.llm = llm

#     def run(self, matched_items: List[Dict]) -> List[Dict]:
#         # Only send compact fields
#         compact_items = [
#             {
#                 "restaurant": item["restaurant"],
#                 "matched_item": item["matched_item"],
#                 "location": item["location"],
#                 "price": item["price"],
#                 "rating": item.get("rating", "N/A"), 
#                 "description": item["ingredients"] or "N/A"
#             }
#             for item in matched_items
#         ]

#         prompt = f"""
# You are a meal recommendation expert.

# You are given a list of menu items, each with:
# - restaurant
# - matched_item
# - location
# - price
# - rating
# - description (or missing ingredients)

# Your job:
# - From the given dishes, select Top 10 best meal options.
# - Prefer items from restaurants with HIGHER RATINGS.
# - Prefer HEALTHIER meals (less fried, less sugary, more veggies/protein).
# - For missing or vague descriptions, assume realistic ingredients intelligently.
# - Keep ingredient description short (under 20 words).
# - If two meals are similar, prefer the CHEAPER one.

# ⚡ Output Format:
# ***ONLY output a VALID JSON array list like***:
# [
#   {{
#     "restaurant": "Restaurant Name",
#     "matched_item": "Dish Name",
#     "location": "Area Name",
#     "price": 199.0,
#     "final_ingredients": "brief healthy ingredients"
#   }},
#   ...
# ]

# Do NOT write explanations. Do NOT write anything else. Only output clean JSON.

# Here is the menu data:
# {compact_items}
# """


#         result = self.llm._call(prompt)
#         try:
#             return json.loads(result)
#         except Exception as e:
#             print(f"Failed to parse response: {e}\nRaw Output: {result}")
#             return []


# # --- Run the full flow ---
# if __name__ == "__main__":
#     # Sample input
#     # extracted_items = ["cheesecake","Cheesecake", "Cheese cake", "cheese cake"]
#     extracted_items = ["wrap"]
#     lat = 17.4110382   # Manikonda
#     lng = 78.3725869   # Manikonda

#     # Step 1: Menu Matching
#     matcher = MenuMatcherAgent(lat=lat, lng=lng, extracted_items=extracted_items)
#     matches = matcher.run()

#     # Step 2: Meal Planning
#     custom_llm = BedrockClaudeLLM()
#     strategist = MealPlanStrategistAgent(llm=custom_llm)
#     top_meals = strategist.run(matches)
#     print("top_meals:", top_meals)
    
#     if not top_meals:
#         print("❌ No meal plans found.")
        
#     # Step 3: Output
#     print("\n🍽️ Top 10 Meal Plans:\n")
#     for idx, meal in enumerate(top_meals, 1):
#         print(f"{idx}. {meal['matched_item']} @ {meal['restaurant']} ({meal['location']}) — ₹{meal['price']}")
#         print(f"   Ingredients: {meal['final_ingredients']}\n")
