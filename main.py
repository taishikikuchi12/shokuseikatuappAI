from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
try:
    # Pydantic v2
    from pydantic import ConfigDict  # type: ignore
    _HAS_PYDANTIC_V2 = True
except Exception:
    # Pydantic v1
    ConfigDict = None  # type: ignore
    _HAS_PYDANTIC_V2 = False

import json
import os
from typing import Dict
import io
from PIL import Image, UnidentifiedImageError
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

app = FastAPI()

# ---- CORS（必要な場合だけ使ってください。ローカル開発だと便利） ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番は絞る
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Food-101 画像分類モデルを読み込む ----
MODEL_NAME = "prithivMLmods/Food-101-93M"

image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
image_model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
image_model.eval()

# ---- Pydantic モデル（Swift側と対応） ----
class Nutrition(BaseModel):
    # JSONは energyKcal（camelCase）なので alias で受ける
    energy_kcal: float = Field(..., alias="energyKcal")
    protein: float
    fat: float
    carbs: float

    # 微量栄養素（抽象値）
    vitamin: float = 0.0
    mineral: float = 0.0

    # 互換性のため当面残す（JSONにある前提）
    sugar: float = 0.0
    salt: float = 0.0

    # JSONに入れているので受け取れるように（無い場合もあるので default）
    fiber: float = 0.0

    # Pydantic v1/v2 compatibility:
    # - accept alias keys (energyKcal)
    # - allow population by field name too
    if _HAS_PYDANTIC_V2 and ConfigDict is not None:
        model_config = ConfigDict(populate_by_name=True)
    else:
        class Config:
            allow_population_by_field_name = True


class MealPrediction(BaseModel):
    name: str              # 料理名（例：ramen）
    unit: str              # 単位（例：1食分）
    base_amount: float     # 基準量（例：1.0 = 1食分）
    nutrition_per_unit: Nutrition


# ---- Food-101 各食品ごとの栄養データベース（JSONから読み込み） ----
FOOD_NUTRITION_DB: Dict[str, Nutrition] = {}

DB_PATH = os.environ.get("NUTRITION_DB_PATH", "FoodNutrition.json")


def load_nutrition_db(path: str) -> Dict[str, Nutrition]:
    """
    FoodNutrition.json を読み込み、{label: Nutrition} を作る。
    JSONは { "ramen": { ... }, ... } の形式を想定。
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("Nutrition DB JSON must be an object at top-level")
    return {k: Nutrition(**v) for k, v in raw.items()}


# 起動時に読み込み（失敗してもサーバーは起動するが、フォールバック値になる）
try:
    FOOD_NUTRITION_DB = load_nutrition_db(DB_PATH)
    print(f"[INFO] Loaded nutrition DB: {len(FOOD_NUTRITION_DB)} items from {DB_PATH}")
except Exception as e:
    print(f"[WARN] Failed to load nutrition DB from {DB_PATH}: {e}")
    FOOD_NUTRITION_DB = {}


# ---- 料理ラベルから栄養情報を取得する ----
def generate_nutrition_for_label(food_label: str) -> Nutrition:
    base = FOOD_NUTRITION_DB.get(food_label)
    if base is not None:
        return base

    # 万が一DBに存在しないラベルが来た場合のフォールバック
    return Nutrition(
        energyKcal=400,
        protein=15,
        fat=15,
        carbs=50,
        vitamin=0.0,
        mineral=0.0,
        sugar=10,
        salt=2.0,
        fiber=0.0,
    )


# ---- 画像から料理名を推定する ----
def classify_food_from_image(image_bytes: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError(f"Invalid image bytes: {e}")

    inputs = image_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = image_model(**inputs)
        logits = outputs.logits
        predicted_id = int(torch.argmax(logits, dim=-1))

    return image_model.config.id2label[predicted_id]


# ---- API エンドポイント ----
@app.post(
    "/analyze_meal",
    response_model=MealPrediction,
    # Swift側が camelCase を期待することが多いので alias で返す（energyKcal など）
    response_model_by_alias=True,
)
async def analyze_meal(request: Request):
    """
    iOSアプリから送られてきた画像バイト列を受け取り、
    料理名と栄養情報を返すエンドポイント。
    Content-Type: application/octet-stream を想定。
    """
    image_bytes = await request.body()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="No image bytes received")

    try:
        food_label = classify_food_from_image(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    nutrition = generate_nutrition_for_label(food_label)

    return MealPrediction(
        name=food_label,
        unit="1食分",
        base_amount=1.0,
        nutrition_per_unit=nutrition,
    )


# ---- ローカル実行用：サーバー起動 ----
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )