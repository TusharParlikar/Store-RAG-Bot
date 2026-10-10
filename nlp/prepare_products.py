"""Turn the downloaded IKEA dataset (data/raw/ikea.csv) into data/products/products.csv.

Source: TidyTuesday 2020-11-03 (IKEA Saudi Arabia scrape, also on Kaggle as
"IKEA SA Furniture Web Scraping"). Raw prices are in SAR; they are converted to INR
at SAR_TO_INR and rounded to whole rupees. The original is kept in price_sar.

The raw data has name, category, price and description. The columns the bot
needs but the dataset lacks (warranty_months, benefit, good_for, goes_with) are
filled per category from CATEGORY_INFO below. Edit that table to change them.
"""

import re
from pathlib import Path

import pandas as pd

# 1 SAR in rupees. SAR is pegged at 3.75 per USD, so this is about USD/INR / 3.75. Update and rerun to reprice.
SAR_TO_INR = 23.5

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "ikea.csv"
OUT = ROOT / "data" / "products" / "products.csv"

# category: (warranty_months, benefit, good_for, goes_with)
CATEGORY_INFO = {
    "Tables & desks": (
        60,
        "a steady surface for meals, work or study",
        "dining, home-office, long-hours",
        "Chairs; Cabinets & cupboards",
    ),
    "Chairs": (
        60,
        "a supportive seat that keeps you comfortable while sitting",
        "long-hours, back-pain, dining, home-office",
        "Tables & desks; Cafe furniture",
    ),
    "Sofas & armchairs": (
        120,
        "soft seating to relax with family and guests",
        "living-room, relaxing, guests",
        "TV & media furniture; Tables & desks",
    ),
    "Bookcases & shelving units": (
        60,
        "open storage that keeps books and decor in sight and in order",
        "storage, small-room, decoration",
        "Cabinets & cupboards; Chests of drawers & drawer units",
    ),
    "Cabinets & cupboards": (
        60,
        "closed storage that hides clutter and keeps rooms tidy",
        "storage, office, small-room",
        "Bookcases & shelving units; Tables & desks",
    ),
    "Wardrobes": (
        120,
        "keeps clothes hung, folded and easy to find",
        "bedroom, storage",
        "Beds; Chests of drawers & drawer units",
    ),
    "Outdoor furniture": (
        12,
        "made for use outside on a balcony, patio or garden",
        "outdoor, balcony, guests",
        "Outdoor furniture; Cafe furniture",
    ),
    "Beds": (
        120,
        "a stable frame for restful sleep",
        "bedroom, sleep",
        "Wardrobes; Chests of drawers & drawer units",
    ),
    "TV & media furniture": (
        60,
        "keeps the TV, devices and cables organised in one place",
        "living-room, storage",
        "Sofas & armchairs; Bookcases & shelving units",
    ),
    "Chests of drawers & drawer units": (
        60,
        "drawer storage for clothes, papers and small items",
        "bedroom, storage, small-room",
        "Beds; Wardrobes",
    ),
    "Children's furniture": (
        24,
        "sized and shaped for children to use safely",
        "kids, playroom, storage",
        "Children's furniture; Bookcases & shelving units",
    ),
    "Nursery furniture": (
        24,
        "practical furniture for caring for a baby",
        "baby, nursery",
        "Nursery furniture; Chests of drawers & drawer units",
    ),
    "Bar furniture": (
        24,
        "raised seating and tables for a casual bar or kitchen counter",
        "kitchen, guests, small-room",
        "Bar furniture; Cafe furniture",
    ),
    "Trolleys": (
        24,
        "movable storage that rolls to where you need it",
        "kitchen, small-room, storage",
        "Tables & desks; Cabinets & cupboards",
    ),
    "Cafe furniture": (
        24,
        "compact tables and chairs for cafes and small dining spaces",
        "cafe, dining, small-room",
        "Cafe furniture; Chairs",
    ),
    "Sideboards, buffets & console tables": (
        60,
        "storage and a display surface for dining or hallway spaces",
        "dining, decoration, storage",
        "Tables & desks; Chairs",
    ),
    "Room dividers": (
        24,
        "splits a room into zones without building walls",
        "small-room, office, decoration",
        "Bookcases & shelving units; Tables & desks",
    ),
}


def clean(text) -> str:
    return re.sub(r"\s+", " ", str(text)).strip(" ,") if pd.notna(text) else ""


def main():
    df = pd.read_csv(RAW, index_col=0, encoding="utf-8")
    # The raw file stores "Café" with a broken character.
    df["category"] = df["category"].where(~df["category"].str.startswith("Caf"), "Cafe furniture")
    df = df.drop_duplicates("item_id")  # same item is listed under several categories

    unknown = set(df["category"]) - set(CATEGORY_INFO)
    assert not unknown, f"add these categories to CATEGORY_INFO: {unknown}"

    info = df["category"].map(CATEGORY_INFO)
    out = pd.DataFrame(
        {
            "item_id": df["item_id"],
            "name": df["name"].map(clean) + " - " + df["short_description"].map(clean),
            "category": df["category"],
            "price": (df["price"] * SAR_TO_INR).round().astype(int),  # INR
            "price_sar": df["price"],
            "warranty_months": info.str[0],
            "shelf_life": "",  # no furniture in this dataset expires
            "benefit": info.str[1],
            "good_for": info.str[2],
            "goes_with": info.str[3],
            "designer": df["designer"].map(clean),
            "link": df["link"],
        }
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    required = ["name", "category", "price", "warranty_months", "benefit", "good_for", "goes_with"]
    assert (
        out[required].notna().all().all() and (out[required] != "").all().all()
    ), "empty required column"
    print(f"wrote {len(out)} products to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
