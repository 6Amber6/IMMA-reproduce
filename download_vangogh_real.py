"""
Download 20 real Van Gogh paintings from Wikimedia Commons.
All are public domain (Van Gogh died in 1890).
"""
import os
import requests
from pathlib import Path
from PIL import Image
from io import BytesIO

# 输出目录
OUT_DIR = Path("data/vangogh_real")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Wikimedia Commons File 名称(根据实际页面校对过)
PAINTINGS = [
    ("01_starry_night",          "Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"),
    ("02_starry_night_rhone",    "Starry_Night_Over_the_Rhone.jpg"),
    ("03_cafe_terrace_night",    "Vincent-van-gogh-cafe-terrace-on-the-place-du-forum-arles-at-night-the.jpg"),
    ("04_sunflowers_london",     "Vincent_van_Gogh_-_Sunflowers_(1888,_National_Gallery_London).jpg"),
    ("05_the_sower",             "Vincent_van_Gogh_-_The_sower_-_Google_Art_Project.jpg"),
    ("06_irises",                "Irises-Vincent_van_Gogh.jpg"),
    ("07_almond_blossoms",       "Vincent_van_Gogh_-_Almond_blossom_-_Google_Art_Project.jpg"),
    ("08_wheatfield_crows",      "Vincent_van_Gogh_-_Wheatfield_with_crows_-_Google_Art_Project.jpg"),
    ("09_harvest_blue_cart",     "Vincent_Van_Gogh,_A_Harvest_Landscape_with_Blue_Cart.jpg"),
    ("10_wheat_cypresses",       "Vincent_van_Gogh_-_Wheat_Field_with_Cypresses_(National_Gallery_version).jpg"),
    ("11_olive_trees",           "Vincent_van_Gogh_-_The_Olive_Trees_-_Google_Art_Project.jpg"),
    ("12_green_wheat_auvers",    "Van_Gogh_-_Grünes_Weizenfeld1.jpeg"),
    ("13_mulberry_tree",         "Van_Gogh_-_Maulbeerbaum.jpeg"),
    ("14_self_portrait_bandage", "Vincent_Willem_van_Gogh_106.jpg"),
    ("15_self_portrait_orsay",   "Vincent_van_Gogh_-_Self-Portrait_-_Google_Art_Project_(454045).jpg"),
    ("16_self_portrait_1887",    "Vincent_van_Gogh_-_Self-portrait_with_grey_felt_hat_-_Google_Art_Project.jpg"),
    ("17_dr_gachet",             "Portrait_of_Dr._Gachet.jpg"),
    ("18_bedroom_arles",         "Vincent_van_Gogh_-_De_slaapkamer_-_Google_Art_Project.jpg"),
    ("19_night_cafe",            "Vincent_Willem_van_Gogh_076.jpg"),
    ("20_yellow_house",          "WLANL_-_Minke_Wagenaar_-_Vincent_van_Gogh_1888_The_yellow_house_('The_street')_-_detail.jpg"),
]

HEADERS = {"User-Agent": "AcademicResearch/1.0 (contact: tong.li003@university.edu)"}


def get_commons_image_url(filename, width=1024):
    """通过Wikimedia API获取指定宽度的图片直链"""
    api = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": f"File:{filename}",
        "prop": "imageinfo",
        "iiprop": "url",
        "iiurlwidth": width,
        "format": "json",
    }
    r = requests.get(api, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    pages = r.json()["query"]["pages"]
    page = next(iter(pages.values()))
    if "imageinfo" not in page:
        return None
    return page["imageinfo"][0]["thumburl"]


def download_and_resize(url, out_path, target_size=512):
    """下载并 center-crop + resize 到 target_size×target_size"""
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content)).convert("RGB")

    # center crop 成正方形
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))

    # resize 到目标尺寸
    img = img.resize((target_size, target_size), Image.LANCZOS)
    img.save(out_path, "JPEG", quality=95)
    print(f"  saved -> {out_path}")


def main():
    failed = []
    for tag, filename in PAINTINGS:
        out_path = OUT_DIR / f"{tag}.jpg"
        if out_path.exists():
            print(f"[skip] {tag} already exists")
            continue
        print(f"[get ] {tag}")
        try:
            url = get_commons_image_url(filename, width=1024)
            if url is None:
                print(f"  !! not found on Commons: {filename}")
                failed.append((tag, filename))
                continue
            download_and_resize(url, out_path, target_size=512)
        except Exception as e:
            print(f"  !! error: {e}")
            failed.append((tag, filename))

    total_ok = 20 - len(failed)
    print(f"\n=== Done. {total_ok}/20 downloaded successfully. ===")
    if failed:
        print("Failed items (need manual download):")
        for tag, fn in failed:
            print(f"  - {tag}: https://commons.wikimedia.org/wiki/File:{fn}")
        print("\n→ Open each link, download manually, place into data/vangogh_real/ "
              "with the corresponding tag name (e.g. 01_starry_night.jpg).")


if __name__ == "__main__":
    main()
