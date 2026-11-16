import re
import requests
from bs4 import BeautifulSoup

class EbayListingIngestor:
    def __init__(self, ebay_oauth_token=None):
        self.oauth_token = ebay_oauth_token
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        }
        self.blacklisted_sellers = {"probstein123", "dcsports87", "pcsports_cards", "comc_consignment"}
        self.red_flag_keywords = ["reprint", "rp", "digital", "lot of", "you pick", "off center", "oc", "crease", "damaged"]

    @staticmethod
    def upgrade_to_high_res(img_url):
        """Rewrites eBay CDN image URLs (e.g., s-l225.jpg, s-l500.webp) to uncompressed 1600px JPG."""
        if not img_url:
            return None
        return re.sub(r"s-l\d+\.(jpg|webp|png)", "s-l1600.jpg", img_url)

    def fetch_single_listing_by_url_or_id(self, url_or_id):
        """Extracts Title, Price, and High-Res Front/Back Images directly from an eBay URL or 12-digit Item ID."""
        match = re.search(r"(\d{12})", str(url_or_id))
        item_id = match.group(1) if match else str(url_or_id).strip()
        target_url = f"https://www.ebay.com/itm/{item_id}"

        resp = requests.get(target_url, headers=self.headers, timeout=12)
        if resp.status_code != 200:
            raise ValueError(f"HTTP {resp.status_code} when fetching item {item_id}")

        soup = BeautifulSoup(resp.text, "html.parser")

        title_el = soup.select_one("h1.x-item-title__mainTitle")
        title = title_el.get_text(strip=True) if title_el else f"eBay Item #{item_id}"

        price_el = soup.select_one("div.x-price-primary span.ux-textspans")
        price_val = 35.0
        if price_el:
            nums = re.findall(r"\d+\.\d+", price_el.get_text().replace(",", ""))
            if nums:
                price_val = float(nums[0])

        raw_img_urls = []
        for img_tag in soup.select("div.ux-image-carousel-item img"):
            src = img_tag.get("data-zoom-src") or img_tag.get("src")
            if src and "ebayimg.com" in src:
                hi_res = self.upgrade_to_high_res(src)
                if hi_res not in raw_img_urls:
                    raw_img_urls.append(hi_res)

        return {
            "item_id": item_id,
            "title": title,
            "price": price_val,
            "shipping": 0.0,
            "total_cost": price_val,
            "url": target_url,
            "front_img_url": raw_img_urls[0] if len(raw_img_urls) > 0 else None,
            "back_img_url": raw_img_urls[1] if len(raw_img_urls) > 1 else None,
            "all_images": raw_img_urls
        }

    def search_live_candidates_api(self, query="2018 Topps Update US189 Shohei Ohtani", max_price=45.0, limit=25):
        """Uses Official eBay Browse API to pull newly listed Ungraded (4000) Buy-It-Now listings."""
        if not self.oauth_token:
            raise ValueError("eBay OAuth Token required for automated Browse API batch search.")

        endpoint = "https://api.ebay.com/buy/browse/v1/item_summary/search"
        api_headers = {
            "Authorization": f"Bearer {self.oauth_token}",
            "X-EBAY-C-MARKETPLACE-ID": "EBAY_US"
        }
        params = {
            "q": f"{query} -PSA -BGS -SGC -CGC -reprint -lot",
            "filter": f"conditionIds:{{4000}},buyingOptions:{{FIXED_PRICE}},price:[10..{max_price}],priceCurrency:USD",
            "sort": "newlyListed",
            "limit": limit
        }

        r = requests.get(endpoint, headers=api_headers, params=params, timeout=15)
        r.raise_for_status()
        items = r.json().get("itemSummaries", [])

        candidates = []
        for item in items:
            title_lower = item.get("title", "").lower()
            seller_name = item.get("seller", {}).get("username", "").lower()
            seller_pct = float(item.get("seller", {}).get("feedbackPercentage", "0"))

            if any(bad in title_lower for bad in self.red_flag_keywords):
                continue
            if seller_name in self.blacklisted_sellers or seller_pct < 98.0:
                continue

            price = float(item.get("price", {}).get("value", 0))
            ship_opts = item.get("shippingOptions", [{}])
            ship_cost = float(ship_opts[0].get("shippingCost", {}).get("value", 0)) if ship_opts else 0.0

            if (price + ship_cost) > max_price:
                continue

            front_url = self.upgrade_to_high_res(item.get("image", {}).get("imageUrl"))
            add_imgs = item.get("additionalImages", [])
            back_url = self.upgrade_to_high_res(add_imgs[0].get("imageUrl")) if add_imgs else None

            if not front_url or not back_url:
                continue

            candidates.append({
                "item_id": item.get("legacyItemId") or item.get("itemId"),
                "title": item.get("title"),
                "seller": f"{seller_name} ({seller_pct}%)",
                "price": price,
                "shipping": ship_cost,
                "total_cost": round(price + ship_cost, 2),
                "url": item.get("itemWebUrl"),
                "front_img_url": front_url,
                "back_img_url": back_url
            })

        return candidates

    def download_image_bytes(self, img_url):
        """Downloads high-res image bytes into memory for OpenCV processing."""
        resp = requests.get(img_url, headers=self.headers, timeout=10)
        resp.raise_for_status()
        return resp.content
