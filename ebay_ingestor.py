import re
import json
import base64
import requests
from bs4 import BeautifulSoup

# Try importing curl_cffi for Chrome TLS fingerprint spoofing
try:
    from curl_cffi import requests as cffi_requests
    HAS_CFFI = True
except ImportError:
    HAS_CFFI = False


class EbayListingIngestor:
    def __init__(self, ebay_oauth_token=None, client_id=None, client_secret=None):
        self.oauth_token = ebay_oauth_token
        self.client_id = client_id
        self.client_secret = client_secret

        self.browser_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.google.com/",
            "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "cross-site",
            "Upgrade-Insecure-Requests": "1",
        }
        self.blacklisted_sellers = {"probstein123", "dcsports87", "pcsports_cards", "comc_consignment"}
        self.red_flag_keywords = ["reprint", "rp", "digital", "lot of", "you pick", "off center", "oc", "crease", "damaged"]

    def get_or_mint_oauth_token(self):
        """Automatically exchanges eBay Client ID + Secret for a fresh 2-hour Application Token."""
        if self.oauth_token:
            return self.oauth_token
        if not (self.client_id and self.client_secret):
            return None

        creds = f"{self.client_id}:{self.client_secret}"
        b64_creds = base64.b64encode(creds.encode("utf-8")).decode("utf-8")

        resp = requests.post(
            "https://api.ebay.com/identity/v1/oauth2/token",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Authorization": f"Basic {b64_creds}",
            },
            data={
                "grant_type": "client_credentials",
                "scope": "https://api.ebay.com/oauth/api_scope",
            },
            timeout=10,
        )
        if resp.status_code == 200:
            self.oauth_token = resp.json().get("access_token")
            return self.oauth_token
        return None

    @staticmethod
    def upgrade_to_high_res(img_url):
        """Rewrites eBay CDN image URLs (s-l225, s-l500, etc.) to uncompressed 1600px JPG."""
        if not img_url:
            return None
        return re.sub(r"s-l\d+\.(jpg|webp|png|jpeg)", "s-l1600.jpg", img_url)

    def fetch_single_listing_by_url_or_id(self, url_or_id):
        """
        3-Tier Cascade to fetch an eBay item without triggering HTTP 403:
        1. Official eBay Browse API (if credentials/token provided) — 100% immune to 403.
        2. curl_cffi Chrome 124 TLS Impersonation scrape.
        3. Mobile/Regex CDN hash extraction fallback.
        """
        match = re.search(r"(\d{12})", str(url_or_id))
        if not match:
            raise ValueError(f"Could not find a valid 12-digit eBay Item ID in: {url_or_id}")
        item_id = match.group(1)
        target_url = f"https://www.ebay.com/itm/{item_id}"

        # --- TIER 1: OFFICIAL EBAY BROWSE API (100% Cloud Safe) ---
        token = self.get_or_mint_oauth_token()
        if token:
            try:
                return self._fetch_single_via_api(item_id, token)
            except Exception:
                pass  # Fall through to TLS scraper if API call fails

        # --- TIER 2: CHROME TLS IMPERSONATION SCRAPER ---
        html_text = None
        status_code = None

        if HAS_CFFI:
            resp = cffi_requests.get(
                target_url,
                headers=self.browser_headers,
                impersonate="chrome120",
                timeout=15,
            )
            status_code = resp.status_code
            if status_code == 200:
                html_text = resp.text

        # Fallback to mobile layout URL if desktop returns 403 or curl_cffi isn't installed
        if not html_text:
            session = requests.Session()
            resp = session.get(
                f"https://m.ebay.com/itm/{item_id}",
                headers=self.browser_headers,
                timeout=15,
            )
            status_code = resp.status_code
            if status_code == 200:
                html_text = resp.text

        if not html_text:
            raise ValueError(
                f"eBay blocked cloud IP (HTTP {status_code}) for {item_id}. "
                "Add your free eBay API Client ID/Secret in the sidebar or install curl_cffi."
            )

        return self._parse_ebay_html(item_id, target_url, html_text)

    def _fetch_single_via_api(self, item_id, token):
        """Fetches a single 12-digit legacy Item ID via the Official eBay Browse API."""
        endpoint = "https://api.ebay.com/buy/browse/v1/item/get_item_by_legacy_id"
        headers = {
            "Authorization": f"Bearer {token}",
            "X-EBAY-C-MARKETPLACE-ID": "EBAY_US",
        }
        r = requests.get(endpoint, headers=headers, params={"legacy_item_id": item_id}, timeout=12)
        r.raise_for_status()
        data = r.json()

        price = float(data.get("price", {}).get("value", 35.0))
        ship_opts = data.get("shippingOptions", [{}])
        ship_cost = float(ship_opts[0].get("shippingCost", {}).get("value", 0.0)) if ship_opts else 0.0

        front_url = self.upgrade_to_high_res(data.get("image", {}).get("imageUrl"))
        add_imgs = data.get("additionalImages", [])
        all_imgs = [front_url] if front_url else []
        for img_obj in add_imgs:
            u = self.upgrade_to_high_res(img_obj.get("imageUrl"))
            if u and u not in all_imgs:
                all_imgs.append(u)

        return {
            "item_id": item_id,
            "title": data.get("title", f"eBay Item #{item_id}"),
            "price": price,
            "shipping": ship_cost,
            "total_cost": round(price + ship_cost, 2),
            "url": data.get("itemWebUrl", f"https://www.ebay.com/itm/{item_id}"),
            "front_img_url": all_imgs[0] if len(all_imgs) > 0 else None,
            "back_img_url": all_imgs[1] if len(all_imgs) > 1 else None,
            "all_images": all_imgs,
        }

    def _parse_ebay_html(self, item_id, target_url, html_text):
        """Extracts Title, Price, and 1600px Image hashes from DOM or raw regex."""
        soup = BeautifulSoup(html_text, "html.parser")

        # 1. Title extraction (DOM + OpenGraph fallback)
        title_el = soup.select_one("h1.x-item-title__mainTitle") or soup.find("meta", property="og:title")
        if title_el:
            title = title_el.get("content") if title_el.name == "meta" else title_el.get_text(strip=True)
        else:
            title = f"eBay Item #{item_id}"

        # 2. Price extraction
        price_val = 35.0
        price_el = soup.select_one("div.x-price-primary span.ux-textspans")
        if price_el:
            nums = re.findall(r"\d+\.\d+", price_el.get_text().replace(",", ""))
            if nums:
                price_val = float(nums[0])

        # 3. Carousel Image extraction + Regex CDN hash fallback
        raw_img_urls = []
        for img_tag in soup.select("div.ux-image-carousel-item img"):
            src = img_tag.get("data-zoom-src") or img_tag.get("src")
            if src and "ebayimg.com" in src:
                hi_res = self.upgrade_to_high_res(src)
                if hi_res not in raw_img_urls:
                    raw_img_urls.append(hi_res)

        # Regex Fallback: Scan raw HTML for unique eBay image hashes (/images/g/HASH/s-l...)
        if len(raw_img_urls) < 2:
            hashes = re.findall(r"https://i\.ebayimg\.com/images/g/[A-Za-z0-9_-]+/s-l\d+\.(?:jpg|webp|png)", html_text)
            seen_hashes = set()
            for full_url in hashes:
                m_hash = re.search(r"/images/g/([A-Za-z0-9_-]+)/", full_url)
                if m_hash and m_hash.group(1) not in seen_hashes:
                    seen_hashes.add(m_hash.group(1))
                    upgraded = self.upgrade_to_high_res(full_url)
                    if upgraded not in raw_img_urls:
                        raw_img_urls.append(upgraded)

        if not raw_img_urls:
            raise ValueError(f"Page loaded for {item_id}, but eBay hid gallery images behind a JS challenge.")

        return {
            "item_id": item_id,
            "title": title,
            "price": price_val,
            "shipping": 0.0,
            "total_cost": price_val,
            "url": target_url,
            "front_img_url": raw_img_urls[0] if len(raw_img_urls) > 0 else None,
            "back_img_url": raw_img_urls[1] if len(raw_img_urls) > 1 else None,
            "all_images": raw_img_urls,
        }

    def search_live_candidates_api(self, query="2018 Topps Update US189 Shohei Ohtani", max_price=45.0, limit=25):
        """Uses Official eBay Browse API to pull newly listed Ungraded (4000) Buy-It-Now listings."""
        token = self.get_or_mint_oauth_token()
        if not token:
            raise ValueError("Provide an eBay OAuth Token or Client ID + Secret to run Browse API sweeps.")

        endpoint = "https://api.ebay.com/buy/browse/v1/item_summary/search"
        api_headers = {
            "Authorization": f"Bearer {token}",
            "X-EBAY-C-MARKETPLACE-ID": "EBAY_US",
        }
        params = {
            "q": f"{query} -PSA -BGS -SGC -CGC -reprint -lot",
            "filter": f"conditionIds:{{4000}},buyingOptions:{{FIXED_PRICE}},price:[10..{max_price}],priceCurrency:USD",
            "sort": "newlyListed",
            "limit": limit,
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
                "back_img_url": back_url,
            })

        return candidates

    def download_image_bytes(self, img_url):
        """Downloads high-res image bytes from i.ebayimg.com (CDN does not block cloud IPs)."""
        if HAS_CFFI:
            resp = cffi_requests.get(img_url, headers=self.browser_headers, impersonate="chrome120", timeout=10)
        else:
            resp = requests.get(img_url, headers=self.browser_headers, timeout=10)
        resp.raise_for_status()
        return resp.content
