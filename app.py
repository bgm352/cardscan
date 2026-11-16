import streamlit as st
import pandas as pd
import numpy as np
import cv2

from cv_engine import CardSieveEngine
from learning_engine import RecursiveLearningEngine
from ebay_ingestor import EbayListingIngestor

st.set_page_config(
    page_title="Statsline | Quantitative CV Deal Sieve",
    page_icon="🎯",
    layout="wide"
)

# Initialize Recursive Brain
brain = RecursiveLearningEngine()
stats = brain.get_calibration_stats()

st.title("🎯 Statsline: Quantitative CV Deal Sieve")
st.caption("Target Asset: **2018 Topps Update Shohei Ohtani (#US189)** | Protocol: *Anchor. Loupe. Liquidate.*")

# --- TOP RECURSIVE LEARNING TELEMETRY BANNER ---
t_col1, t_col2, t_col3, t_col4 = st.columns(4)
t_col1.metric("🧠 Optical Training Samples", f"{stats['optical_samples']} Overrides")
t_col2.metric("🎯 CV Caliper Drift (MAE)", f"±{stats['mean_pixel_error']} px")
t_col3.metric("🏆 Logged PSA Returns", f"{stats['psa_returns_logged']} Slabs")
t_col4.metric("📈 Personal Gem Rate", f"{stats['historical_gem_rate']}%")

st.divider()

# --- SIDEBAR: GEOMETRY GATES, UNIT ECONOMICS & PSA LEDGER ---
with st.sidebar:
    st.header("⚙️ 1. Geometry Gates")
    front_gate = st.slider("Max Front Centering Tolerance (%)", 50.0, 65.0, 55.0, 0.5,
                           help="PSA 10 standard requires 55/45 or better on the front.")
    back_gate = st.slider("Max Back Centering Tolerance (%)", 50.0, 75.0, 60.0, 0.5,
                          help="Strict Loupe Gate defaults to 60/40 to avoid grader subjectivity.")
    lum_thresh = st.slider("Baseline Luminance Cutoff", 140, 245, 200, 5,
                           help="Default brightness cutoff before ML auto-tuning activates.")
    auto_crop = st.checkbox("Auto-Isolate Card from Background", value=False)

    st.divider()
    st.header("💰 2. Unit Economics (#US189)")
    raw_ask = st.number_input("Default Raw Price ($)", value=35.0, step=1.0)
    grading_fee = st.number_input("PSA Grading + Ship Fee ($)", value=24.0, step=1.0)
    psa10_comp = st.number_input("PSA 10 Market Value ($)", value=165.0, step=5.0)
    psa9_comp = st.number_input("PSA 9 Market Value ($)", value=62.0, step=2.0)

    st.divider()
    with st.expander("🏆 Log PSA Return (Slow Loop)"):
        ret_id = st.text_input("eBay Item # / Cert ID", placeholder="e.g. 336856970182")
        ret_front = st.number_input("Front Worst Axis (%)", 50.0, 70.0, 52.5, 0.5)
        ret_back = st.number_input("Back Worst Axis (%)", 50.0, 75.0, 56.0, 0.5)
        ret_grade = st.selectbox("Actual PSA Grade Received", [10, 9, 8])

        if st.button("Save PSA Outcome to Ledger", use_container_width=True):
            brain.log_psa_outcome(ret_id or "Manual-Entry", ret_front, ret_back, 0.0, ret_grade)
            st.success("Outcome recorded! Gem Probability model updated.")
            st.rerun()

engine = CardSieveEngine(front_max_tolerance=front_gate, back_max_tolerance=back_gate)

# --- 3-TAB COMMAND CENTER ---
tab_upload, tab_live, tab_memory = st.tabs([
    "📥 Manual Screenshot Dropzone & Caliper Trainer",
    "🌐 Live eBay Ingestion & Auto-Sieve",
    "🧠 Recursive Memory & EV Matrix"
])

# ==============================================================================
# TAB 1: MANUAL SCREENSHOT DROPZONE & RECURSIVE CALIPER TRAINER
# ==============================================================================
with tab_upload:
    uploaded_files = st.file_uploader(
        "Drop eBay Listing Screenshots Here (e.g., 5114.png, 5115.png, 5116.png, 5117.png)",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("👆 Upload one or more card screenshots above to run the automated geometry sieve and train the optical calipers.")
    else:
        batch_results = []
        processed_images = {}

        for f in uploaded_files:
            raw_bytes = f.read()
            norm_img = engine.load_and_normalize(raw_bytes, auto_crop=auto_crop)
            gray_img = cv2.cvtColor(norm_img, cv2.COLOR_BGR2GRAY)

            mean_lum = np.mean(norm_img)
            default_side = "BACK" if mean_lum > 125 else "FRONT"

            learned_thresh, is_trained, _ = brain.predict_optimal_lum_threshold(
                gray_img, side=default_side, default_thresh=lum_thresh
            )
            res = engine.analyze_card_side(norm_img, side=default_side, lum_threshold=learned_thresh)

            # Calculate Gem Probability & EV
            f_worst = res["worst_axis"] if default_side == "FRONT" else 52.5
            b_worst = res["worst_axis"] if default_side == "BACK" else 56.5
            gem_prob, _, _ = brain.predict_gem_probability(f_worst, b_worst)

            total_cost = raw_ask + grading_fee
            p_gem = gem_prob / 100.0
            expected_net = (p_gem * (psa10_comp * 0.87)) + ((1 - p_gem) * (psa9_comp * 0.87)) - total_cost

            batch_results.append({
                "File": f.name,
                "Side": default_side,
                "Centering Ratio": res["ratio_str"],
                "Worst Axis (%)": res["worst_axis"],
                "Gate Limit (%)": res["gate_limit"],
                "Gem Prob (%)": f"{gem_prob:.1f}%",
                "Expected EV ($)": f"${expected_net:+.2f}",
                "Verdict": "✅ PASS GATE" if res["passed"] else "❌ AUTO-REJECT"
            })
            processed_images[f.name] = (norm_img, gray_img, default_side, res, learned_thresh, is_trained)

        st.subheader(f"📊 Batch Triage Queue ({len(uploaded_files)} Screenshots Evaluated)")
        passed_count = sum(1 for r in batch_results if "PASS" in r["Verdict"])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Scanned", len(uploaded_files))
        m2.metric("Passed Geometry Gate", passed_count)
        m3.metric("Auto-Rejected Duds", len(uploaded_files) - passed_count)
        m4.metric("PSA Fees Saved", f"${(len(uploaded_files) - passed_count) * grading_fee:.2f}")

        st.dataframe(pd.DataFrame(batch_results), use_container_width=True, hide_index=True)
        st.divider()

        # Deep-Dive Inspector & Recursive Trainer
        st.subheader("🔬 Deep-Dive Optical Caliper & Recursive Trainer")
        selected_file = st.selectbox("Select Screenshot to Inspect / Calibrate:", [f.name for f in uploaded_files])
        norm_img, gray_img, detected_side, auto_res, active_thresh, is_trained = processed_images[selected_file]

        col_ctrl, col_view = st.columns([1, 1.35])
        with col_ctrl:
            side_mode = st.radio("Card Orientation", ["BACK", "FRONT"],
                                 index=0 if detected_side == "BACK" else 1, horizontal=True)

            learned_t, trained_flag, n_samp = brain.predict_optimal_lum_threshold(
                gray_img, side=side_mode, default_thresh=lum_thresh
            )
            if trained_flag:
                st.caption(f"🧠 **ML Auto-Tuned Cutoff:** `{learned_t}` (learned from {n_samp} overrides)")
            else:
                st.caption(f"⚙️ **Baseline Cutoff:** `{learned_t}` ({5 - n_samp} more overrides needed to activate ML)")

            use_manual = st.checkbox("Enable Manual Caliper Override (Teach the Scanner)", value=False)
            if use_manual:
                m_left = st.slider("Left Border / Anchor Width (px)", 5, 250, auto_res["left_px"])
                m_right = st.slider("Right Border / Anchor Width (px)", 5, 250, auto_res["right_px"])
                live_res = engine.analyze_card_side(norm_img, side=side_mode, lum_threshold=learned_t,
                                                    manual_left_px=m_left, manual_right_px=m_right)

                if st.button("🧠 Commit Caliper Override to Training Memory", type="primary", use_container_width=True):
                    ideal_cutoff = brain.log_caliper_override(
                        gray_img, side_mode,
                        auto_l=auto_res["left_px"], auto_r=auto_res["right_px"],
                        true_l=m_left, true_r=m_right
                    )
                    st.toast(f"Learned! Reverse-engineered ideal luminance cutoff ({ideal_cutoff}).", icon="🧠")
                    st.rerun()
            else:
                live_res = engine.analyze_card_side(norm_img, side=side_mode, lum_threshold=learned_t)

            if live_res["passed"]:
                st.success(f"**✅ PASS ({side_mode} GATE):** {live_res['ratio_str']} is within your {live_res['gate_limit']}% tolerance.")
            else:
                st.error(f"**❌ REJECT ({side_mode} SKEW):** {live_res['ratio_str']} exceeds your {live_res['gate_limit']}% max gate.")

            st.markdown("#### 📈 Horizontal Luminance Scanline (Y = 700px)")
            st.line_chart(pd.DataFrame({"Luminance (0-255)": live_res["scanline"]}), height=180)

        with col_view:
            st.image(live_res["overlay_rgb"], caption=f"Statsline CV HUD Overlay — {selected_file}", use_container_width=True)

# ==============================================================================
# TAB 2: LIVE EBAY INGESTION & DUAL FRONT+BACK AUTO-SIEVE
# ==============================================================================
with tab_live:
    st.subheader("🌐 Automated eBay Listing Sieve (1600px Front + Back Dual Scan)")
    ingestor = EbayListingIngestor()

    col_l1, col_l2 = st.columns([2, 1])
    with col_l1:
        urls_input = st.text_area(
            "Paste eBay Item URLs or 12-Digit Item IDs (One per line):",
            placeholder="https://www.ebay.com/itm/336856970182\nhttps://www.ebay.com/itm/137826440827",
            height=120
        )
        run_url_batch = st.button("🔍 Pull 1600px Gallery Photos & Run Dual CV Sieve", type="primary")

    with col_l2:
        st.markdown("**Optional: Official Browse API Sweep**")
        ebay_token = st.text_input("eBay OAuth Bearer Token", type="password")
        max_buy_price = st.number_input("Max All-In Raw Ask ($)", value=42.0, step=1.0)
        run_auto_sweep = st.button("🚀 Sweep Newly Listed #US189s", use_container_width=True)

    listings_to_scan = []
    if run_url_batch and urls_input.strip():
        raw_lines = [line.strip() for line in urls_input.split("\n") if line.strip()]
        with st.spinner(f"Pulling high-res gallery images for {len(raw_lines)} listings..."):
            for line in raw_lines:
                try:
                    listings_to_scan.append(ingestor.fetch_single_listing_by_url_or_id(line))
                except Exception as e:
                    st.warning(f"Skipped `{line}`: {e}")

    elif run_auto_sweep and ebay_token:
        ingestor.oauth_token = ebay_token
        with st.spinner("Querying eBay Browse API for newly listed raw #US189s..."):
            try:
                listings_to_scan = ingestor.search_live_candidates_api(max_price=max_buy_price)
            except Exception as e:
                st.error(f"API Error: {e}")

    if listings_to_scan:
        live_rows = []
        preview_overlays = {}

        for item in listings_to_scan:
            try:
                f_bytes = ingestor.download_image_bytes(item["front_img_url"])
                f_img = engine.load_and_normalize(f_bytes, auto_crop=True)
                f_res = engine.analyze_card_side(f_img, side="FRONT", lum_threshold=lum_thresh)

                if item["back_img_url"]:
                    b_bytes = ingestor.download_image_bytes(item["back_img_url"])
                    b_img = engine.load_and_normalize(b_bytes, auto_crop=True)
                    b_res = engine.analyze_card_side(b_img, side="BACK", lum_threshold=lum_thresh)
                    back_ratio = b_res["ratio_str"]
                    back_worst = b_res["worst_axis"]
                    both_passed = f_res["passed"] and b_res["passed"]
                    b_overlay = b_res["overlay_rgb"]
                else:
                    back_ratio = "Missing Back Photo"
                    back_worst = 65.0
                    both_passed = False
                    b_overlay = None

                gem_prob, _, _ = brain.predict_gem_probability(f_res["worst_axis"], back_worst)
                p_gem = gem_prob / 100.0
                all_in = item["total_cost"] + grading_fee
                expected_ev = (p_gem * (psa10_comp * 0.87)) + ((1 - p_gem) * (psa9_comp * 0.87)) - all_in

                live_rows.append({
                    "Item ID": item["item_id"],
                    "Title": item["title"][:45] + "...",
                    "Ask Price": f"${item['total_cost']:.2f}",
                    "Front Ratio": f_res["ratio_str"],
                    "Back Ratio": back_ratio,
                    "Gem Prob (%)": f"{gem_prob:.1f}%",
                    "Expected EV ($)": f"${expected_ev:+.2f}",
                    "Decision": "🟢 BUY / LOUPE" if (both_passed and expected_ev > 10.0) else "🔴 AUTO-SKIP",
                    "eBay Link": item["url"]
                })
                preview_overlays[item["item_id"]] = (f_res["overlay_rgb"], b_overlay)
            except Exception as e:
                st.warning(f"Error analyzing item {item.get('item_id')}: {e}")

        if live_rows:
            st.dataframe(
                pd.DataFrame(live_rows),
                column_config={"eBay Link": st.column_config.LinkColumn("Open Listing")},
                use_container_width=True,
                hide_index=True
            )

            inspect_id = st.selectbox("Select Scanned eBay Item to View Dual Front/Back HUD:", list(preview_overlays.keys()))
            f_ov, b_ov = preview_overlays[inspect_id]
            ov_c1, ov_c2 = st.columns(2)
            ov_c1.image(f_ov, caption=f"Item #{inspect_id} — FRONT Anchor Scan", use_container_width=True)
            if b_ov is not None:
                ov_c2.image(b_ov, caption=f"Item #{inspect_id} — BACK Checklist Border Scan", use_container_width=True)

# ==============================================================================
# TAB 3: RECURSIVE MEMORY & EXPECTED VALUE (EV) MATRIX
# ==============================================================================
with tab_memory:
    st.subheader("🧠 Self-Recursive Memory Ledger")
    mem_col1, mem_col2 = st.columns(2)

    with mem_col1:
        st.markdown("#### 1. Fast Loop: Optical Caliper Overrides (`optical_feedback`)")
        if len(stats["opt_df"]) > 0:
            st.dataframe(stats["opt_df"], use_container_width=True, hide_index=True)
        else:
            st.caption("No manual caliper overrides recorded yet. Adjust a slider in Tab 1 and click 'Commit'.")

    with mem_col2:
        st.markdown("#### 2. Slow Loop: Logged PSA Grade Returns (`psa_outcomes`)")
        if len(stats["psa_df"]) > 0:
            st.dataframe(stats["psa_df"], use_container_width=True, hide_index=True)
        else:
            st.caption("No PSA slab returns logged yet. Use the Sidebar expander to log grades.")
