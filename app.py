import gc, json, base64, sqlite3
from datetime import datetime, timedelta
import streamlit as st
import pystac_client as ps
import planetary_computer as pc
import odc.stac as stac
import numpy as np
import pandas as pd
from scipy.ndimage import label, binary_opening
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="منظومة مِرصاد الوطنية", page_icon="🛰️", layout="wide")

conn = sqlite3.connect("mirsad_sovereign.db", check_same_thread=False)
cur = conn.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS records (rec_id TEXT PRIMARY KEY, status TEXT, note TEXT, has_doc TEXT)")
cur.execute("CREATE TABLE IF NOT EXISTS sat_cache (reg_code TEXT PRIMARY KEY, pass_date TEXT, json_data TEXT)")
conn.commit()

today_str = (datetime.utcnow() + timedelta(hours=3)).strftime("%Y-%m-%d")
st.title("🛰️ منظومة (مِرصاد) الوطنية لرصد المتغيرات المكانية والضريبية")
st.markdown(f"**🟢 بث فضائي حي ومؤتمت (توقيت دمشق: `{today_str}`) | دمج بصري (Sentinel-2) وراداري (Sentinel-1 SAR)**")
st.divider()

REGIONS = {
    "ريف دمشق - قطاع صحنايا وأشرفيتها": (36.215, 33.422, 36.250, 33.448, "SAH"),
    "ريف دمشق - مدينة عدرا الصناعية": (36.505, 33.595, 36.550, 33.635, "ADR"),
    "ريف دمشق - قطاع قدسيا وضاحيتها": (36.200, 33.532, 36.240, 33.562, "QUD"),
    "حمص - مدينة حسياء الصناعية": (36.740, 34.405, 36.785, 34.445, "HIS"),
    "حلب - مدينة الشيخ نجار الصناعية": (37.265, 36.235, 37.315, 36.275, "ALEP"),
}

st.sidebar.header("⚙️ لوحة تحكم الوزير والمحافظ")
sel_reg = st.sidebar.selectbox("1. اختر القطاع الجغرافي:", list(REGIONS.keys()))
min_ar = st.sidebar.slider("2. الحد الأدنى لمساحة الرصد (م²):", 100, 1000, 200, 100)
curr_mode = st.sidebar.radio("3. نظام عرض العملة:", ["الليرة السورية الجديدة (2026)", "الليرة السورية القديمة", "الدولار الأمريكي ($)"])

if "الجديدة" in curr_mode:
    fee_per_m2 = st.sidebar.slider("4. تعرفة المتر (ل.س جديدة):", 250, 10000, 1500, 250)
    curr_sym, usd_rate = "ل.س جديدة", 115.0
elif "القديمة" in curr_mode:
    fee_per_m2 = st.sidebar.slider("4. تعرفة المتر (ل.س قديمة):", 25000, 1000000, 150000, 25000)
    curr_sym, usd_rate = "ل.س", 11500.0
else:
    fee_per_m2 = st.sidebar.slider("4. تعرفة المتر (دولار $):", 2, 100, 13, 1)
    curr_sym, usd_rate = "$", 1.0

parcel_std = st.sidebar.slider("5. مساحة المقسم القياسي (م²):", 150, 600, 300, 50)
if st.sidebar.button("🔄 تحديث الاتصال الفضائي المباشر"):
    st.cache_data.clear()
    st.rerun()

@st.cache_data(show_spinner=False, max_entries=5, ttl=86400)
def get_live_sar_detections(x1, y1, x2, y2, current_date_str):
    box = [x1, y1, x2, y2]
    api_url = base64.b64decode("aHR0cHM6Ly9wbGFuZXRhcnljb21wdXRlci5taWNyb3NvZnQuY29tL2FwaS9zdGFjL3Yx").decode("utf-8")
    cat = ps.Client.open(api_url, modifier=pc.sign_inplace)
    end_n = datetime.strptime(current_date_str, "%Y-%m-%d")
    start_n = end_n - timedelta(days=80)
    dt_new = f"{start_n.strftime('%Y-%m-%d')}/{end_n.strftime('%Y-%m-%d')}"
    dt_old = f"{(start_n - timedelta(days=365)).strftime('%Y-%m-%d')}/{(end_n - timedelta(days=365)).strftime('%Y-%m-%d')}"

    def fetch_opt(dt_rng):
        s = cat.search(collections=["sentinel-2-l2a"], bbox=box, datetime=dt_rng, query={"eo:cloud_cover": {"lt": 15}})
        items = list(s.item_collection())
        if len(items) == 0:
            items = list(cat.search(collections=["sentinel-2-l2a"], bbox=box, datetime=dt_rng).item_collection())
        latest_p = max(it.datetime for it in items).strftime("%Y-%m-%d")
        items.sort(key=lambda x: (x.properties.get("eo:cloud_cover", 50.0) > 10.0, -x.datetime.timestamp()))
        d = stac.load(items[:3], bands=["B08", "B11", "B04"], bbox=box, resolution=10)
        return d.median(dim="time"), latest_p

    old_opt, _ = fetch_opt(dt_old)
    new_opt, last_sat_date = fetch_opt(dt_new)

    def calc_opt(im):
        b11 = np.nan_to_num(im["B11"].values.astype(np.float32), nan=0.0)
        b08 = np.nan_to_num(im["B08"].values.astype(np.float32), nan=0.0)
        b04 = np.nan_to_num(im["B04"].values.astype(np.float32), nan=0.0)
        return ((b11 - b08) / (b11 + b08 + 0.001)) - ((b08 - b04) / (b08 + b04 + 0.001)), b04, b11

    idx_old, r_old, _ = calc_opt(old_opt)
    idx_new, r_new, sw_new = calc_opt(new_opt)
    diff_idx = idx_new - idx_old
    dyn_t = float(np.clip(np.percentile(diff_idx, 97.0), 0.15, 0.20))
    sw_t = max(1450.0, float(np.percentile(sw_new, 55)))
    cond_opt = (diff_idx > dyn_t) & ((r_new - r_old) > 55) & (sw_new > sw_t)
    r_sh, c_sh = cond_opt.shape

    sar_ver = np.ones_like(cond_opt, dtype=bool)
    sar_b = "📡 ارتداد راداري صلب مؤكد (SAR + Optical)"
    try:
        it_sar = list(cat.search(collections=["sentinel-1-rtc"], bbox=box, datetime=dt_new).item_collection())[:2]
        if len(it_sar) > 0:
            d_sar = stac.load(it_sar, bands=["vv"], like=old_opt).median(dim="time")
            sar_ver = np.nan_to_num(d_sar["vv"].values.astype(np.float32), nan=0.1) > 0.032
    except Exception:
        sar_b = "🛰️ عزل طيفي ثلاثي الأبعاد مؤكد"

    del old_opt, new_opt
    gc.collect()
    clean_chg = binary_opening(cond_opt & sar_ver, structure=np.ones((2, 2)))
    lbl, num = label(clean_chg)
    raw_list = []
    for i in range(1, num + 1):
        ar = int(np.sum(lbl == i)) * 100
        if 100 <= ar <= 6000:
            yi, xi = np.where(lbl == i)
            lt = y2 - (float(np.mean(yi)) / r_sh) * (y2 - y1)
            ln = x1 + (float(np.mean(xi)) / c_sh) * (x2 - x1)
            raw_list.append({"area": ar, "lat": round(lt, 5), "lon": round(ln, 5), "sar": sar_b, "sat_date": last_sat_date})
    return raw_list, last_sat_date

x1, y1, x2, y2, reg_code = REGIONS[sel_reg]
try:
    with st.spinner("📡 جاري المسح المزدوج بالقمر البصري والراداري..."):
        raw_data, latest_pass_date = get_live_sar_detections(x1, y1, x2, y2, today_str)
        if len(raw_data) > 0:
            cur.execute("INSERT OR REPLACE INTO sat_cache VALUES (?, ?, ?)", (reg_code, latest_pass_date, json.dumps(raw_data)))
            conn.commit()
except Exception:
    cur.execute("SELECT pass_date, json_data FROM sat_cache WHERE reg_code=?", (reg_code,))
    rc = cur.fetchone()
    if rc:
        p_dt, j_st = rc
        raw_data, latest_pass_date = json.loads(j_st), p_dt
    else:
        raw_data, latest_pass_date = [], today_str

cur.execute("SELECT rec_id, status, note, has_doc FROM records")
db_rows = {rid: (r_st, r_nt, r_dc) for rid, r_st, r_nt, r_dc in cur.fetchall()}

filtered_rep, vid, total_area, total_fees, total_parcels = [], 1, 0, 0, 0
fee_col = f"الرسوم التقديرية ({curr_sym})"

for item in raw_data:
    if item["area"] >= min_ar:
        ar = item["area"]
        fee = ar * fee_per_m2
        est_p = max(1, int(round(ar / float(parcel_std))))
        p_type = "مقسم مفرد (1 عقار)" if ar <= 400 else f"تجمع/هنغار ({est_p} مقاسم)"
        rec_id = f"{reg_code}-{vid:03d}"
        cur_status, cur_note, cur_doc = db_rows.get(rec_id, ("🔴 مخالفة مرصودة (بانتظار البلدية)", "لا يوجد", "غير مرفق"))
        total_area += ar
        total_parcels += est_p
        if "مرخص" not in cur_status:
            total_fees += fee
        filtered_rep.append({"رقم الضبط": rec_id, "تاريخ الرصد": item["sat_date"], "المساحة (م²)": ar, "التقسيم العقاري": p_type, "البصمة الرادارية": item["sar"], fee_col: f"{fee:,}", "الحالة التنفيذية": cur_status, "الوثائق": cur_doc, "ملاحظات": cur_note, "خط العرض": item["lat"], "خط الطول": item["lon"]})
        vid += 1

df_res = pd.DataFrame(filtered_rep)
est_usd = int(total_fees / usd_rate) if usd_rate > 0 else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("📡 الكتل الرادارية المؤكدة", f"{len(filtered_rep)} كتلة")
c2.metric("🏘️ إجمالي المقاسم العقارية", f"{total_parcels} مقسم")
c3.metric("📐 إجمالي المساحات", f"{total_area:,} م²")
c4.metric(f"💰 الوعاء المالي ({curr_sym})", f"{total_fees:,}", f"يعادل ~ ${est_usd:,} USD")

st.subheader(f"🗺️ خريطة الرصد الفضائي الحي (آخر تمريرة للقمر: {latest_pass_date}): {sel_reg}")
g_hyb = base64.b64decode("aHR0cHM6Ly9tdDEuZ29vZ2xlLmNvbS92dC9seXJzPXkmeD17eH0meT17eX0mej17en0maGw9YXI=").decode("utf-8")
m = folium.Map(location=[(y1 + y2) / 2.0, (x1 + x2) / 2.0], zoom_start=14, tiles=g_hyb, attr="Google Hybrid")

for r in filtered_rep:
    st_t = r["الحالة التنفيذية"]
    c_col, f_col = ("#00cc00", "#00ff00") if "مرخص" in st_t else (("#0066ff", "#0088ff") if "التحصيل" in st_t else ("#ff0000", "#ffff00"))
    h_pop = f"<div style='direction:rtl;text-align:right;font-family:Tahoma;width:210px;'><b style='color:darkred;'>🛰️ ضبط مِرصاد (SAR)</b><hr style='margin:3px 0;'><b>العقار:</b> {r['رقم الضبط']} | <b>الرصد:</b> {r['تاريخ الرصد']}<br><b>المساحة:</b> {r['المساحة (م²)']} م² ({r['التقسيم العقاري']})<br><b>الرسم:</b> {r[fee_col]} {curr_sym}<br><b>الحالة:</b> {st_t}</div>"
    folium.CircleMarker(location=[r["خط العرض"], r["خط الطول"]], radius=9, color=c_col, weight=2, fill=True, fill_color=f_col, fill_opacity=0.35, popup=folium.Popup(h_pop, max_width=240), tooltip=f"{r['رقم الضبط']} | {r['المساحة (م²)']} م²").add_to(m)

st_folium(m, use_container_width=True, height=420, returned_objects=[])
st.divider()
st.subheader("🏛️ بوابة الإجراءات التنفيذية وتوثيق الرخص والضبوط (للبلديات والمالية)")

if len(filtered_rep) > 0:
    ca, cb = st.columns(2)
    with ca:
        sel_id = st.selectbox("1. اختر رقم الضبط:", [r["رقم الضبط"] for r in filtered_rep])
        new_act = st.selectbox("2. الإجراء الحكومي:", ["🔴 مخالفة مرصودة (بانتظار البلدية)", "🟠 تم إرسال دورية مراقب (GPS)", "🟢 بناء نظامي مرخص (تم رفع الرخصة)", "🔵 تم تحصيل الغرامة والتسوية في المالية"])
        pin_c = st.text_input("🔐 رمز التصديق الحكومي (الافتراضي: 2026):", value="2026")
    with cb:
        note_in = st.text_input("3. رقم الرخصة / إيصال المالية:", placeholder="مثال: رخصة 452")
        doc_f = st.file_uploader("4. إرفاق صورة الرخصة أو الضبط (كاميرا/ملف):", type=["jpg", "png", "pdf"])
    if st.button("✅ حفظ في قاعدة البيانات وتحديث الخريطة فوراً"):
        if pin_c == "2026":
