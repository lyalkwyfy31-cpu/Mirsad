import base64
import streamlit as st
import pystac_client as ps
import planetary_computer as pc
import odc.stac as stac
import numpy as np
import pandas as pd
from scipy.ndimage import label, binary_opening
import folium
from streamlit_folium import st_folium

st.set_page_config(
    page_title="منظومة مِرصاد الوطنية",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ منظومة (مِرصاد) لرصد المتغيرات المكانية والضريبية")
st.markdown("**النظام السيادي المتكامل (المرحلة 1 + 2 + 3): رصد فضائي، تقسيم عقاري ذكي، وحوكمة البلديات والمالية**")
st.divider()

REGIONS = {
    "ريف دمشق - قطاع صحنايا وأشرفيتها": (36.21, 33.42, 36.25, 33.45),
    "ريف دمشق - مدينة عدرا الصناعية": (36.50, 33.60, 36.55, 33.64),
    "ريف دمشق - قطاع قدسيا وضاحيتها": (36.20, 33.53, 36.24, 33.56),
    "حمص - مدينة حسياء الصناعية": (36.73, 34.40, 36.78, 34.44),
    "حلب - مدينة الشيخ نجار الصناعية": (37.22, 36.24, 37.27, 36.28),
}

st.sidebar.header("⚙️ لوحة تحكم الوزير والمحافظ")
sel_reg = st.sidebar.selectbox(
    "1. اختر القطاع الجغرافي:",
    list(REGIONS.keys())
)
min_ar = st.sidebar.slider(
    "2. الحد الأدنى لمساحة الرصد (م²):",
    100, 1000, 200, 100
)
fee_per_m2 = st.sidebar.slider(
    "3. تعرفة الرسم/الغرامة للمتر المربع (ل.س):",
    25000, 1000000, 150000, 25000
)
parcel_std = st.sidebar.slider(
    "4. معيار مساحة المقسم العقاري القياسي (م²):",
    150, 600, 300, 50
)

@st.cache_data(show_spinner=False)
def get_raw_detections(x1, y1, x2, y2):
    box = [x1, y1, x2, y2]
    b_api = "aHR0cHM6Ly9wbGFuZXRhcnljb21wdXRlci5taWNyb3NvZnQuY29tL2FwaS9zdGFjL3Yx"
    api_url = base64.b64decode(b_api).decode("utf-8")
    cat = ps.Client.open(api_url, modifier=pc.sign_inplace)
    
    def fetch_img(dt):
        s = cat.search(
            collections=["sentinel-2-l2a"],
            bbox=box,
            datetime=dt,
            query={"eo:cloud_cover": {"lt": 15}}
        )
        it = s.item_collection()
        d = stac.load(
            it,
            bands=["B08", "B11", "B04"],
            bbox=box,
            resolution=10
        )
        return d.median(dim="time")

    old = fetch_img("2025-05-01/2025-08-30")
    new = fetch_img("2026-06-01/2026-10-30")

    def calc_i(im):
        b11 = im["B11"].values.astype(float)
        b08 = im["B08"].values.astype(float)
        b04 = im["B04"].values.astype(float)
        i1 = (b11 - b08) / (b11 + b08 + 0.001)
        i2 = (b08 - b04) / (b08 + b04 + 0.001)
        return i1 - i2

    raw_chg = (calc_i(new) - calc_i(old)) > 0.16
    clean_chg = binary_opening(raw_chg, structure=np.ones((2, 2)))
    lbl, num = label(clean_chg)
    r, c = clean_chg.shape
    raw_list = []

    for i in range(1, num + 1):
        ar = int(np.sum(lbl == i)) * 100
        if ar >= 100:
            yi, xi = np.where(lbl == i)
            lt = y2 - (float(np.mean(yi)) / r) * (y2 - y1)
            ln = x1 + (float(np.mean(xi)) / c) * (x2 - x1)
            raw_list.append({
                "area": ar,
                "lat": round(lt, 5),
                "lon": round(ln, 5)
            })
    return raw_list

x1, y1, x2, y2 = REGIONS[sel_reg]

try:
    with st.spinner("🛰️ جاري المسح الفضائي وإجراء التقسيم العقاري الذكي..."):
        raw_data = get_raw_detections(x1, y1, x2, y2)
except Exception:
    raw_data = []

if "status_db" not in st.session_state:
    st.session_state["status_db"] = {}

filtered_rep = []
vid = 1
total_area = 0
total_fees = 0
total_parcels = 0

for item in raw_data:
    if item["area"] >= min_ar:
        ar = item["area"]
        fee = ar * fee_per_m2
        
        est_parcels = max(1, int(round(ar / float(parcel_std))))
        if ar <= 400:
            p_type = "مقسم بناء مفرد (1 عقار)"
        elif ar <= 1200:
            p_type = f"كتلة مقاسم متلاصقة ({est_parcels} مقاسم)"
        else:
            p_type = f"تجمع عمراني / هنغار ({est_parcels} مقاسم)"

        rec_id = f"MRS-{vid:03d}"
        cur_status = st.session_state["status_db"].get(
            rec_id, "🔴 مخالفة مرصودة (بانتظار البلدية)"
        )

        total_area += ar
        total_parcels += est_parcels
        if "مرخص" not in cur_status:
            total_fees += fee

        filtered_rep.append({
            "رقم الضبط": rec_id,
            "المساحة (م²)": ar,
            "التقسيم العقاري الذكي": p_type,
            "الرسوم التقديرية (ل.س)": f"{fee:,}",
            "خط العرض": item["lat"],
            "خط الطول": item["lon"],
            "الحالة التنفيذية": cur_status
        })
        vid += 1

df_res = pd.DataFrame(filtered_rep)

c1, c2, c3, c4 = st.columns(4)
c1.metric("🚨 الكتل المرصودة", f"{len(filtered_rep)} كتلة")
c2.metric("🏘️ إجمالي المقاسم العقارية", f"{total_parcels} مقسم")
c3.metric("📐 إجمالي المساحات", f"{total_area:,} م²")
c4.metric("💰 الوعاء الضريبي والبلدي", f"{total_fees:,} ل.س")

st.subheader(f"🗺️ الخريطة الفضائية التفاعلية والتقسيم العقاري: {sel_reg}")

b_hyb = "aHR0cHM6Ly9tdDEuZ29vZ2xlLmNvbS92dC9seXJzPXkmeD17eH0meT17eX0mej17en0maGw9YXI="
g_hybrid = base64.b64decode(b_hyb).decode("utf-8")

clat = (y1 + y2) / 2.0
clon = (x1 + x2) / 2.0

m = folium.Map(
    location=[clat, clon],
    zoom_start=14,
    tiles=g_hybrid,
    attr="Google Hybrid"
)

for row in filtered_rep:
    st_txt = row["الحالة التنفيذية"]
    if "مرخص" in st_txt:
        c_col, f_col = "#00cc00", "#00ff00"
    elif "تم التحصيل" in st_txt:
        c_col, f_col = "#0066ff", "#0088ff"
    elif "دورية" in st_txt:
        c_col, f_col = "#ff8800", "#ffaa00"
    else:
        c_col, f_col = "#ff0000", "#ffff00"

    html = f"""
    <div style="direction:rtl; text-align:right; font-family:Tahoma; width:200px;">
        <b style="color:darkred;">🛰️ ضبط منظومة مِرصاد</b><hr style="margin:4px 0;">
        <b>رقم العقار/الضبط:</b> {row['رقم الضبط']}<br>
        <b>المساحة الكلية:</b> {row['المساحة (م²)']} م²<br>
        <b>التقسيم العقاري:</b> {row['التقسيم العقاري الذكي']}<br>
        <b>الرسم المقدر:</b> {row['الرسوم التقديرية (ل.س)']} ل.س<br>
        <b>الحالة:</b> <b>{st_txt}</b>
    </div>
    """
    folium.CircleMarker(
        location=[row["خط العرض"], row["خط الطول"]],
        radius=9,
        color=c_col,
        weight=2,
        fill=True,
        fill_color=f_col,
        fill_opacity=0.35,
        popup=folium.Popup(html, max_width=230),
        tooltip=f"عقار {row['رقم الضبط']} | {row['المساحة (م²)']} م² ({row['التقسيم العقاري الذكي']})"
    ).add_to(m)

st_folium(m, width=1100, height=460, returned_objects=[])

st.divider()
st.subheader("🏛️ المرحلة الثالثة: غرفة العمليات التنفيذية (ربط البلديات، الدوريات، والتحصيل المالي)")
st.markdown("تتيح هذه الوحدة للسيد المحافظ ورؤساء البلديات ومراقبي المالية معالجة الضبوط الفضائية لحظياً وتغيير حالتها ولونها على الخريطة:")

if len(filtered_rep) > 0:
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        id_list = [r["رقم الضبط"] for r in filtered_rep]
        sel_id = st.selectbox("1. اختر رقم الضبط الفضائي لمعالجته:", id_list)
    with col_b:
        new_action = st.selectbox(
            "2. الإجراء الحكومي المتخذ:",
            [
                "🔴 مخالفة مرصودة (بانتظار البلدية)",
                "🟠 تم إرسال دورية مراقب (مقفول بالـ GPS)",
                "🟢 بناء نظامي مرخص (تم رفع الرخصة)",
                "🔵 تم تحصيل الغرامة والتسوية في المالية"
            ]
        )
    with col_c:
        st.write("")
        st.write("")
        if st.button("✅ اعتماد الإجراء وتحديث الخريطة فوراً"):
            st.session_state["status_db"][sel_id] = new_action
            st.rerun()

st.subheader("📋 السجل الرسمي للتقسيم العقاري والتكليف المالي")
if len(df_res) > 0:
    csv_data = df_res.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="📥 تحميل الكشف العقاري والمالي الرسمي (Excel / CSV)",
        data=csv_data,
        file_name="Mirsad_Full_Report.csv",
        mime="text/csv"
    )
    st.dataframe(df_res, use_container_width=True)
else:
    st.success("✅ لا توجد تكتلات عمرانية تتجاوز المساحة المحددة في هذا القطاع.")
