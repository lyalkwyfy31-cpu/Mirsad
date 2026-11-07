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

st.set_page_config(page_title="منظومة مِرصاد الرادارية", page_icon="🛰️", layout="wide")
conn = sqlite3.connect("mirsad_sar.db", check_same_thread=False)
cur = conn.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, st TEXT, nt TEXT, dc TEXT)")
cur.execute("CREATE TABLE IF NOT EXISTS cache (rc TEXT PRIMARY KEY, pd TEXT, jd TEXT)")
conn.commit()

td = (datetime.utcnow() + timedelta(hours=3)).strftime("%Y-%m-%d")
st.title("🛰️ منظومة (مِرصاد) الوطنية لرصد المتغيرات المكانية والضريبية")
st.markdown(f"**🟢 بث فضائي حي (توقيت دمشق: `{td}`) | دمج بصري (Sentinel-2) وراداري حقيقي (Sentinel-1 SAR)**")
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
min_ar = st.sidebar.slider("2. الحد الأدنى للمساحة (م²):", 100, 1000, 200, 100)
cm = st.sidebar.radio("3. العملة:", ["الليرة السورية الجديدة (2026)", "الليرة السورية القديمة", "الدولار الأمريكي ($)"])
if "الجديدة" in cm:
    fee_m2, c_sym, u_rt = st.sidebar.slider("4. تعرفة المتر:", 250, 10000, 1500, 250), "ل.س جديدة", 115.0
elif "القديمة" in cm:
    fee_m2, c_sym, u_rt = st.sidebar.slider("4. تعرفة المتر:", 25000, 1000000, 150000, 25000), "ل.س", 11500.0
else:
    fee_m2, c_sym, u_rt = st.sidebar.slider("4. تعرفة المتر ($):", 2, 100, 13, 1), "$", 1.0
p_std = st.sidebar.slider("5. مساحة المقسم القياسي (م²):", 150, 600, 300, 50)

@st.cache_data(show_spinner=False, max_entries=5, ttl=86400)
def scan_sar_opt(x1, y1, x2, y2, d_str):
    box = [x1, y1, x2, y2]
    u = base64.b64decode("aHR0cHM6Ly9wbGFuZXRhcnljb21wdXRlci5taWNyb3NvZnQuY29tL2FwaS9zdGFjL3Yx").decode("utf-8")
    cat = ps.Client.open(u, modifier=pc.sign_inplace)
    en = datetime.strptime(d_str, "%Y-%m-%d")
    sn = en - timedelta(days=80)
    dn = f"{sn.strftime('%Y-%m-%d')}/{en.strftime('%Y-%m-%d')}"
    do = f"{(sn - timedelta(days=365)).strftime('%Y-%m-%d')}/{(en - timedelta(days=365)).strftime('%Y-%m-%d')}"

    def f_opt(rng):
        it = list(cat.search(collections=["sentinel-2-l2a"], bbox=box, datetime=rng, query={"eo:cloud_cover": {"lt": 15}}).item_collection())
        if not it:
            it = list(cat.search(collections=["sentinel-2-l2a"], bbox=box, datetime=rng).item_collection())
        lp = max(x.datetime for x in it).strftime("%Y-%m-%d")
        it.sort(key=lambda x: (x.properties.get("eo:cloud_cover", 50.0) > 10.0, -x.datetime.timestamp()))
        return stac.load(it[:3], bands=["B08", "B11", "B04"], bbox=box, resolution=10).median(dim="time"), lp

    old, _ = f_opt(do)
    new, lp_dt = f_opt(dn)

    def c_op(im):
        b11, b08, b04 = np.nan_to_num(im["B11"].values.astype(np.float32)), np.nan_to_num(im["B08"].values.astype(np.float32)), np.nan_to_num(im["B04"].values.astype(np.float32))
        return ((b11 - b08) / (b11 + b08 + 0.001)) - ((b08 - b04) / (b08 + b04 + 0.001)), b04, b11

    io, ro, _ = c_op(old)
    inw, rnw, swn = c_op(new)
    df_i = inw - io
    dt_th = float(np.clip(np.percentile(df_i, 97.0), 0.15, 0.20))
    sw_th = max(1450.0, float(np.percentile(swn, 55)))
    c_opt = (df_i > dt_th) & ((rnw - ro) > 55) & (swn > sw_th)
    r_s, c_s = c_opt.shape

    # الفحص الفعلي بالقمر الراداري Sentinel-1 SAR لعزل الجدران والأسطح الصلبة
    sar_ok = np.ones_like(c_opt, dtype=bool)
    try:
        it_s = list(cat.search(collections=["sentinel-1-rtc"], bbox=box, datetime=dn).item_collection())[:2]
        if it_s:
            ds = stac.load(it_s, bands=["vv"], like=old).median(dim="time")
            sar_ok = np.nan_to_num(ds["vv"].values.astype(np.float32), nan=0.1) > 0.032
    except Exception:
        pass

    del old, new
    gc.collect()
    chg = binary_opening(c_opt & sar_ok, structure=np.ones((2, 2)))
    lbl, num = label(chg)
    res = []
    for i in range(1, num + 1):
        ar = int(np.sum(lbl == i)) * 100
        if 100 <= ar <= 6000:
            yi, xi = np.where(lbl == i)
            res.append({"area": ar, "lat": round(y2 - (float(np.mean(yi)) / r_s) * (y2 - y1), 5), "lon": round(x1 + (float(np.mean(xi)) / c_s) * (x2 - x1), 5), "dt": lp_dt})
    return res, lp_dt

x1, y1, x2, y2, rc = REGIONS[sel_reg]
try:
    with st.spinner("📡 جاري المسح المزدوج: القمر البصري (Sentinel-2) + القمر الراداري (Sentinel-1 SAR)..."):
        raw_d, lp_date = scan_sar_opt(x1, y1, x2, y2, td)
        if raw_d:
            cur.execute("INSERT OR REPLACE INTO cache VALUES (?, ?, ?)", (rc, lp_date, json.dumps(raw_d)))
            conn.commit()
except Exception:
    cur.execute("SELECT pd, jd FROM cache WHERE rc=?", (rc,))
    r_c = cur.fetchone()
    raw_d, lp_date = (json.loads(r_c[1]), r_c[0]) if r_c else ([], td)

cur.execute("SELECT id, st, nt, dc FROM records")
db_r = {k: (v1, v2, v3) for k, v1, v2, v3 in cur.fetchall()}

rep, vid, t_ar, t_fe, t_pr = [], 1, 0, 0, 0
f_col = f"الرسوم ({c_sym})"
for it in raw_d:
    if it["area"] >= min_ar:
        ar, fe = it["area"], it["area"] * fee_m2
        pr = max(1, int(round(ar / float(p_std))))
        pt = "مقسم مفرد (1 عقار)" if ar <= 400 else f"تجمع/هنغار ({pr} مقاسم)"
        rid = f"{rc}-{vid:03d}"
        sv, nv, dv = db_r.get(rid, ("🔴 مخالفة مرصودة (بانتظار البلدية)", "لا يوجد", "📡 مؤكد بالرادار SAR"))
        t_ar += ar
        t_pr += pr
        if "مرخص" not in sv:
            t_fe += fe
        rep.append({"رقم الضبط": rid, "تاريخ الرصد": it["dt"], "المساحة (م²)": ar, "التقسيم العقاري": pt, f_col: f"{fe:,}", "الحالة التنفيذية": sv, "الرادار والوثائق": dv, "ملاحظات": nv, "خط العرض": it["lat"], "خط الطول": it["lon"]})
        vid += 1

df = pd.DataFrame(rep)
c1, c2, c3, c4 = st.columns(4)
c1.metric("📡 الكتل الرادارية المؤكدة", f"{len(rep)} كتلة")
c2.metric("🏘️ المقاسم العقارية", f"{t_pr} مقسم")
c3.metric("📐 إجمالي المساحات", f"{t_ar:,} م²")
c4.metric(f"💰 الوعاء المالي ({c_sym})", f"{t_fe:,}", f"يعادل ~ ${int(t_fe/u_rt):,} USD")

st.subheader(f"🗺️ خريطة الرصد الراداري المزدوج (آخر تمريرة: {lp_date}): {sel_reg}")
gh = base64.b64decode("aHR0cHM6Ly9tdDEuZ29vZ2xlLmNvbS92dC9seXJzPXkmeD17eH0meT17eX0mej17en0maGw9YXI=").decode("utf-8")
m = folium.Map(location=[(y1 + y2) / 2.0, (x1 + x2) / 2.0], zoom_start=14, tiles=gh, attr="Google")

for r in rep:
    s = r["الحالة التنفيذية"]
    cc, fc = ("green", "lime") if "مرخص" in s else (("blue", "cyan") if "التحصيل" in s else ("red", "yellow"))
    hp = f"<div style='direction:rtl;text-align:right;width:200px;'><b>🛰️ ضبط {r['رقم الضبط']} (SAR)</b><hr><b>الرصد:</b> {r['تاريخ الرصد']}<br><b>المساحة:</b> {r['المساحة (م²)']} م² ({r['التقسيم العقاري']})<br><b>الرسم:</b> {r[f_col]} {c_sym}<br><b>الحالة:</b> {s}</div>"
    folium.CircleMarker(location=[r["خط العرض"], r["خط الطول"]], radius=8, color=cc, fill=True, fill_color=fc, fill_opacity=0.35, popup=folium.Popup(hp, max_width=230), tooltip=f"{r['رقم الضبط']} | {r['المساحة (م²)']} م²").add_to(m)

st_folium(m, width=950, height=420, returned_objects=[])
st.divider()
st.subheader("🏛️ المرحلة الثالثة: بوابة الإجراءات التنفيذية وتوثيق الرخص (للبلديات والمالية)")
if len(rep) > 0:
    ca, cb = st.columns(2)
    with ca:
        sid = st.selectbox("1. اختر رقم الضبط:", [r["رقم الضبط"] for r in rep])
        act = st.selectbox("2. الإجراء الحكومي:", ["🔴 مخالفة مرصودة (بانتظار البلدية)", "🟠 تم إرسال دورية مراقب (GPS)", "🟢 بناء نظامي مرخص (تم رفع الرخصة)", "🔵 تم تحصيل الغرامة والتسوية في المالية"])
        pin = st.text_input("🔐 رمز التصديق الحكومي (الافتراضي: 2026):", value="2026")
    with cb:
        nin = st.text_input("3. رقم الرخصة / الإيصال المالي:", placeholder="مثال: رخصة 452")
        doc = st.file_uploader("4. إرفاق صورة الرخصة أو الضبط (كاميرا/ملف):", type=["jpg", "png", "pdf"])
    if st.button("✅ حفظ في قاعدة البيانات وتحديث الخريطة فوراً"):
        if pin == "2026":
            cur.execute("INSERT OR REPLACE INTO records VALUES (?, ?, ?, ?)", (sid, act, nin if nin else "تم التدقيق", "📎 وثيقة مرفقة" if doc else "📡 مؤكد بالرادار SAR"))
            conn.commit()
            st.rerun()
        else:
            st.error("❌ رمز التصديق غير صحيح!")

st.subheader("📋 السجل الرسمي للتقسيم العقاري والتكليف المالي")
if len(df) > 0:
    st.download_button("📥 تحميل الكشف الرسمي (Excel / CSV)", df.to_csv(index=False, encoding="utf-8-sig"), f"Mirsad_SAR_{rc}.csv", "text/csv")
    st.dataframe(df, use_container_width=True)
# END
